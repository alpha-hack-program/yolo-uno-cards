import os
import sys

import kfp

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath

from kfp import kubernetes

from kubernetes import client, config

@dsl.component(
    base_image='python:3.11',  # Use an appropriate base image
    packages_to_install=['kubernetes==23.6.0'],  # Install the required packages
)
def pipeline():
    # Define the function to list PVCs
    def list_pvcs(namespace: str):
        import os
        from kubernetes import client, config

        # Create configuration using the service account token and namespace
        configuration = client.Configuration()

        # Load the service account token and namespace from the mounted paths
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        namespace_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'

        # Read the token
        with open(token_path, 'r') as token_file:
            token = token_file.read().strip()

        # Read the namespace
        with open(namespace_path, 'r') as namespace_file:
            namespace = namespace_file.read().strip()

        # Configure the client
        # curl -k --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
        #   -H "Authorization: Bearer $(cat /var/run/secrets/kubernetes.io/serviceaccount/token)"  \
        #   https://kubernetes.default.svc/api/v1/namespaces/iniciativa-2/persistentvolumeclaims/images-datasets-pvc
        kubernetes_host = f"https://{os.getenv('KUBERNETES_SERVICE_HOST', 'kubernetes.default.svc')}:{os.getenv('KUBERNETES_SERVICE_PORT', '443')}"
        print(f"kubernetes_host: {kubernetes_host}")
        configuration.host = 'https://kubernetes.default.svc'
        configuration.verify_ssl = True
        configuration.ssl_ca_cert = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
        configuration.api_key['authorization'] = token
        configuration.api_key_prefix['authorization'] = 'Bearer'

        # Print all the configuration settings
        print("Configuration settings:")
        for attr, value in vars(configuration).items():
            print(f"{attr}: {value}")

        print("Configured Kubernetes API Host:", configuration.host)
        # Create an API client with the configuration
        api_client = client.ApiClient(configuration)
        print("API Client Host:", api_client.configuration.host)

        # Use the CoreV1 API to list PVCs
        v1 = client.CoreV1Api(api_client)
        pvc_list = v1.list_namespaced_persistent_volume_claim(namespace)
        
        # Print PVC names and statuses
        for pvc in pvc_list.items:
            print(f"PVC Name: {pvc.metadata.name}, Status: {pvc.status.phase}")

    # Call the function to list PVCs
    list_pvcs("default")  # You can replace "default" with a variable if needed

def get_pipeline_by_name(client: kfp.Client, pipeline_name: str):
    import json

    # Define filter predicates
    filter_spec = json.dumps({
        "predicates": [{
            "key": "display_name",
            "operation": "EQUALS",
            "stringValue": pipeline_name,
        }]
    })

    # List pipelines with the specified filter
    pipelines = client.list_pipelines(filter=filter_spec)

    if not pipelines.pipelines:
        return None
    for pipeline in pipelines.pipelines:
        if pipeline.display_name == pipeline_name:
            return pipeline

    return None

# Get the service account token or return None
def get_token():
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get the route host for the specified route name in the specified namespace
def get_route_host(route_name: str):
    # Load in-cluster Kubernetes configuration but if it fails, load local configuration
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()

    # Get the current namespace
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
        namespace = f.read().strip()

    # Create Kubernetes API client
    api_instance = client.CustomObjectsApi()

    try:
        # Retrieve the route object
        route = api_instance.get_namespaced_custom_object(
            group="route.openshift.io",
            version="v1",
            namespace=namespace,
            plural="routes",
            name=route_name
        )

        # Extract spec.host field
        route_host = route['spec']['host']
        return route_host
    
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    import time

    pipeline_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_package_path
    )

    # Take token and kfp_endpoint as optional command-line arguments
    token = sys.argv[1] if len(sys.argv) > 1 else None
    kfp_endpoint = sys.argv[2] if len(sys.argv) > 2 else None

    if not token:
        print("Token endpoint not provided finding it automatically.")
        token = get_token()

    if not kfp_endpoint:
        print("KFP endpoint not provided finding it automatically.")
        kfp_endpoint = get_route_host(route_name="ds-pipeline-dspa")

    # Pipeline name
    pipeline_name = os.path.basename(__file__).replace('.py', '')

    # If both kfp_endpoint and token are provided, upload the pipeline
    if kfp_endpoint and token:
        client = kfp.Client(host=kfp_endpoint, existing_token=token)

        # If endpoint doesn't have a protocol (http or https), add https
        if not kfp_endpoint.startswith("http"):
            kfp_endpoint = f"https://{kfp_endpoint}"

        try:
            # Get the pipeline by name
            print(f"Pipeline name: {pipeline_name}")
            existing_pipeline = get_pipeline_by_name(client, pipeline_name)
            if existing_pipeline:
                print(f"Pipeline {existing_pipeline.pipeline_id} already exists. Uploading a new version.")
                # Upload a new version of the pipeline with a version name equal to the pipeline package path plus a timestamp
                pipeline_version_name=f"{pipeline_name}-{int(time.time())}"
                client.upload_pipeline_version(
                    pipeline_package_path=pipeline_package_path,
                    pipeline_id=existing_pipeline.pipeline_id,
                    pipeline_version_name=pipeline_version_name
                )
                print(f"Pipeline version uploaded successfully to {kfp_endpoint}")
            else:
                print(f"Pipeline {pipeline_name} does not exist. Uploading a new pipeline.")
                print(f"Pipeline package path: {pipeline_package_path}")
                # Upload the compiled pipeline
                client.upload_pipeline(
                    pipeline_package_path=pipeline_package_path,
                    pipeline_name=pipeline_name
                )
                print(f"Pipeline uploaded successfully to {kfp_endpoint}")
        except Exception as e:
            print(f"Failed to upload the pipeline: {e}")
    else:
        print("KFP endpoint or token not provided. Skipping pipeline upload.")