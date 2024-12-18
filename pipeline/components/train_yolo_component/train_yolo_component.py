# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os
import sys
import time

from kubernetes import client, config

import kfp

from kfp import compiler
from kfp import dsl

from kfp import kubernetes

from src.train_yolo import train_yolo

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    model_name: str = "yolov8n", 
    image_size: int = 640, 
    epochs: int = 2, 
    batch_size: int = 2,
    optimizer: str = "Adam", #  [Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]
    learning_rate: float = 0.001,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    label_smoothing: float = 0.1,
    experiment_name: str = "uno-cards-v1.2-0",
    run_name: str = "uno-cards",
    tracking_uri: str = "http://mlflow-server:8080",
    images_dataset_name: str = "uno-cards-v1.2",
    images_datasets_root_folder: str = "datasets",
    images_dataset_yaml: str = "data.yaml",
    models_root_folder: str = "models",
    images_dataset_pvc_name: str = "images-datasets-pvc",
    images_dataset_pvc_size_in_gi: int = 5,
    force_clean: bool = False):

    root_mount_path = "/opt/app/src"

    # Train the model
    train_model_task = train_yolo(
        model_name=model_name, 
        image_size=image_size, 
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        label_smoothing=label_smoothing,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        root_mount_path=root_mount_path,
        images_dataset_name=images_dataset_name,
        images_datasets_root_folder=images_datasets_root_folder,
        images_dataset_yaml=images_dataset_yaml,
        models_root_folder=models_root_folder
    ).set_caching_options(False)

if __name__ == '__main__':
    import time
    import sys
    import os

    from kfp import compiler

    from kubernetes import client, config

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