import os
import yaml

from typing import Optional

import boto3
import botocore

from kubernetes import client as k8s_cli, config as k8s_conf
from kfp import client as kfp_cli

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
region_name = os.environ.get('AWS_DEFAULT_REGION')
bucket_name = os.environ.get('AWS_S3_BUCKET')
experiment_reports_folder = os.environ.get('EXPERIMENT_REPORTS_FOLDER_S3_KEY')

# Get token path from environment or default to kubernetes token location
TOKEN_PATH = os.environ.get("TOKEN_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/token")

# Get the service account token or return None
def get_token():
    try:
        with open(TOKEN_PATH, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Get the route host for the specified route name in the specified namespace
def get_route_host(route_name: str):
    # Load in-cluster Kubernetes configuration but if it fails, load local configuration
    try:
        k8s_conf.load_incluster_config()
    except k8s_conf.config_exception.ConfigException:
        k8s_conf.load_kube_config()

    # Get token path from environment or default to kubernetes token location
    NAMESPACE_PATH = os.environ.get("NAMESPACE_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/namespace")

    # Get the current namespace
    with open(NAMESPACE_PATH, "r") as f:
        namespace = f.read().strip()

    print(f"namespace: {namespace}")

    # Create Kubernetes API client
    api_instance = k8s_cli.CustomObjectsApi()

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

def get_pipeline_id_by_name(client: kfp_cli.Client, pipeline_name: str):
    return client.get_pipeline_id(pipeline_name)

def get_pipeline(client: kfp_cli.Client, pipeline_id: str):
    return client.get_pipeline(pipeline_id)

def get_latest_pipeline_version_id(client: kfp_cli.Client, pipeline_id: str) -> Optional[str]:
    pipeline_versions = client.list_pipeline_versions(
        pipeline_id=pipeline_id,
        sort_by="created_at desc",
        page_size=10
    )
    if pipeline_versions and len(pipeline_versions.pipeline_versions) >= 1:
        # Order pipeline_versions by created_at in descending order and return the first one
        pipeline_versions.pipeline_versions.sort(key=lambda x: x.created_at, reverse=True)
        return pipeline_versions.pipeline_versions[0].pipeline_version_id
    else:
        return None

# Function that create a kpf experiment and returns the id
def create_experiment(client: kfp_cli.Client, experiment_name: str) -> str:
    experimment = client.create_experiment(name=experiment_name)
    print(f">>>experimment: {experimment}")
    return experimment.experiment_id

# Function that creates a run of a pipeline id in a given experiment id with the latest version of the pipeline
def create_run(client: kfp_cli.Client, pipeline_id: str, experiment_id: str, run_name: str, params: dict) -> str:
    pipeline_version_id = get_latest_pipeline_version_id(client, pipeline_id)
    print(f">>>pipeline_version_id: {pipeline_version_id}")
    if pipeline_version_id is None:
        raise ValueError(f"No pipeline versions found for pipeline_id: {pipeline_id}")
    run = client.run_pipeline(
        experiment_id=experiment_id,
        job_name=run_name,
        pipeline_id=pipeline_id,
        version_id=pipeline_version_id,
        params=params
    )
    print(f"run: {run}")
    return run.run_id

# Download experiment report from S3 bucket and folder
def download_experiment_report(experiment_name: str) -> dict:
    # Experiment report s3 key to store the report
    experiment_report_file_name = f"{experiment_name}.yaml"
    experiment_report_s3_key = f"{experiment_reports_folder}/{experiment_report_file_name}"
    print(f"experiment_report_s3_key = {experiment_report_s3_key}")
        # Create a session using the provided credentials and region
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Create an S3 resource object using the session and region
    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    # Get the bucket
    bucket = s3_resource.Bucket(bucket_name)

    # Create a temporary directory to store the report
    local_tmp_dir = '/tmp/experiment_report'
    print(f">>> local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Local file to store the report
    experiment_report_file_path = f'{local_tmp_dir}/{experiment_report_file_name}'

    print(f"Downloading {experiment_report_s3_key} to {experiment_report_file_path}")
    bucket.download_file(experiment_report_s3_key, experiment_report_file_path)
    print(f"Downloaded {experiment_report_s3_key}")

    # Read the content of the file and retun it
    experiment_report = None
    with open(experiment_report_file_path, 'r') as file:
        experiment_report = file.read()
    return load_yaml(experiment_report)

# Return a dict from a yaml string
def load_yaml(search_space: str) -> dict:
    return yaml.safe_load(search_space)