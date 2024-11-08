# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os
from pyexpat import model
import sys

import kfp

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath, InputPath

from kfp import kubernetes

from kubernetes import client, config

DATASETS_CONNECTION_SECRET = "aws-connection-datasets"
MODELS_CONNECTION_SECRET = "aws-connection-models"

# This component creates a PersistentVolumeClaim (PVC) in the current namespace with the specified size, 
# access mode and storage class
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["kubernetes==23.6.0"]
)
def setup_storage(
    pvc_name: str,
    size_in_gi: int,
    access_mode: str = "ReadWriteOnce",
    storage_class: str = ""
) -> None:
    """Sets up a PersistentVolumeClaim (PVC) if it does not exist.

    Args:
        pvc_name (str): Name of the PVC to create.
        size_in_gi (int): Size of the PVC in GiB.
        storage_class (str): Storage class for the PVC. Default is an empty string.

    Raises:
        ValueError: If `size_in_gi` is less than 0.
        RuntimeError: If there's any other issue in PVC creation.
    """
    import os
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    if size_in_gi < 0:
        raise ValueError("size_in_gi must be a non-negative integer.")

    print(f"Creating PVC '{pvc_name}' with size {size_in_gi}Gi, access mode '{access_mode}', and storage class '{storage_class}'.") 

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

    print(f"Token: {token} Namespace: {namespace}")

    # Configure the client
    # curl -k --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
    #   -H "Authorization: Bearer $(cat /var/run/secrets/kubernetes.io/serviceaccount/token)"  \
    #   https://kubernetes.default.svc/api/v1/namespaces/iniciativa-2/persistentvolumeclaims/images-datasets-pvc
    kubernetes_host = f"https://{os.getenv('KUBERNETES_SERVICE_HOST', 'kubernetes.default.svc')}:{os.getenv('KUBERNETES_SERVICE_PORT', '443')}"
    print(f"kubernetes_host: {kubernetes_host}")
    configuration.host = kubernetes_host
    # configuration.host = 'https://kubernetes.default.svc'
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

    # Check if the PVC already exists
    try:
        v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        print(f"PVC '{pvc_name}' already exists.")
        return
    except ApiException as e:
        if e.status != 404:
            raise RuntimeError(f"Error checking for existing PVC: {e}")

    # Define PVC spec
    pvc_spec = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(name=pvc_name),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=[access_mode],
            resources=client.V1ResourceRequirements(
                requests={"storage": f"{size_in_gi}Gi"}
            )
        )
    )

    # Add storage class if provided
    if storage_class:
        pvc_spec.spec.storage_class_name = storage_class

    # Attempt to create the PVC
    try:
        v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_spec)
        print(f"PVC '{pvc_name}' created successfully in namespace '{namespace}'.")
    except ApiException as e:
        raise RuntimeError(f"Failed to create PVC: {e.reason}")
    
# This component downloads the dataset from an S3 bucket and unzips it in the specified volume mount path.
# The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
# - SCALER_S3_KEY
# - EVALUATION_DATA_S3_KEY
# - MODELS_S3_KEY
# The data is in pickel format and the file name is passed as an environment variable S3_KEY.
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def get_images_dataset(
    images_datasets_root_folder: str, 
    images_dataset_name: str,
    images_dataset_yaml: str,
    root_mount_path: str,
    force_clean: bool
):
    import boto3
    import botocore
    import os
    import shutil
    import re

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    # Construct and set the IMAGES_DATASET_S3_KEY environment variable
    images_dataset_s3_key = f"{images_datasets_root_folder}/{images_dataset_name}.zip"

    print(f"images_dataset_s3_key = {images_dataset_s3_key}")

    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    bucket = s3_resource.Bucket(bucket_name)

    # Create a temporary directory to store the dataset
    local_tmp_dir = '/tmp/get_images_dataset'
    print(f">>> local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Get the file name from the S3 key
    file_name = f"{images_dataset_name}.zip"
    # Download the file
    local_file_path = f'{local_tmp_dir}/{file_name}'

    # If file doesn't exist in the bucket raise a ValueError
    objs = list(bucket.objects.filter(Prefix=images_dataset_s3_key))
    if not any(obj.key == images_dataset_s3_key for obj in objs):
        raise ValueError(f"File {images_dataset_s3_key} does not exist in the bucket {bucket_name}")
    
    print(f"Downloading {images_dataset_s3_key} to {local_file_path}")
    bucket.download_file(images_dataset_s3_key, local_file_path)
    print(f"Downloaded {images_dataset_s3_key}")

    # Dataset path
    images_dataset_path = f"{root_mount_path}/{images_datasets_root_folder}"

    # Ensure dataset path exists
    if not os.path.exists(images_dataset_path):
        os.makedirs(images_dataset_path)

    # List the files in the dataset path
    print(f"Listing files in {images_dataset_path}")
    print(os.listdir(images_dataset_path))

    # If we haven't unzipped the file yet or we're forced to, unzip it
    images_dataset_folder = f"{images_dataset_path}/{images_dataset_name}"
    if not os.path.exists(images_dataset_folder) or force_clean:
        # Unzip the file into the images dataset volume mount path
        print(f"Unzipping {local_file_path} to {images_dataset_path}")
        shutil.unpack_archive(f'{local_file_path}', f'{images_dataset_path}')
        print(f"Unzipped {local_file_path} to {images_dataset_path}")

        # List the files inside images_dataset_folder folder
        print(f"Listing files in {images_dataset_folder}")
        print(os.listdir(images_dataset_folder))

    # Locate the YAML file in the dataset folder and replace the path with the actual path
    images_dataset_yaml_path = os.path.join(images_dataset_folder, images_dataset_yaml)
    print(f"images_dataset_yaml_path: {images_dataset_yaml_path}")

    # If the YAML file doesn't exist, raise a ValueError
    if not os.path.exists(images_dataset_yaml_path):
        raise ValueError(f"Dataset YAML file {images_dataset_yaml} not found in {images_dataset_folder}")

    # Replace regex 'path: .*' with 'path: {images_dataset_folder}'
    with open(images_dataset_yaml_path, 'r') as f:
        data = f.read()
        data = re.sub(r'path: .*', f'path: {images_dataset_folder}', data)
        # Print the updated YAML file
        print(f"Updated YAML file: {data}")
        # Write the updated YAML file
        with open(images_dataset_yaml_path, 'w') as f:
            f.write(data)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["ultralytics==8.3.22", "load_dotenv==0.1.0", "numpy==1.26.4", "mlflow==2.17.1", "onnxruntime==1.19.2", "onnxslim==0.1.36"]
)
def train_model(
    model_name: str, 
    image_size: int, 
    batch_size: int, 
    epochs: int, 
    experiment_name: str,
    run_name: str,
    tracking_uri: str,
    images_dataset_name: str,
    images_datasets_root_folder: str,
    images_dataset_yaml: str,
    models_root_folder: str,
    root_mount_path: str,
    model_name_output: OutputPath(str),
    results_output_metrics: Output[Metrics]
):
    import os
    import shutil
    import time

    import torch
    from ultralytics import YOLO, settings
    import mlflow

    import numpy as np
    
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')
    images_dataset_s3_key = os.environ.get('IMAGES_DATASET_S3_KEY')

    print(f"S3: endpoint_url {endpoint_url}")
    print(f"S3: bucket_name {bucket_name}")
    print(f"S3: images_dataset_s3_key {images_dataset_s3_key}")

    print(f"tracking_uri {tracking_uri}")
    print(f"experiment_name {experiment_name}")
    print(f"images_dataset_name {images_dataset_name}")
    print(f"images_datasets_root_folder {images_datasets_root_folder}")
    print(f"images_dataset_yaml {images_dataset_yaml}")
    print(f"models_root_folder {models_root_folder}")
    print(f"root_mount_path {root_mount_path}")

    # If root_mount_path is not set or doesn't exist, raise a ValueError
    if not root_mount_path or not os.path.exists(root_mount_path):
        raise ValueError(f"Root mount path '{root_mount_path}' does not exist")

    # Set the images dataset folder
    images_dataset_folder = os.path.join(root_mount_path, images_datasets_root_folder, images_dataset_name)

    # If the images dataset folder doesn't exist, raise a ValueError
    if not os.path.exists(images_dataset_folder):
        raise ValueError(f"Images dataset folder {images_dataset_folder} does not exist")

    # Set the models folder
    models_folder = os.path.join(root_mount_path, models_root_folder)

    # Make sure the models folder exists
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Set the images dataset YAML path
    images_dataset_yaml_path = os.path.join(images_dataset_folder, images_dataset_yaml)
    print(f"Checking if {images_dataset_yaml_path} exists")
    if not os.path.exists(images_dataset_yaml_path):
        raise ValueError(f"Dataset YAML file {images_dataset_yaml} not found in {images_dataset_folder}")
    print(f"Dataset YAML file found in {images_dataset_yaml_path}")

    # Set the MLflow tracking URI and experiment
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    # Update a setting
    settings.update({"mlflow": True})

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using device: MPS")
    print(f"Using device: {device}")

    # Reset settings to default values
    settings.reset()

    # Load the model
    model = YOLO(f'{model_name}.pt')

    # Set the run name
    train_run_name = f"{run_name}-{model_name}-train-{int(time.time())}"
    print(f"Current run name: {train_run_name}")

    # Start the MLflow run for training
    with mlflow.start_run(run_name=train_run_name) as training_mlrun:
        mlflow.log_param("dataset_file", f"{endpoint_url}/{bucket_name}/{images_dataset_s3_key}")
        mlflow.log_param("dataset_name", images_dataset_name)
        mlflow.log_param("datasets_root_folder", images_datasets_root_folder)
        mlflow.log_param("dataset_yaml", images_dataset_yaml)
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("imgsz", image_size)

        print(f"Training model {model_name} with dataset {images_dataset_yaml_path}.")
        results = model.train(
            data=images_dataset_yaml_path, 
            epochs=epochs, 
            imgsz=image_size, 
            batch=batch_size, 
            device=device
        )
        
        if hasattr(results, 'box'):
            results_output_metrics.log_metric("training/map", results.box.map)
            results_output_metrics.log_metric("training/map50", results.box.map50)
            results_output_metrics.log_metric("training/map75", results.box.map75)
            results_output_metrics.log_metric("training/mp", results.box.mp)
            results_output_metrics.log_metric("training/mr", results.box.mr)
            results_output_metrics.log_metric("training/nc", results.box.nc)
        else:
            print("No box attribute in the results!!!")

        # Save the trained model
        print(f"Saving model to {models_folder}")
        trained_model_name = f"model-{train_run_name}"
        print(f"Trained model name: {trained_model_name}")
        trained_model_pt_path = os.path.join(models_folder, f"{trained_model_name}.pt")
        print(f"Saving model to {trained_model_pt_path}")
        model.save(trained_model_pt_path)
        print(f"Model saved to {trained_model_pt_path}")

        #  If the trained model was not saved, raise a ValueError
        if not os.path.exists(trained_model_pt_path):
            raise ValueError(f"Model was not saved at {trained_model_pt_path}")
        
        # End the run
        mlflow.end_run()

        # Start the MLflow run for validation
        val_run_name = f"{run_name}-{model_name}-val-{int(time.time())}"
        print(f"Current run name: {train_run_name}")
        with mlflow.start_run(run_name=val_run_name) as validation_mlrun:
            # Validate the model    
            validation_results = model.val()

            # If the results have the box attribute, log the metrics
            if hasattr(validation_results, 'box'):
                mlflow.log_metric("val/map", validation_results.box.map)
                mlflow.log_metric("val/map50", validation_results.box.map50)
                mlflow.log_metric("val/map75", validation_results.box.map75)
                mlflow.log_metric("val/mp", validation_results.box.mp)
                mlflow.log_metric("val/mr", validation_results.box.mr)
                mlflow.log_metric("val/nc", validation_results.box.nc)
                results_output_metrics.log_metric("val/map", validation_results.box.map)
                results_output_metrics.log_metric("val/map50", validation_results.box.map50)
                results_output_metrics.log_metric("val/map75", validation_results.box.map75)
                results_output_metrics.log_metric("val/mp", validation_results.box.mp)
                results_output_metrics.log_metric("val/mr", validation_results.box.mr)
                results_output_metrics.log_metric("val/nc", validation_results.box.nc)
            else:
                print("No box attribute in the results!!!")
            
        # Convert the model to ONNX
        trained_model_onnx_path_tmp = model.export(format="onnx")
        if not trained_model_onnx_path_tmp:
            print("Failed to export model to ONNX format")

        # Save the onnx model
        trained_model_onnx_path = os.path.join(models_folder, f"{trained_model_name}.onnx")
        print(f"Copying {trained_model_onnx_path_tmp} to {trained_model_onnx_path}")
        shutil.copy(trained_model_onnx_path_tmp, trained_model_onnx_path)
        print(f"Copied {trained_model_onnx_path_tmp} to {trained_model_onnx_path}")

        # Set the output paths
        with open(model_name_output, 'w') as f:
            f.write(trained_model_name)

    if not training_mlrun:
        raise ValueError("MLflow run was not started")

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["onnx==1.16.1", "onnxruntime==1.18.0", "scikit-learn==1.5.0", "numpy==1.24.3", "pandas==2.2.2"]
)
def yield_not_deployed_error():
    raise ValueError("Model not deployed")

# This component parses the metrics and extracts the accuracy
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301"
)
def parse_metrics(metrics_input: Input[Metrics], map75_output: OutputPath(float)):
    print(f"metrics_input: {dir(metrics_input)}")
    map75 = metrics_input.metadata["val/map75"]

    with open(map75_output, 'w') as f:
        f.write(str(map75))

# This component uploads the model to an S3 bucket. The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(
    root_mount_path: str,
    models_root_folder: str,
    model_name: str,
    ):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    models_s3_key = os.environ.get("MODELS_S3_KEY")

    # Set the models folder
    models_folder = os.path.join(root_mount_path, models_root_folder)

    # Set the model paths for the model.pt and model.onnx
    model_pt_path = os.path.join(models_folder, f"{model_name}.pt")
    model_onnx_path = os.path.join(models_folder, f"{model_name}.onnx")

    print(f"Uploading {model_pt_path} and {model_onnx_path} to {models_s3_key} in {bucket_name} bucket in {endpoint_url} endpoint")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading to {models_s3_key}")

    # Upload the model.pt and model.onnx files to the S3 bucket
    for model_path in [model_pt_path, model_onnx_path]:
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist")

        # Upload the file
        bucket.upload_file(model_path, f"{models_s3_key}/{os.path.basename(model_path)}")

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["kubernetes"]
)
def refresh_deployment(deployment_name: str):
    import datetime
    import kubernetes

    # Use the in-cluster config
    # Load in-cluster Kubernetes configuration but if it fails, load local configuration
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()

    # Get the current namespace
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
        namespace = f.read().strip()

    # Create Kubernetes API client
    api_instance = kubernetes.client.CustomObjectsApi()

    # Define the deployment patch
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}"
                    }
                }
            }
        }
    }

    try:
        # Patch the deployment
        api_instance.patch_namespaced_custom_object(
            group="apps",
            version="v1",
            namespace=namespace,
            plural="deployments",
            name=deployment_name,
            body=patch
        )
        print(f"Deployment {deployment_name} patched successfully")
    except Exception as e:
        print(f"Failed to patch deployment {deployment_name}: {e}")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    map75_threshold: float = 0.90, 
    model_name: str = "yolov8n", 
    image_size: int = 640, 
    batch_size: int = 2, 
    epochs: int = 1, 
    experiment_name: str = "YOLOv8n",
    run_name: str = "uno-cards",
    tracking_uri: str = "http://mlflow-server:8080",
    images_dataset_name: str = "uno-cards-v1.0",
    images_datasets_root_folder: str = "datasets",
    images_dataset_yaml: str = "data.yaml",
    images_dataset_pvc_name: str = "images-datasets-pvc",
    images_dataset_pvc_size_in_gi: int = 5,
    models_root_folder: str = "models",
    force_clean: bool = False,
    enable_caching: bool = False):

    # Define the root mount path
    root_mount_path = '/opt/app-root/src'

    # Define the datasets volume
    datasets_pvc_name = images_dataset_pvc_name
    datasets_pvc_size_in_gi = images_dataset_pvc_size_in_gi
    datasets_volume_mount_path = f"{root_mount_path}/{images_datasets_root_folder}"

    setup_storage_task = setup_storage(
        pvc_name=datasets_pvc_name, size_in_gi=datasets_pvc_size_in_gi
    ).set_caching_options(False)

    # Get the dataset
    force_dataset_path_clean = force_clean
    get_images_dataset_task = get_images_dataset(
        images_datasets_root_folder=images_datasets_root_folder,
        images_dataset_name=images_dataset_name,
        images_dataset_yaml=images_dataset_yaml,
        root_mount_path=root_mount_path,
        force_clean=force_dataset_path_clean
    ).set_caching_options(False)
    get_images_dataset_task.after(setup_storage_task)

    # Mount the PVC to the task 
    kubernetes.mount_pvc(
        get_images_dataset_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path,
    )

    # Train the model
    train_model_task = train_model(
        model_name=model_name,
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        images_dataset_name=images_dataset_name,
        images_datasets_root_folder=images_datasets_root_folder,
        images_dataset_yaml=images_dataset_yaml,
        models_root_folder=models_root_folder,
        root_mount_path=root_mount_path
    ).set_caching_options(False)
    train_model_task.after(get_images_dataset_task)
    train_model_task.set_memory_request("6Gi")
    train_model_task.set_cpu_request("4")
    train_model_task.set_memory_limit("8Gi")
    train_model_task.set_cpu_limit("6")
    # train_model_task.set_accelerator_type("nvidia.com/gpu.product")
    
    # Extract model name
    model_name = train_model_task.outputs["model_name_output"]

    # Mount the PVC to the task 
    kubernetes.mount_pvc(
        train_model_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path,
    )

    # Parse the metrics and extract the mean average precision at 75
    parse_metrics_task = parse_metrics(metrics_input=train_model_task.outputs["results_output_metrics"]).set_caching_options(False)
    map75 = parse_metrics_task.outputs["map75_output"]

    # Use the parsed mean average precision at 75 to decide if we should upload the model
    # Doc: https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/
    with dsl.If(map75 >= map75_threshold):
        upload_model_task = upload_model(
            root_mount_path=root_mount_path,
            models_root_folder=models_root_folder,
            model_name=model_name
        ).after(parse_metrics_task).set_caching_options(False)

        # Mount the PVC to the task 
        kubernetes.mount_pvc(
            upload_model_task,
            pvc_name=datasets_pvc_name,
            mount_path=root_mount_path,
        )

        # Setting environment variables for upload_model_task
        upload_model_task.set_env_variable(name="MODELS_S3_KEY", value="models/yolo/")
        kubernetes.use_secret_as_env(
            task=upload_model_task,
            secret_name=MODELS_CONNECTION_SECRET,
            secret_key_to_env={
                'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
                'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
                'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
                'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            }
        )
        
        # Refresh the deployment
        # refresh_deployment(deployment_name=deployment_name).after(upload_model_task).set_caching_options(False)
    with dsl.Else():
        yield_not_deployed_error().set_caching_options(False)

    # Define the IMAGES_DATASET_S3_KEY value
    images_dataset_s3_key = f"{images_datasets_root_folder}/{images_dataset_name}.zip"

    # Set the S3 keys for get_images_dataset_task and kubernetes secret to be used in the task
    get_images_dataset_task.set_env_variable(name="IMAGES_DATASET_S3_KEY", value=images_dataset_s3_key)
    kubernetes.use_secret_as_env(
        task=get_images_dataset_task,
        secret_name=DATASETS_CONNECTION_SECRET,
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        }
    )

    # Set the S3 keys for train_model and kubernetes secret to be used in the task
    train_model_task.set_env_variable(name="IMAGES_DATASET_S3_KEY", value=images_dataset_s3_key)

    kubernetes.use_secret_as_env(
        task=train_model_task,
        secret_name=DATASETS_CONNECTION_SECRET,
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        }
    )

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