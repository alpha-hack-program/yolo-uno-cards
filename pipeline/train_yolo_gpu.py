# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os
import sys

from typing import Dict

from kubernetes import client, config

import kfp

from kfp import kubernetes
from kfp import compiler
from kfp import dsl

from kfp.dsl import Input, Output, Metrics, OutputPath
from kfp.components import load_component_from_file

# Load the register model from a url
REGISTER_MODEL_COMPONENT_URL = "https://raw.githubusercontent.com/alpha-hack-program/model-serving-utils/refs/heads/main/components/register_model/src/component_metadata/register_model.yaml"
register_model_component = kfp.components.load_component_from_url(REGISTER_MODEL_COMPONENT_URL)

# Load the components from the files
if not os.path.exists('setup_storage_component.yaml'):
    from setup_storage_component import main as setup_storage_main
    setup_storage_main()
setup_storage_component = load_component_from_file('setup_storage_component.yaml')

if not os.path.exists('get_images_dataset_component.yaml'):
    from get_images_dataset_component import main as get_images_dataset_main
    get_images_dataset_main()
get_images_dataset_component = load_component_from_file('get_images_dataset_component.yaml')

if not os.path.exists('train_yolo_component.yaml'):
    from train_yolo_component import main as train_yolo_main
    train_yolo_main()
train_yolo_component = load_component_from_file('train_yolo_component.yaml')

if not os.path.exists('upload_model_component.yaml'):
    from upload_model_component import main as upload_model_main
    upload_model_main()
upload_model_component = load_component_from_file('upload_model_component.yaml')

if not os.path.exists('upload_experiment_report_component.yaml'):
    from upload_experiment_report_component import main as upload_experiment_report_main
    upload_experiment_report_main()
upload_experiment_report_component = load_component_from_file('upload_experiment_report_component.yaml')

DATASETS_CONNECTION_SECRET = "aws-connection-datasets"
MODELS_CONNECTION_SECRET = "aws-connection-models"

@dsl.component(
    # base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
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

    # Define the root mount path
    root_mount_path = "/opt/app-root/src"

    # Define the datasets volume
    datasets_pvc_name = images_dataset_pvc_name
    datasets_pvc_size_in_gi = images_dataset_pvc_size_in_gi

    setup_storage_task = setup_storage_component(
        pvc_name=datasets_pvc_name, size_in_gi=datasets_pvc_size_in_gi
    ).set_caching_options(False).set_display_name("dataset_storage")

    setup_shm_task = setup_storage_component(
        pvc_name='shm-pvc', size_in_gi=2
    ).set_caching_options(False).set_display_name("shm_storage")

    # Get the dataset
    force_dataset_path_clean = force_clean
    get_images_dataset_task = get_images_dataset_component(
        images_datasets_root_folder=images_datasets_root_folder,
        images_dataset_name=images_dataset_name,
        images_dataset_yaml=images_dataset_yaml,
        root_mount_path=root_mount_path,
        force_clean=force_dataset_path_clean
    ).set_caching_options(False)
    get_images_dataset_task.after(setup_storage_task)

    # Train the model
    train_model_task = train_yolo_component(
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
    train_model_task.after(get_images_dataset_task)
    train_model_task.after(setup_shm_task)
    # TODO externalize these values
    train_model_task.set_memory_request("16Gi")
    train_model_task.set_cpu_request("4")
    train_model_task.set_memory_limit("20Gi")
    train_model_task.set_cpu_limit("6")
    # This need empty_dir_mount which is not available in the current version of the RHOAI pipelines
    # train_model_task.set_accelerator_type("nvidia.com/gpu").set_gpu_limit(1)
    kubernetes.add_node_selector(
        train_model_task,
        label_key='nvidia.com/gpu.product',
        label_value='NVIDIA-A10G'
    )
    kubernetes.add_toleration(
        train_model_task,
        key='nvidia.com/gpu',
        operator='Exists',
        effect='NoSchedule'
    )

    # Extract model name and metric value
    model_name = train_model_task.outputs["model_name_output"]
    metric_value = train_model_task.outputs["metric_value_output"]

    # Upload the model
    upload_model_task = upload_model_component(
        root_mount_path=root_mount_path,
        models_root_folder=models_root_folder,
        model_name=model_name
    ).after(train_model_task).set_caching_options(False)

    # Upload the experiment report
    upload_experiment_report_component_task = upload_experiment_report_component(
        experiment_name=experiment_name,
        run_name=run_name,
        metric_value=metric_value,
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
    ).after(train_model_task).set_caching_options(False)

    model_registry_name = "model-registry-dev"
    istio_system_namespace = "istio-system"
    model_name = f"{experiment_name}"
    model_uri = "oci://model_uri"
    model_version = f"{run_name}"
    model_description = "Model to detect uno cards"
    model_format_name = "onnx"
    model_format_version = "1"
    author = "author"
    owner = "owner"

    labels = {
        "vision": "",
        "yolo": "",
        "uno-cards": "",
        "dataset": images_dataset_name,
        "experiment": experiment_name,
        "run": run_name,
        "metric_value": metric_value,
    }

    # Register model
    register_model_task = register_model_component(
        model_registry_name=model_registry_name,
        istio_system_namespace=istio_system_namespace,
        model_name=model_name,
        model_uri=model_uri,
        model_version=model_version,
        model_description=model_description,
        model_format_name=model_format_name,
        model_format_version=model_format_version,
        author=author,
        owner=owner,
        labels=labels,
        input_metrics=train_model_task.outputs["results_output_metrics"],
    ).after(train_model_task).set_caching_options(False)

    # Mount the PVC to the task 
    kubernetes.mount_pvc(
        upload_model_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path,
    )

    # Setting environment variables for upload_model_task
    upload_model_task.set_env_variable(name="MODELS_S3_KEY", value="models/yolo")
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

    # Setting environment variables for upload_experiment_report_component_task
    upload_experiment_report_component_task.set_env_variable(name="EXPERIMENT_REPORTS_FOLDER_S3_KEY", value="experiment-reports")
    kubernetes.use_secret_as_env(
        task=upload_experiment_report_component_task,
        secret_name=MODELS_CONNECTION_SECRET,
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        }
    )

    ## Prepare environment variables and PVC mounts for get_images_dataset_task

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

    # Mount the PVC to the task 
    kubernetes.mount_pvc(
        get_images_dataset_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path
    )

    ## Prepare environment variables and PVC mounts for train_model_task

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

    kubernetes.mount_pvc(
        train_model_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path,
    )

    # While the empty_dir_mount is not available, we can use ephemeral volume
    kubernetes.mount_pvc(
        train_model_task,
        pvc_name='shm-pvc',
        mount_path='/dev/shm'
    )

    # This needs an upcoming version of the RHOAI pipelines
    # # Mount the PVC to the task 
    # kubernetes.empty_dir_mount(
    #     train_model_task,
    #     volume_name='shm',
    #     mount_path='/dev/shm',
    #     medium='Memory',
    #     size_limit='2Gi'
    # )

if __name__ == '__main__':
    import time

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