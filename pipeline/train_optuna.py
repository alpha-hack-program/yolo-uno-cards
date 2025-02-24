# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os
import sys
import time

from kubernetes import client, config

import kfp

from kfp import compiler
from kfp import dsl
from kfp.dsl import Output, Metrics

from kfp import kubernetes

DATASETS_CONNECTION_SECRET = "aws-connection-datasets"
MODELS_CONNECTION_SECRET = "aws-connection-models"

BASE_IMAGE="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111"

KFP_PIP_VERSION="2.8.0"
K8S_PIP_VERSION="23.6.0"
OPTUNA_PIP_VERSION="4.1.0"

# Load the train_yolo_optuna component from a url
TRAIN_YOLO_OPTUNA_COMPONENT_URL = "https://raw.githubusercontent.com/alpha-hack-program/yolo-uno-cards/refs/heads/main/pipeline/components/train_yolo_optuna_component/src/component_metadata/train_model_optuna.yaml"
train_yolo_optuna_component = kfp.components.load_component_from_url(TRAIN_YOLO_OPTUNA_COMPONENT_URL)

# This component checks the kfp env
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[f"kfp[kubernetes]=={KFP_PIP_VERSION}", f"kubernetes=={K8S_PIP_VERSION}"]
)
def check_env() -> bool:
    
    try:
        from kfp import client as kfp_cli
        print(f"The package kfp is installed.")

        try:
            client = kfp_cli.Client()
            print("kfp.Client() works")
            return True
        except:
            print("kfp.Client() fails")
    except ImportError:
        print(f"The package kfp is not installed.")
 
    return False

# This component generates a yaml document as a string with this shape:
# image_size:
#   type: categorical
#   choices: [320, 416, 608]
# confidence_threshold:
#   type: uniform
#   low: 0.1
#   high: 0.5
# iou_threshold:
#   type: uniform
#   low: 0.4
#   high: 0.6
# optimizer:
#   type: categorical
#   choices: ["SGD", "Adam", "AdamW"]
# label_smoothing:
#   type: uniform
#   low: 0.0
#   high: 0.1
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["load_dotenv==0.1.0"]
)
def generate_search_space(
    epochs_type: str,
    epochs_bounds: str,
    lr_type: str,
    lr_bounds: str,
    momentum_type: str,
    momentum_bounds: str,
    weight_decay_type: str,
    weight_decay_bounds: str,
    image_size_type: str,
    image_size_bounds: str,
    confidence_threshold_type: str,
    confidence_threshold_bounds: str,
    iou_threshold_type: str,
    iou_threshold_bounds: str,
    optimizer_type: str,
    optimizer_bounds: str,
    label_smoothing_type: str,
    label_smoothing_bounds: str,
    batch_size_type: str,
    batch_size_bounds: str,
) -> str:
    def add_to_yaml(yaml_str, param_name, param_type, param_bounds=None):
        yaml_str += f"{param_name}:\n  type: {param_type}\n"
        if param_type == "float":
            param_value = [float(x) for x in param_bounds.split(",")]
            yaml_str += f"  low: {param_value[0]}\n  high: {param_value[1]}\n"
        elif param_type == "categorical":
            param_value = [x.strip() for x in param_bounds.split(",")]
            yaml_str += f"  choices: {param_value}\n"
        elif param_type == "uniform":
            param_value = [float(x) for x in param_bounds.split(",")]
            yaml_str += f"  low: {param_value[0]}\n  high: {param_value[1]}\n"
        else:
            raise ValueError(f"Invalid parameter type: {param_type}")

        return yaml_str

    yaml_str = ""

    if epochs_type is not None:
        yaml_str = add_to_yaml(yaml_str, "epochs", epochs_type, epochs_bounds)

    if lr_type is not None:
        yaml_str = add_to_yaml(yaml_str, "learning_rate", lr_type, lr_bounds)

    if momentum_type is not None:
        yaml_str = add_to_yaml(yaml_str, "momentum", momentum_type, momentum_bounds)

    if weight_decay_type is not None:
        yaml_str = add_to_yaml(yaml_str, "weight_decay", weight_decay_type, weight_decay_bounds)

    if image_size_type is not None:
        yaml_str = add_to_yaml(yaml_str, "image_size", image_size_type, image_size_bounds)

    if confidence_threshold_type is not None:
        yaml_str = add_to_yaml(yaml_str, "confidence_threshold", confidence_threshold_type, confidence_threshold_bounds)

    if iou_threshold_type is not None:
        yaml_str = add_to_yaml(yaml_str, "iou_threshold", iou_threshold_type, iou_threshold_bounds)

    if optimizer_type is not None:
        yaml_str = add_to_yaml(yaml_str, "optimizer", optimizer_type, optimizer_bounds)

    if label_smoothing_type is not None:
        yaml_str = add_to_yaml(yaml_str, "label_smoothing", label_smoothing_type, label_smoothing_bounds)

    if batch_size_type is not None:
        yaml_str = add_to_yaml(yaml_str, "batch_size", batch_size_type, batch_size_bounds)

    return yaml_str

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    experiment_name_prefix: str = "uno-cards",
    model_name: str = "yolov8n", 
    n_trials: int = 3,
    epochs_type: str = "categorical",
    epochs_bounds: str = "2",
    lr_type: str = "float",                           # float, categorical uniform
    lr_bounds: str = "0.001, 0.01",
    momentum_type: str = "uniform",                   # float, categorical uniform
    momentum_bounds: str = "0.9, 0.99",
    weight_decay_type: str = "float",                 # float, categorical uniform
    weight_decay_bounds: str = "0.0005, 0.001",
    image_size_type: str = "categorical",             # float, categorical uniform
    image_size_bounds: str = "640",
    confidence_threshold_type: str = "uniform",       # float, categorical uniform
    confidence_threshold_bounds: str = "0.001, 0.005",
    iou_threshold_type: str = "uniform",              # float, categorical uniform
    iou_threshold_bounds: str = "0.4, 0.6",
    optimizer_type: str = "categorical",              # float, categorical uniform
    optimizer_bounds: str = "Adam",                   # Adam, SGD, AdamW
    label_smoothing_type: str = "uniform",            # float, categorical uniform
    label_smoothing_bounds: str = "0.07, 0.2",
    batch_size_type: str = "categorical",             # float, categorical uniform
    batch_size_bounds: str = "8, 16, 32",
    pipeline_name: str = 'train_yolo',
    images_dataset_name: str = "uno-cards-v1.2",
    images_datasets_root_folder: str = "datasets",
    images_dataset_yaml: str = "data.yaml",
    models_root_folder: str = "models",
    images_dataset_pvc_name: str = "images-datasets-pvc",
    images_dataset_pvc_size_in_gi: int = 5,
    author: str = "John Doe",
    owner: str = "acme",
    model_tags: str = "vision, yolo, uno-cards",
    model_registry_name: str = "model-registry-dev",
    istio_system_namespace: str = "istio-system"):

    check_env_task = check_env()
    
    # Generate search_space
    generate_search_space_task = generate_search_space(
        epochs_type=epochs_type,
        epochs_bounds=epochs_bounds,
        lr_type=lr_type,
        lr_bounds=lr_bounds,
        momentum_type=momentum_type,
        momentum_bounds=momentum_bounds,
        weight_decay_type=weight_decay_type,
        weight_decay_bounds=weight_decay_bounds,
        image_size_type=image_size_type,
        image_size_bounds=image_size_bounds,
        confidence_threshold_type=confidence_threshold_type,
        confidence_threshold_bounds=confidence_threshold_bounds,
        iou_threshold_type=iou_threshold_type,
        iou_threshold_bounds=iou_threshold_bounds,
        optimizer_type=optimizer_type,
        optimizer_bounds=optimizer_bounds,
        label_smoothing_type=label_smoothing_type,
        label_smoothing_bounds=label_smoothing_bounds,
        batch_size_type=batch_size_type,
        batch_size_bounds=batch_size_bounds,
    ).set_caching_options(False)

    # Train the model (local train_model_optuna) or remote component train_yolo_optuna_component
    train_model_task = train_yolo_optuna_component(
        model_name=model_name,
        n_trials=n_trials,
        search_space=generate_search_space_task.outputs["Output"],
        experiment_name_prefix=experiment_name_prefix,
        pipeline_name=pipeline_name,
        images_dataset_name=images_dataset_name,
        images_datasets_root_folder=images_datasets_root_folder,
        images_dataset_yaml=images_dataset_yaml,
        models_root_folder=models_root_folder,
        images_dataset_pvc_name=images_dataset_pvc_name,
        images_dataset_pvc_size_in_gi=images_dataset_pvc_size_in_gi,
        author=author,
        owner=owner,
        model_tags=model_tags,
        model_registry_name=model_registry_name,
        istio_system_namespace=istio_system_namespace
    ).set_caching_options(False).after(check_env_task)

    # Setting environment variables for train_model_task
    train_model_task.set_env_variable(name="EXPERIMENT_REPORTS_FOLDER_S3_KEY", value="experiment-reports")
    kubernetes.use_secret_as_env(
        task=train_model_task,
        secret_name=MODELS_CONNECTION_SECRET,
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        }
    )

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
        client = kfp.Client(host=kfp_endpoint, existing_token=token, verify_ssl=False)

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