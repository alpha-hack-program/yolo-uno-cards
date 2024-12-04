# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os
import sys

from typing import Dict

from jmespath import search
from kubernetes import client, config

import kfp

from kfp import kubernetes
from kfp import compiler
from kfp import dsl

from kfp.dsl import Input, Output, Metrics, OutputPath

DATASETS_CONNECTION_SECRET = "aws-connection-datasets"
MODELS_CONNECTION_SECRET = "aws-connection-models"

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
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
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

# This component trains a model using Optuna to optimize hyperparameters.
# Arguments:
# - model_name: The name of the model to train.
# - n_trials: The number of trials to run.
# - epochs: The number of epochs to train the model.
# - experiment_name: The name of the MLflow experiment.
# - run_name: The name of the MLflow run.
# - tracking_uri: The URI of the MLflow tracking server.
# - images_dataset_name: The name of the images dataset.
# - images_datasets_root_folder: The root folder where the images datasets are stored.
# - images_dataset_yaml: The YAML file containing the images dataset information.
# - models_root_folder: The root folder where the models are stored.
# - root_mount_path: The root mount path for the volumes.
# - model_name_output: The output path for the trained model name.
# - results_output_metrics: The output for the training results metrics.
@dsl.component(
    # base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
    packages_to_install=["load_dotenv==0.1.0", "optuna==4.1.0", "kfp[kubernetes]==2.8.0", "kubernetes==23.6.0"]
)
def train_model_optuna(
    model_name: str,                    # e.g: yolov8n
    n_trials: int,                      # e.g: 5
    search_space: str,
    experiment_name_prefix: str,
    pipeline_name: str,
):
    import os
    import time
    import yaml
    import re

    import optuna

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
            config.load_incluster_config()
        except config.config_exception.ConfigException:
            config.load_kube_config()

        # Get token path from environment or default to kubernetes token location
        NAMESPACE_PATH = os.environ.get("NAMESPACE_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/namespace")

        # Get the current namespace
        with open(NAMESPACE_PATH, "r") as f:
            namespace = f.read().strip()

        print(f"namespace: {namespace}")

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

    def get_pipeline_id_by_name(client: kfp.Client, pipeline_name: str):
        return client.get_pipeline_id(pipeline_name)

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
    
    def get_pipeline(client: kfp.Client, pipeline_id: str):
        return client.get_pipeline(pipeline_id)

    def get_latest_pipeline_version_id(client: kfp.Client, pipeline_id: str) -> Optional[str]:
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
    def create_experiment(client: kfp.Client, experiment_name: str) -> str:
        experimment = client.create_experiment(name=experiment_name)
        print(f">>>experimment: {experimment}")
        return experimment.experiment_id

    # Function that creates a run of a pipeline id in a given experiment id with the latest version of the pipeline
    def create_run(client: kfp.Client, pipeline_id: str, experiment_id: str, run_name: str, params: dict) -> str:
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


    # Return a dict from a yaml string
    def load_search_space(search_space: str) -> dict:
        return yaml.safe_load(search_space)

    # Define the objective function
    def objective(trial: optuna.Trial, search_space: dict, experiment_name: str, token: str, kfp_endpoint: str):
        # Dynamically define hyperparameter search space
        params = {}
        for param_name, param_config in search_space.items():
            param_type = param_config["type"]
            if param_type == "float":
                low = float(param_config["low"])
                high = float(param_config["high"])
                params[param_name] = trial.suggest_float(
                    param_name, low, high
                )
            elif param_type == "uniform":
                low = float(param_config["low"])
                high = float(param_config["high"])
                params[param_name] = trial.suggest_uniform(
                    param_name, low, high
                )
            elif param_type == "categorical":
                # Check if choices match a number pattern
                choices = []
                for choice in param_config["choices"]:
                    # check if matches number pattern
                    if re.match(r'^-?\d+(?:\.\d+)?$', choice):
                        # Convert to float or int
                        choices.append(float(choice) if '.' in choice else int(choice))
                    else:
                        choices.append(choice)
                params[param_name] = trial.suggest_categorical(
                    param_name, choices
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

        print(f"params: {params}")

        # Use the evaluation metric as the objective
        metric_name = "training/map"
        metric_value = 0.0
        try:
            client = kfp.Client(host=kfp_endpoint, existing_token=token)

            # Get the pipeline by name
            print(f">>> Pipeline name: {pipeline_name}")
            pipeline_id = get_pipeline_id_by_name(client, pipeline_name)
            if pipeline_id:
                print(f"Pipeline ID {pipeline_id}")
                
                # Create a experiment
                experiment_id = create_experiment(client, experiment_name)

                # Get pipeline
                pipeline = get_pipeline(client, pipeline_id)
                print(f"pipeline: {pipeline}")

                # Create a run
                run_name = f"{experiment_name}-trial-{trial.number}"
                print(f"Run name: {run_name}")
                run_id = create_run(client, pipeline_id, experiment_id, run_name, params)

                # Wait for the run to complete
                # wait_for_run_completion(client, run_id)
                run_details = client.wait_for_run_completion(run_id=run_id, timeout=3600, sleep_duration=10)

                # Get the run
                # run_details = client.get_run(run_id=run_id)
                print(f"run_details {run_details}")

                # Get metrics from the run and find metric_name
                task_details = run_details.run_details.task_details
                print(f"task_details {task_details}")

                # .get('pipeline_runtime', {}).get('execution_metrics', [])
                # for metric in metrics:
                #     print(f"Metric Name: {metric['name']}, Value: {metric['value']}")

            else:
                print(f"Pipeline {pipeline_name} does not exist.") 
                raise ValueError(f"Pipeline {pipeline_name} does not exist.")
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError(f"Error: {e}")
        
        return metric_value
    
    
    # Get token and kfp endpoint
    token = get_token()
    print(f"Token: {token}")
    kfp_endpoint = get_route_host(route_name="ds-pipeline-dspa")
    print(f"KFP endpoint: {kfp_endpoint}")

    # Generate the experiment name from the prefix and timestamp
    experiment_name = f"{experiment_name_prefix}-{int(time.time())}"
    print(f"Experiment name: {experiment_name}")

    # Get the optuna study file name from the S3 key
    study_name = f"{experiment_name}"
    
    # Create a local temporary directory
    local_tmp_dir = "/tmp/optuna"

    # Make sure the local temporary directory exists
    os.makedirs(local_tmp_dir, exist_ok=True)

    # optuna storage
    storage = f"sqlite:///{local_tmp_dir}/optuna.db"

    # Create or load the study
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True
    )

    # Load search space from str
    search_space = load_search_space(search_space)

    # Run the optimization callback a func
    study.optimize(lambda trial: objective(trial, search_space, experiment_name, token, kfp_endpoint), n_trials=n_trials)

    # Print the best hyperparameters
    print("\nBest Hyperparameters:")
    print(study.best_params)

    # Print study statistics
    print("\nStudy best trial:")
    print(study.best_trial)
    print("\nStudy best value:")
    print(study.best_value)
    print("\nStudy trials:")
    print(study.trials)

@dsl.component(
    # base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
    packages_to_install=["onnx==1.16.1", "onnxruntime==1.18.0", "scikit-learn==1.5.0", "numpy==1.24.3", "pandas==2.2.2"]
)
def yield_not_deployed_error():
    raise ValueError("Model not deployed")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    experiment_name_prefix: str = "yolo-uno-cards-",
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
    pipeline_name: str = 'train_yolo'):

    
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

    # Train the model
    train_model_task = train_model_optuna(
        model_name=model_name,
        n_trials=n_trials,
        search_space=generate_search_space_task.outputs["Output"],
        experiment_name_prefix=experiment_name_prefix,
        pipeline_name=pipeline_name
    ).set_caching_options(False)



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