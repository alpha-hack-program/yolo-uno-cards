# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import json
import os

from kfp import dsl
from kfp.dsl import Output, Metrics

import os
import re
import time

import optuna

from kfp import client as kfp_cli

from utils import get_route_host, get_token, get_pipeline, get_pipeline_id_by_name, download_experiment_report, create_experiment, create_run, load_yaml

COMPONENT_NAME=os.getenv("COMPONENT_NAME", "train_yolo_optuna")

DATASETS_CONNECTION_SECRET = "aws-connection-datasets"
MODELS_CONNECTION_SECRET = "aws-connection-models"

BASE_IMAGE="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111"

NAMESPACE = os.environ.get("NAMESPACE", "default")

REGISTRY=os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")

TARGET_IMAGE=f"{REGISTRY}/{COMPONENT_NAME}:latest"

KFP_PIP_VERSION="2.8.0"
K8S_PIP_VERSION="23.6.0"
OPTUNA_PIP_VERSION="4.1.0"
LOAD_DOTENV_PIP_VERSION="0.1.0"

# Define the objective function
def objective(trial: optuna.Trial, 
              search_space: dict, 
              experiment_name: str,
              pipeline_name: str,
              model_name: str,
              images_dataset_name: str,
              images_datasets_root_folder: str,
              images_dataset_yaml: str,
              models_root_folder: str,
              images_dataset_pvc_name: str,
              images_dataset_pvc_size_in_gi: int,
              token: str, 
              kfp_endpoint: str):
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

    # run_name compose of the experiment name and the optuna 'trial' number
    run_name = f"{experiment_name}-{model_name}-trial-{trial.number}"
    print(f"Run name: {run_name}")
    
    # Add experiment_name, here is the experiment name is the run_name 
    params["experiment_name"] = run_name
    params["run_name"] = run_name

    # Add parmaters to set the dataset to use, etc.
    params["images_dataset_name"] = images_dataset_name
    params["images_datasets_root_folder"] = images_datasets_root_folder
    params["images_dataset_yaml"] = images_dataset_yaml
    params["models_root_folder"] = models_root_folder
    params["images_dataset_pvc_name"] = images_dataset_pvc_name
    params["images_dataset_pvc_size_in_gi"] = images_dataset_pvc_size_in_gi
    params["model_name"] = model_name

    print(f"params: {params}")

    # Use the evaluation metric as the objective
    metric_name = "training/map"
    metric_value = 0.0
    try:
        client = kfp_cli.Client(host=kfp_endpoint, existing_token=token)

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
            run_id = create_run(client, pipeline_id, experiment_id, run_name, params)

            # Wait for the run to complete
            # wait_for_run_completion(client, run_id)
            run_details = client.wait_for_run_completion(run_id=run_id, timeout=3600, sleep_duration=10)
            print(f"run_details {run_details}")

            # Get metrics from run_details is not working...
            if run_details.state == "FAILED":
                print(f"Run {run_name} failed.")
                raise ValueError(f"Run {run_name} failed.")

            # Load the experiment report from S3 we use the run_name as the experiment name
            experiment_report = download_experiment_report(run_name)
            # Access the values
            metric_value = experiment_report['report']['metric_value']
            print(f"metric_value: {metric_value}")
        else:
            print(f"Pipeline {pipeline_name} does not exist.") 
            raise ValueError(f"Pipeline {pipeline_name} does not exist.")
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(f"Error: {e}")
    
    return metric_value

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
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[f"load_dotenv=={LOAD_DOTENV_PIP_VERSION}", f"optuna=={OPTUNA_PIP_VERSION}", f"kfp[kubernetes]=={KFP_PIP_VERSION}", f"kubernetes=={K8S_PIP_VERSION}"]
)
def train_model_optuna(
    model_name: str,                    # e.g: yolov8n
    n_trials: int,                      # e.g: 5
    search_space: str,
    experiment_name_prefix: str,
    pipeline_name: str,
    images_dataset_name: str,
    images_datasets_root_folder: str,
    images_dataset_yaml: str,
    models_root_folder: str,
    images_dataset_pvc_name: str,
    images_dataset_pvc_size_in_gi: int,
    results_output_metrics: Output[Metrics]
):
    experiment_name = f"{experiment_name_prefix}-{int(time.time())}"

    # Get token and kfp endpoint
    token = get_token()
    print(f"Token: {token}")
    kfp_endpoint = get_route_host(route_name="ds-pipeline-dspa")
    print(f"KFP endpoint: {kfp_endpoint}")

    # Generate the experiment name from the prefix and timestamp
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
    search_space = load_yaml(search_space)

    # Run the optimization callback a func
    study.optimize(lambda trial: objective(trial=trial,
                                           search_space=search_space, 
                                           experiment_name=experiment_name, 
                                           pipeline_name=pipeline_name,
                                           model_name=model_name,
                                           images_dataset_name=images_dataset_name,
                                           images_datasets_root_folder=images_datasets_root_folder,
                                           images_dataset_yaml=images_dataset_yaml,
                                           models_root_folder=models_root_folder,
                                           images_dataset_pvc_name=images_dataset_pvc_name,
                                           images_dataset_pvc_size_in_gi=images_dataset_pvc_size_in_gi,
                                           token=token, 
                                           kfp_endpoint=kfp_endpoint), n_trials=n_trials)

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

    # Log the best value
    results_output_metrics.log_metric("best_value", study.best_value)

    # Log the best hyperparameters as a JSON string
    best_params_json = json.dumps(study.best_params)
    results_output_metrics.log_metric("best_hyperparameters", best_params_json)
