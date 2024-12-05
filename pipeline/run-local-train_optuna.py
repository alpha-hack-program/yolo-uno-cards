# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# pip install -r requirements-local.txt

import os

from typing import Dict, Optional

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['AWS_DEFAULT_REGION'] = 'none'
os.environ['AWS_S3_BUCKET'] = 'models'
os.environ['AWS_S3_ENDPOINT'] = f'https://minio-api-ic-shared-minio.apps.ocp.sandbox2563.opentlc.com'
os.environ['EXPERIMENT_REPORTS_FOLDER_S3_KEY'] = 'experiment-reports'


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

# run pipeline
generate_search_space_task = generate_search_space(
    epochs_type="categorical",
    epochs_bounds="2",
    lr_type="float",                           # float, categorical uniform
    lr_bounds="0.001, 0.01",
    momentum_type="uniform",                   # float, categorical uniform
    momentum_bounds="0.9, 0.99",
    weight_decay_type="float",                 # float, categorical uniform
    weight_decay_bounds="0.0005, 0.001",
    image_size_type="categorical",             # float, categorical uniform
    image_size_bounds="640",
    confidence_threshold_type="uniform",       # float, categorical uniform
    confidence_threshold_bounds="0.001, 0.005",
    iou_threshold_type="uniform",              # float, categorical uniform
    iou_threshold_bounds="0.4, 0.6",
    optimizer_type="categorical",              # float, categorical uniform
    optimizer_bounds="Adam",
    label_smoothing_type="uniform",            # float, categorical uniform
    label_smoothing_bounds="0.07, 0.2",
    batch_size_type="categorical",             # float, categorical uniform
    batch_size_bounds="2, 16, 32",
)
# print(generate_search_space_task.outputs["Output"])
print(generate_search_space_task)


# train_model_task = train_model_optuna(
#         model_name="model_name",
#         n_trials=2,
#         search_space=generate_search_space_task.outputs["Output"],
#         experiment_name_prefix="exp-001",
#         pipeline_name="train_yolo"
#     )

def train_model_optuna(
    model_name: str,                    # e.g: yolov8n
    n_trials: int,                      # e.g: 5
    search_space: str,
    experiment_name: str,
    pipeline_name: str,
    images_dataset_name: str = "uno-cards-v1.2",
    images_datasets_root_folder: str = "datasets",
    images_dataset_yaml: str = "data.yaml",
    models_root_folder: str = "models",
    images_dataset_pvc_name: str = "images-datasets-pvc",
    images_dataset_pvc_size_in_gi: int = 5,
):
    import os
    import yaml
    import re

    import optuna

    import boto3
    import botocore
    import os

    from kfp import client as kfp_cli

    from kubernetes import client as k8s_cli, config as k8s_conf

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

        # run_name compose of the experiment name and the optuna 'trial' number
        run_name = f"{experiment_name}-trial-{trial.number}"
        print(f"Run name: {run_name}")
        
        # Add experiment_name, here is the experiment name is the run_name 
        params["experiment_name"] = run_name

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

import time

experiment_name_prefix="exp-002"
experiment_name = f"{experiment_name_prefix}-{int(time.time())}"
train_model_task = train_model_optuna(
        model_name="yolov8n",
        n_trials=2,
        search_space=generate_search_space_task,
        experiment_name=experiment_name,
        pipeline_name="train_yolo"
    )

