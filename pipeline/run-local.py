# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# pip install -r requirements-local.txt

import os

from kfp import local

from train import train_model_optuna

local.init(runner=local.SubprocessRunner())

experiment_name = "uno-cards-v1.2-1"
bucket_name = "yolo-uno-cards-2024"
region = "eu-central-1"
images_dataset_name = "uno-cards-v1.2"
images_datasets_root_folder = "datasets"
images_dataset_s3_key = f"{images_datasets_root_folder}/{images_dataset_name}.zip"
experiments_root_folder = "experiments"

os.environ['AWS_ACCESS_KEY_ID'] = 'XXXX'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YYYY'
os.environ['AWS_DEFAULT_REGION'] = region
os.environ['AWS_S3_BUCKET'] = bucket_name
# os.environ['AWS_S3_ENDPOINT'] = f'https://{bucket_name}.s3.{region}.amazonaws.com'
os.environ['IMAGES_DATASET_S3_KEY'] = images_dataset_s3_key

# run train_model_optuna
train_model_optuna_task = train_model_optuna(
    model_name="yolov8n",
    n_trials=2,
    epochs=1,
    experiment_name=experiment_name,
    run_name=f"{experiment_name}",
    tracking_uri="http://localhost:5000",
    images_dataset_name=images_dataset_name,
    images_datasets_root_folder=images_datasets_root_folder,
    images_dataset_yaml="local.yaml",
    models_root_folder="models",
    experiments_root_folder=experiments_root_folder,
    root_mount_path="/Users/cvicensa/Projects/openshift/indra/iniciativa-2",
)

# run pipeline
# pipeline_task = pipeline(tracking_uri="http://localhost:8080")

