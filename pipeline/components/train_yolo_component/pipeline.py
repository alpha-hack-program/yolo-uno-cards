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
        
    pipeline_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_package_path
    )