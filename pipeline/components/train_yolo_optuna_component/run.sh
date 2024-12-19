#!/bin/sh

COMPONENT_NAME=train_yolo_optuna
ORG=atarazana
REGISTRY=quay.io/${ORG}

podman run -it --rm localhost/${COMPONENT_NAME}:latest $@
# podman run -it --rm --entrypoint bash localhost/${COMPONENT_NAME}:latest 
