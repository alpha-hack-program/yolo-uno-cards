#!/bin/sh

export COMPONENT_NAME=train_yolo_optuna
export ORG=atarazana
export REGISTRY=quay.io/${ORG}

podman tag localhost/${COMPONENT_NAME}:latest quay.io/atarazana/${COMPONENT_NAME}:latest
podman push quay.io/atarazana/${COMPONENT_NAME}:latest
