#!/bin/bash

COMPONENTS=("train_yolo_optuna" "train_yolo")

# If there is not exactly one argument and is not in the COMPONENTS array, print usage and exit
if [ "$#" -ne 1 ] || ! [[ " ${COMPONENTS[@]} " =~ " ${1} " ]]; then
    echo "Usage: $0 ${COMPONENTS[@]}"
    exit 1
fi

# Set the COMPONENT_NAME variable to the first argument
COMPONENT_NAME=$1

# Source the .env file
. .env

# Source the component-specific environment variables if the file exists
[ -f ${COMPONENT_NAME}/.env ] && . ${COMPONENT_NAME}/.env

# Build the image using the BASE_IMAGE build arg
podman build -t ${COMPONENT_NAME}:latest -f ./Containerfile . \
  --build-arg BASE_IMAGE=${BASE_IMAGE} \
  --build-arg COMPONENT_NAME=${COMPONENT_NAME}

# Build the component using the kfp CLI
export PYTHONPATH=${PYTHONPATH}:$(pwd)/${COMPONENT_NAME}/src:$(pwd)/shared
export REGISTRY
kfp component build ${COMPONENT_NAME}/src/ --component-filepattern ${COMPONENT_NAME}.py --no-push-image --no-build-image