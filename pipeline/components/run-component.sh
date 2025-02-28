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

# Run the component image
podman run -it --rm localhost/${COMPONENT_NAME}:latest bash
# podman run -it --rm --entrypoint bash localhost/${COMPONENT_NAME}:latest 
