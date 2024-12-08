#!/bin/sh

COMPONENT_NAME=train_yolo_optuna

kfp component build src/ --component-filepattern ${COMPONENT_NAME}.py --no-push-image --overwrite-dockerfile --no-build-image

podman build -t ${COMPONENT_NAME}:latest -f src/Dockerfile src/
