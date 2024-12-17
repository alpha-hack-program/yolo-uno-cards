#!/bin/sh

export COMPONENT_NAME=train_yolo
export ORG=atarazana
export REGISTRY=quay.io/${ORG}

kfp component build src/ --component-filepattern ${COMPONENT_NAME}.py --no-push-image --overwrite-dockerfile --no-build-image

podman build -t ${COMPONENT_NAME}:latest -f src/Dockerfile src/
