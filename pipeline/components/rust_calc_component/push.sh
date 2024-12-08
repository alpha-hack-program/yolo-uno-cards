#!/bin/sh

COMPONENT_NAME=rust_calc

podman tag localhost/${COMPONENT_NAME}:latest quay.io/atarazana/${COMPONENT_NAME}:latest
podman push quay.io/atarazana/${COMPONENT_NAME}:latest
