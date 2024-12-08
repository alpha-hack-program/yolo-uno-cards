#!/bin/sh

COMPONENT_NAME=rust_calc

podman run -it --rm localhost/${COMPONENT_NAME}:latest $@
# podman run -it --rm --entrypoint bash localhost/${COMPONENT_NAME}:latest 
