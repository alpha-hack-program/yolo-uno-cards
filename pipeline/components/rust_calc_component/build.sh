#!/bin/sh

COMPONENT_NAME=rust_calc

podman build -t ${COMPONENT_NAME}:latest -f ${COMPONENT_NAME}/Containerfile ${COMPONENT_NAME}/
