#!/bin/bash

export COMPONENT_NAME=train_yolo
export ORG=atarazana
export REGISTRY=quay.io/${ORG}

DATA_SCIENCE_PROJECT_NAMESPACE=$(oc project --short)

# If DATA_SCIENCE_PROJECT_NAMESPACE is empty print error and exit
if [ -z "$DATA_SCIENCE_PROJECT_NAMESPACE" ]; then
  echo "Error: No namespace found. Please set the namespace in bootstrap/.env file."
  exit 1
fi

TOKEN=$(oc whoami -t)

# If TOKEN is empty print error and exit
if [ -z "$TOKEN" ]; then
  echo "Error: No token found. Please login to OpenShift using 'oc login' command."
  echo "Compile only mode."

  python train_yolo_component.py

  exit 1
fi

DSPA_HOST=$(oc get route ds-pipeline-dspa -n ${DATA_SCIENCE_PROJECT_NAMESPACE} -o jsonpath='{.spec.host}')

echo "DSPA_HOST: $DSPA_HOST"

# If DSPA_HOST is empty print error and exit
if [ -z "$DSPA_HOST" ]; then
  echo "Error: No host found for ds-pipeline-dspa. Please check if the deployment is successful."
  exit 1
fi

python train_yolo_component.py $TOKEN $DSPA_HOST



