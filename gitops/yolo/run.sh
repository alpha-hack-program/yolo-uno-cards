#!/bin/sh
ARGOCD_APP_NAME=pipelines

# Load environment variables
INSTANCE_NAME="yolo-uno-cards"
DATA_SCIENCE_PROJECT_NAMESPACE="yolo"

helm template . --name-template ${ARGOCD_APP_NAME} \
  --set instanceName="${INSTANCE_NAME}" \
  --set dataScienceProjectNamespace=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set dataScienceProjectDisplayName=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set mountCaCerts="false" \
  --include-crds