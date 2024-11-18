#!/bin/sh
ARGOCD_APP_NAME=pipelines

# Load environment variables
REPO_URL="uri: https://github.com/alpha-hack-program/yolo-uno-cards.git"
TARGET_REVISION="fb-mount"
INSTANCE_NAME="yolo-uno-cards"
DATA_SCIENCE_PROJECT_NAMESPACE="yolo"

helm template . --name-template ${ARGOCD_APP_NAME} \
  --set vcs.url="${REPO_URL}" \
  --set vcs.ref="${TARGET_REVISION}" \
  --set instanceName="${INSTANCE_NAME}" \
  --set dataScienceProjectNamespace=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set dataScienceProjectDisplayName=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set mountCaCerts="false" \
  --set pipelines.connection.name="pipelines" \
  --set pipelines.connection.displayName="pipelines" \
  --set pipelines.connection.type="s3" \
  --set pipelines.connection.scheme="http" \
  --set pipelines.connection.awsAccessKeyId="minio" \
  --set pipelines.connection.awsSecretAccessKey="minio123" \
  --set pipelines.connection.awsDefaultRegion="none" \
  --set pipelines.connection.awsS3Bucket="pipelines" \
  --set pipelines.connection.awsS3Endpoint="minio.ic-shared-minio.svc:9000" \
  --include-crds