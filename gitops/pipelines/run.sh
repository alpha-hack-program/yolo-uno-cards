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
  --set partOf="${DATA_SCIENCE_PROJECT_NAMESPACE}" \
  --set mountCaCerts="false" \
  --set pipelines.enabled="true" \
  --set pipelines.connection.name="pipelines" \
  --set pipelines.connection.displayName="Pipelines S3 Storage" \
  --set pipelines.connection.description="S3-compatible storage connection for ML pipelines" \
  --set pipelines.connection.type="s3" \
  --set pipelines.connection.typeProtocol="s3" \
  --set pipelines.connection.typeRef="s3" \
  --set pipelines.connection.scheme="http" \
  --set pipelines.connection.host="minio.ic-shared-minio.svc" \
  --set pipelines.connection.port="9000" \
  --set pipelines.connection.awsAccessKeyId="minio" \
  --set pipelines.connection.awsSecretAccessKey="minio123" \
  --set pipelines.connection.awsDefaultRegion="none" \
  --set pipelines.connection.awsS3Bucket="pipelines" \
  --set s3.config.awsS3BucketList="pipelines\,models\,datasets" \
  --set s3.connection.awsAccessKeyId="minio" \
  --set s3.connection.awsSecretAccessKey="minio123" \
  --set s3.connection.awsS3Endpoint="http://minio.ic-shared-minio.svc:9000" \
  --set images.python3="registry.redhat.io/ubi8/python-39:1-158" \
  --include-crds