#!/bin/bash

# Load environment variables
. .env

# Install the Crunchy PostgreSQL Operator in manual mode
echo "Installing the Crunchy PostgreSQL Operator in manual mode"
cat <<EOF | oc create -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: crunchy-postgres-operator
  namespace: openshift-operators
spec:
  channel: v5
  installPlanApproval: Manual
  name: crunchy-postgres-operator
  source: certified-operators
  sourceNamespace: openshift-marketplace
  startingCSV: ${CRUNCHY_POSTGRES_CSV}
EOF

# Wait for the install plan to be created
echo "Waiting for the install plan to be created"
while [ -z "$(oc get installplans -n openshift-operators | grep ${CRUNCHY_POSTGRES_CSV})" ]; do
  echo "Waiting for the install plan to be created"
  sleep 5
done

# List the names of all install plans using -o jsonpath in namespace openshift-operators that have 
# spec.clusterServiceVersionNames which contain ${CRUNCHY_POSTGRES_CSV} and delete all but the first one
# Get the list of installplan names
installplans=$(oc get installplans -n openshift-operators -o json | jq -r --arg csv "$CRUNCHY_POSTGRES_CSV" '.items[] | select(.spec.clusterServiceVersionNames[]? == $csv) | .metadata.name')

# Iterate through each installplan name
for installplan in $installplans; do
  echo "Processing installplan: $installplan"
  # Check if the `installplan` environment variable is empty.
  if [ -z "$installplan" ]; then
    echo "Install plan collection empty..."
    continue
  fi
  # Check if the `INSTALL_PLAN_TO_PATCH` environment variable is empty.
  if [ -z "$INSTALL_PLAN_TO_PATCH" ]; then
    INSTALL_PLAN_TO_PATCH=$installplan
    echo "Approving $installplan"
    oc patch installplan/$installplan -n openshift-operators --type merge --patch '{"spec":{"approved":true}}'
    continue
  fi
  echo "Deleting $installplan"
  oc delete installplan/$installplan -n openshift-operators
done

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/doc-bot
cat <<EOF | oc apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ${DATA_SCIENCE_PROJECT_NAMESPACE}
  namespace: ${ARGOCD_NAMESPACE}
spec:
  project: default
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: ${DATA_SCIENCE_PROJECT_NAMESPACE}
  source:
    path: gitops/yolo
    repoURL: ${REPO_URL}
    targetRevision: ${TARGET_REVISION}
    helm:
      parameters:
        - name: argocdNamespace
          value: "${ARGOCD_NAMESPACE}"
        - name: vcs.url
          value: "${REPO_URL}"
        - name: vcs.ref
          value: "${TARGET_REVISION}"
        - name: instanceName
          value: "${INSTANCE_NAME}"
        - name: dataScienceProjectNamespace
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}"
        - name: dataScienceProjectDisplayName
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}"
        - name: pipelines.connection.name
          value: "pipelines"
        - name: pipelines.connection.displayName
          value: "pipelines"
        - name: pipelines.connection.type
          value: "s3"
        - name: pipelines.connection.awsAccessKeyId
          value: "${MINIO_ACCESS_KEY}"
        - name: pipelines.connection.awsSecretAccessKey
          value: "${MINIO_SECRET_KEY}"
        - name: pipelines.connection.awsDefaultRegion
          value: "none"
        - name: pipelines.connection.awsS3Bucket
          value: "pipelines"
        - name: pipelines.connection.scheme
          value: "http"
        - name: pipelines.connection.awsS3Endpoint
          value: "minio.ic-shared-minio.svc:9000"

        - name: datasets.connection.name
          value: "datasets"
        - name: datasets.connection.displayName
          value: "datasets"
        - name: datasets.connection.type
          value: "s3"
        - name: datasets.connection.awsAccessKeyId
          value: "${MINIO_ACCESS_KEY}"
        - name: datasets.connection.awsSecretAccessKey
          value: "${MINIO_SECRET_KEY}"
        - name: datasets.connection.awsDefaultRegion
          value: "none"
        - name: datasets.connection.awsS3Bucket
          value: "datasets"
        - name: pipelines.connection.scheme
          value: "http"
        - name: datasets.connection.awsS3Endpoint
          value: "minio.ic-shared-minio.svc:9000"

        - name: models.connection.name
          value: "models"
        - name: models.connection.displayName
          value: "models"
        - name: models.connection.type
          value: "s3"
        - name: models.connection.awsAccessKeyId
          value: "${MINIO_ACCESS_KEY}"
        - name: models.connection.awsSecretAccessKey
          value: "${MINIO_SECRET_KEY}"
        - name: models.connection.awsDefaultRegion
          value: "none"
        - name: models.connection.awsS3Bucket
          value: "models"
        - name: pipelines.connection.scheme
          value: "http"
        - name: models.connection.awsS3Endpoint
          value: "minio.ic-shared-minio.svc:9000"
  syncPolicy:
    automated:
      selfHeal: true      
EOF


# Wait until the project is created
echo "Waiting for the project to be created"
while [ -z "$(oc get project ${DATA_SCIENCE_PROJECT_NAMESPACE})" ]; do
  echo "Waiting for the project to be created"
  sleep 5
done

# Deploy the MLflow server
helm repo add strangiato https://strangiato.github.io/helm-charts/
helm upgrade -n ${DATA_SCIENCE_PROJECT_NAMESPACE} -i mlflow-server \
  --set objectStorage.objectBucketClaim.enable=false \
  --set objectStorage.mlflowBucketName=mlflow \
  --set objectStorage.s3EndpointUrl=http://minio.ic-shared-minio.svc:9000 \
  --set objectStorage.s3AccessKeyId=minio \
  --set objectStorage.s3SecretAccessKey=minio123 \
  strangiato/mlflow-server