#!/bin/sh

. .env

CHART_REPO="strangiato"
CHART_NAME="mlflow-server"
CHART_VERSION="0.7.1"

# List of images
IMAGES=(
    "quay.io/troyer/mlflow-server"
    "registry.redhat.io/openshift4/ose-oauth-proxy:v4.12"
    "quay.io/troyer/mlflow-server-training-test:latest"
)

# untar ./"${CHART_NAME}"-v"${CHART_VERSION}".tgz to ./${CHART_NAME}
tar -xvzf ./"${CHART_NAME}"-v"${CHART_VERSION}".tgz  "./${CHART_NAME}"

# Function to import images into OpenShift
import_images_to_openshift() {
    for image in "${IMAGES[@]}"; do
        tar_filename=./images/$(echo "$image" | sed 's|/|_|g; s|:|_|g').tar

        if [ -f "$tar_filename" ]; then
            echo "Importing image $image from $tar_filename into OpenShift"
            oc image import "${image}" --from-file="$tar_filename" --confirm
            if [ $? -eq 0 ]; then
                echo "Successfully imported image $image into OpenShift"
            else
                echo "Failed to import image $image into OpenShift"
            fi
        else
            echo "Tar file $tar_filename not found for image $image"
        fi
    done
}

# Execute the import function
import_images_to_openshift

# Deploy the MLflow server
helm install mlflow-server ./${CHART_NAME} -n ${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set objectStorage.objectBucketClaim.enabled=false \
  --set objectStorage.mlflowBucketName=${PIPELINES_BUCKET_NAME} \
  --set objectStorage.s3EndpointUrl=https://${S3_ENDPOINT} \
  --set objectStorage.s3AccessKeyId=${PIPELINES_ACCESS_KEY} \
  --set objectStorage.s3SecretAccessKey=${PIPELINES_SECRET_KEY} \
  strangiato/${CHART_NAME}





