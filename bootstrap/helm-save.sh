#!/bin/sh

. .env

CHART_REPO="strangiato"
CHART_NAME="mlflow-server"
CHART_VERSION="0.7.1"

helm pull "${CHART_REPO}/${CHART_NAME}" --version "${CHART_VERSION}" --untar

helm dependency update ./${CHART_NAME}

# Tar ./${CHART_NAME}
tar -cvzf ./"${CHART_NAME}"-v"${CHART_VERSION}".tgz  "./${CHART_NAME}"

# List of images
IMAGES=(
    "quay.io/troyer/mlflow-server"
    "registry.redhat.io/openshift4/ose-oauth-proxy:v4.12"
    "quay.io/troyer/mlflow-server-training-test:latest"
)

# Function to pull and save images
pull_and_save_images() {
    for image in "${IMAGES[@]}"; do
        echo "Pulling image: $image"
        if podman pull "$image"; then
            echo "Successfully pulled image: $image"

            # Generate tar file name by replacing special characters
            tar_filename=./images/$(echo "$image" | sed 's|/|_|g; s|:|_|g').tar

            echo "Saving image $image to $tar_filename"
            if podman save -o "$tar_filename" "$image"; then
                echo "Successfully saved image $image to $tar_filename"
            else
                echo "Failed to save image: $image"
            fi
        else
            echo "Failed to pull image: $image"
        fi
    done
}

# Execute the function
pull_and_save_images





