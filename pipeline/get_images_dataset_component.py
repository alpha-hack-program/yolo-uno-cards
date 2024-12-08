from kfp import compiler

from kfp.dsl import component

BASE_IMAGE="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111"
BOTOCORE_PIP_VERSION= "1.35.54"

# This component downloads the dataset from an S3 bucket and unzips it in the specified volume mount path.
# The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
# - SCALER_S3_KEY
# - EVALUATION_DATA_S3_KEY
# - MODELS_S3_KEY
# The data is in pickel format and the file name is passed as an environment variable S3_KEY.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[f"boto3=={BOTOCORE_PIP_VERSION}", f"botocore=={BOTOCORE_PIP_VERSION}"],
)
def get_images_dataset(
    images_datasets_root_folder: str, 
    images_dataset_name: str,
    images_dataset_yaml: str,
    root_mount_path: str,
    force_clean: bool
):
    import boto3
    import botocore
    import os
    import shutil
    import re

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    # Construct the S3 key for the dataset
    images_dataset_s3_key = f"{images_datasets_root_folder}/{images_dataset_name}.zip"
    print(f"images_dataset_s3_key = {images_dataset_s3_key}")

    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    bucket = s3_resource.Bucket(bucket_name)

    # Local temporary directory and dataset folder paths
    local_tmp_dir = f"{root_mount_path}/tmp"
    images_dataset_path = f"{root_mount_path}/{images_datasets_root_folder}"
    images_dataset_folder = f"{images_dataset_path}/{images_dataset_name}"

    # Check if the dataset folder already exists
    if os.path.exists(images_dataset_folder) and not force_clean:
        print(f"Dataset folder {images_dataset_folder} already exists. Skipping download and extraction.")
    else:
        # Ensure local_tmp_dir exists
        if not os.path.exists(local_tmp_dir):
            os.makedirs(local_tmp_dir)

        # Get the file name from the S3 key
        file_name = f"{images_dataset_name}.zip"
        local_file_path = f"{local_tmp_dir}/{file_name}"

        # Check if the file exists in S3
        objs = list(bucket.objects.filter(Prefix=images_dataset_s3_key))
        if not any(obj.key == images_dataset_s3_key for obj in objs):
            raise ValueError(f"File {images_dataset_s3_key} does not exist in the bucket {bucket_name}")
        
        # Download the file
        print(f"Downloading {images_dataset_s3_key} to {local_file_path}")
        bucket.download_file(images_dataset_s3_key, local_file_path)
        print(f"Downloaded {images_dataset_s3_key}")

        # Ensure dataset path exists
        if not os.path.exists(images_dataset_path):
            os.makedirs(images_dataset_path)

        # Unzip the file into the images dataset volume mount path
        print(f"Unzipping {local_file_path} to {images_dataset_path}")
        shutil.unpack_archive(local_file_path, images_dataset_path)
        print(f"Unzipped {local_file_path} to {images_dataset_path}")

        # Clean up the temporary zip file
        os.remove(local_file_path)

    # Verify the dataset YAML file exists
    images_dataset_yaml_path = os.path.join(images_dataset_folder, images_dataset_yaml)
    if not os.path.exists(images_dataset_yaml_path):
        raise ValueError(f"Dataset YAML file {images_dataset_yaml} not found in {images_dataset_folder}")

    # Update the YAML file with the correct dataset path
    with open(images_dataset_yaml_path, 'r') as f:
        data = f.read()
        data = re.sub(r'path: .*', f'path: {images_dataset_folder}', data)
        print(f"Updated YAML file: {data}")
        with open(images_dataset_yaml_path, 'w') as f:
            f.write(data)

if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=get_images_dataset,
        package_path=component_package_path
    )
