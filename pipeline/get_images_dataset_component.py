from kfp import compiler

from kfp.dsl import component

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
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
    packages_to_install=["boto3", "botocore"]
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

    # Construct and set the IMAGES_DATASET_S3_KEY environment variable
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

    # TODO don't download the file if it already exists in the volume mount path!!
    # Create a temporary directory to store the dataset => TODO USE PVC INSTEAD!!!!! CLEAN LATER!!!
    local_tmp_dir = '/tmp/get_images_dataset'
    print(f">>> local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Get the file name from the S3 key
    file_name = f"{images_dataset_name}.zip"
    # Download the file
    local_file_path = f'{local_tmp_dir}/{file_name}'

    # If file doesn't exist in the bucket raise a ValueError
    objs = list(bucket.objects.filter(Prefix=images_dataset_s3_key))
    if not any(obj.key == images_dataset_s3_key for obj in objs):
        raise ValueError(f"File {images_dataset_s3_key} does not exist in the bucket {bucket_name}")
    
    print(f"Downloading {images_dataset_s3_key} to {local_file_path}")
    bucket.download_file(images_dataset_s3_key, local_file_path)
    print(f"Downloaded {images_dataset_s3_key}")

    # Dataset path
    images_dataset_path = f"{root_mount_path}/{images_datasets_root_folder}"

    # Ensure dataset path exists
    if not os.path.exists(images_dataset_path):
        os.makedirs(images_dataset_path)

    # List the files in the dataset path
    print(f"Listing files in {images_dataset_path}")
    print(os.listdir(images_dataset_path))

    # If we haven't unzipped the file yet or we're forced to, unzip it
    images_dataset_folder = f"{images_dataset_path}/{images_dataset_name}"
    if not os.path.exists(images_dataset_folder) or force_clean:
        # Unzip the file into the images dataset volume mount path
        print(f"Unzipping {local_file_path} to {images_dataset_path}")
        shutil.unpack_archive(f'{local_file_path}', f'{images_dataset_path}')
        print(f"Unzipped {local_file_path} to {images_dataset_path}")

        # List the files inside images_dataset_folder folder
        print(f"Listing files in {images_dataset_folder}")
        print(os.listdir(images_dataset_folder))

    # Locate the YAML file in the dataset folder and replace the path with the actual path
    images_dataset_yaml_path = os.path.join(images_dataset_folder, images_dataset_yaml)
    print(f"images_dataset_yaml_path: {images_dataset_yaml_path}")

    # If the YAML file doesn't exist, raise a ValueError
    if not os.path.exists(images_dataset_yaml_path):
        raise ValueError(f"Dataset YAML file {images_dataset_yaml} not found in {images_dataset_folder}")

    # Replace regex 'path: .*' with 'path: {images_dataset_folder}'
    with open(images_dataset_yaml_path, 'r') as f:
        data = f.read()
        data = re.sub(r'path: .*', f'path: {images_dataset_folder}', data)
        # Print the updated YAML file
        print(f"Updated YAML file: {data}")
        # Write the updated YAML file
        with open(images_dataset_yaml_path, 'w') as f:
            f.write(data)

if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=get_images_dataset,
        package_path=component_package_path
    )
