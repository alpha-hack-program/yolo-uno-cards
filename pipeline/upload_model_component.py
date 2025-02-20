from kfp import compiler

from kfp import dsl
from kfp.dsl import OutputPath


# This component uploads the model to an S3 bucket. The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
@dsl.component(
    # base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(
    root_mount_path: str,
    models_root_folder: str,
    model_name: str,
    models_s3_uri_output: OutputPath(str),
    model_pt_s3_uri_output: OutputPath(str),
    model_onnx_s3_uri_output: OutputPath(str),
    ):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    models_s3_key = os.environ.get("MODELS_S3_KEY")

    # Set the models folder
    models_folder = os.path.join(root_mount_path, models_root_folder)

    # Set the model paths for the model.pt and model.onnx
    model_pt_path = os.path.join(models_folder, f"{model_name}.pt")
    model_onnx_path = os.path.join(models_folder, f"{model_name}.onnx")

    print(f"Uploading {model_pt_path} and {model_onnx_path} to {models_s3_key}/{model_name}/1 in {bucket_name} bucket in {endpoint_url} endpoint")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading to {models_s3_key}")

    # Upload the model.pt and model.onnx files to the S3 bucket
    for model_path in [model_pt_path, model_onnx_path]:
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist")

        # Upload the file
        print(f"Uploading {model_path} to {models_s3_key}/{model_name}/1/{os.path.basename(model_path)}")
        bucket.upload_file(model_path, f"{models_s3_key}/{model_name}/1/{os.path.basename(model_path)}")

    # Build the models folder S3 URI using the S3 endpoint, bucket name, and models S3 key
    models_s3_uri = f"{endpoint_url}/{bucket_name}/{models_s3_key}/{model_name}"
    with open(models_s3_uri_output, 'w') as f:
        f.write(models_s3_uri)
    
    # Build the onnx model S3 URI using the S3 endpoint, bucket name, and model_name
    model_onnx_s3_uri = f"{endpoint_url}/{bucket_name}/{models_s3_key}/{model_name}.onnx"
    with open(model_onnx_s3_uri_output, 'w') as f:
        f.write(model_onnx_s3_uri)

    # Build the pt model S3 URI using the S3 endpoint, bucket name, and model_name
    model_pt_s3_uri = f"{endpoint_url}/{bucket_name}/{models_s3_key}/{model_name}.pt"
    with open(model_pt_s3_uri_output, 'w') as f:
        f.write(model_pt_s3_uri)

def main():
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    print(f"Compiling component to {component_package_path}")
    compiler.Compiler().compile(
        pipeline_func=upload_model,
        package_path=component_package_path
    )

if __name__ == "__main__":
    main()
