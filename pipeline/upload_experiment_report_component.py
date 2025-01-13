from kfp import compiler

from kfp.dsl import Input, component

# This component generates a yaml report and saves it in a bucket
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
# - SCALER_S3_KEY
# - EVALUATION_DATA_S3_KEY
# - EXPERIMENT_REPORTS_FOLDER_S3_KEY
# The data is in pickel format and the file name is passed as an environment variable S3_KEY.
@component(
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
    packages_to_install=["boto3", "botocore"]
)
def upload_experiment_report(
    experiment_name: str,
    run_name: str,
    metric_value: float,
    model_name: str, 
    image_size: int, 
    epochs: int, 
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    confidence_threshold: float,
    iou_threshold: float,
    label_smoothing: float,
):
    import boto3
    import botocore
    import os

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')
    experiment_reports_folder = os.environ.get('EXPERIMENT_REPORTS_FOLDER_S3_KEY')

    # metric_value = None
    # # Load the metric value from the input path
    # try:
    #     print(f"Loading metric value from {metric_value_input.path}")
    #     with open(metric_value_input.path, 'rb') as f:
    #         # Read the metric value str from the input and convert it to float
    #         metric_value = float(f.read().decode('utf-8'))
    #         print(f"metric_value = {metric_value}")
    # except Exception as e:
    #     raise ValueError(f"Failed to load metric value: {e}")

    # Experiment report s3 key to store the report
    experiment_report_file_name = f"{experiment_name}.yaml"
    experiment_report_s3_key = f"{experiment_reports_folder}/{experiment_report_file_name}"
    print(f"experiment_report_s3_key = {experiment_report_s3_key}")

    # Create a session using the provided credentials and region
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Create an S3 resource object using the session and region
    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    # Get the bucket
    bucket = s3_resource.Bucket(bucket_name)

    # Create a temporary directory to store the report
    local_tmp_dir = '/tmp/experiment_report'
    print(f">>> local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Local file to store the report
    experiment_report_file_path = f'{local_tmp_dir}/{experiment_report_file_name}'

    # Write the report to a string
    experiment_report_str = f"report:\n  experiment_name: {experiment_name}"
    experiment_report_str += f"\n  run_name: {run_name}"
    experiment_report_str += f"\n  metric_value: {metric_value}"
    experiment_report_str += f"\n  params:"
    experiment_report_str += f"\n    model_name: {model_name}"
    experiment_report_str += f"\n    image_size: {image_size}"
    experiment_report_str += f"\n    epochs: {epochs}"
    experiment_report_str += f"\n    batch_size: {batch_size}"
    experiment_report_str += f"\n    optimizer: {optimizer}"
    experiment_report_str += f"\n    learning_rate: {learning_rate}"
    experiment_report_str += f"\n    momentum: {momentum}"
    experiment_report_str += f"\n    weight_decay: {weight_decay}"
    experiment_report_str += f"\n    confidence_threshold: {confidence_threshold}"
    experiment_report_str += f"\n    iou_threshold: {iou_threshold}"
    experiment_report_str += f"\n    label_smoothing: {label_smoothing}"

    # Write the report to a file
    with open(experiment_report_file_path, 'w') as f:
        f.write(experiment_report_str)

    print(f"Uploading report to {experiment_report_s3_key}")

    # Upload the report to S3
    if not os.path.exists(experiment_report_file_path):
        raise ValueError(f"Model file {experiment_report_file_path} does not exist")

    # Upload the file
    print(f"Uploading {experiment_report_file_path} to {experiment_report_s3_key}")
    bucket.upload_file(experiment_report_file_path, f"{experiment_report_s3_key}")

def main():
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    print(f"Compiling component to {component_package_path}")
    compiler.Compiler().compile(
        pipeline_func=upload_experiment_report,
        package_path=component_package_path
    )

if __name__ == "__main__":
    main()
