# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# pip install -r requirements-local.txt

import os

from kfp import local
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath

from train import pipeline

local.init(runner=local.SubprocessRunner())

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['AWS_DEFAULT_REGION'] = 'none'
os.environ['AWS_S3_ENDPOINT'] = 'https://minio-s3-ic-shared-minio.apps.cluster-8q7tj.8q7tj.sandbox277.opentlc.com/'
os.environ['AWS_S3_BUCKET'] = 'datasets'

# run pipeline
pipeline_task = pipeline(tracking_uri="http://localhost:8080")

