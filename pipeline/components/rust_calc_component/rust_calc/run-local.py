# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# pip install -r requirements-local.txt

import kfp

from kfp import local

multiply_component = kfp.components.load_component_from_file("component.yaml")

local.init(runner=local.SubprocessRunner())

# run train_model_optuna
train_model_optuna_task = multiply_component(num1=10.0, num2=2.0)

# run pipeline
# pipeline_task = pipeline(tracking_uri="http://localhost:8080")

