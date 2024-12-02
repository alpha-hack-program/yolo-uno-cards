from typing import Dict

from kfp import compiler

from kfp import dsl
from kfp.dsl import Input, Metrics

# Component to validate the metrics. It receives a list of tuples with the metric name and the threshold and the metrics as input.
@dsl.component(
    # base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111",
    packages_to_install=["onnx==1.16.1", "onnxruntime==1.18.0", "scikit-learn==1.5.0", "numpy==1.24.3", "pandas==2.2.2"]
)
def validate_metrics(thresholds: Dict[str, float], metrics_input: Input[Metrics]):
    print(f"thresholds: {thresholds}")

    # For each threshold, check if the metric is above the threshold
    for metric_name, threshold in thresholds.items():
        print(f"metric_name: {metric_name}, threshold: {threshold}")
        metric_value = float(metrics_input.metadata[metric_name])
        print(f"metric_value: {metric_value}")
        # Make sure metric_value and threshold are floats
        metric_value = float(metric_value)
        threshold = float(threshold)
        # If the metric is below the threshold, raise a ValueError
        if metric_value <= threshold:
            raise ValueError(f"{metric_name} is below the threshold of {threshold}")

if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=validate_metrics,
        package_path=component_package_path
    )
