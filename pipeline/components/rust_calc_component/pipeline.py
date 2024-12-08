import kfp
from kfp.dsl import component, pipeline, Input, Output, Artifact, OutputPath

BASE_IMAGE="python:3.11-slim-bullseye"

multiply_component = kfp.components.load_component_from_file("component.yaml")

# This component parses the metrics and extracts the accuracy
@component(
    base_image=BASE_IMAGE
)
def parse_value(value_input: Input[Artifact], value_output: OutputPath(float)):
    # Read the content of the input file
    with open(value_input.path, 'r') as f:
        value = float(f.read())

    # Write the value to the output file
    with open(value_output, 'w') as f:
        f.write(str(value))

@pipeline(name='multiply-numbers-pipeline')
def multiply_pipeline(num1: float, num2: float) -> float:
    multiply_component_task = multiply_component(num1=num1, num2=num2)

    # Parse the value from the output of the multiply_component
    parse_value_task = parse_value(value_input=multiply_component_task.outputs["result"])

    # Return the content of the output file
    return parse_value_task.outputs["value_output"]

if __name__ == '__main__':

    from kfp import compiler

    pipeline_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=multiply_pipeline,
        package_path=pipeline_package_path
    )