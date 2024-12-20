# Yolo Training Pipelines

This repo is in progress, it explores different options to train yolo for a data set of Uno Cards images.

At some point it should cover the whole lifecycle of training a yolo model:

- Experimentation using Jupyter
- Externalize code from Jupyter to python files and functions
- Create a training Kubeflow pipeline, including input and output of the models
- Use MLFlow to track metrics and models
- Use Optuna for fine tuning hyperparameters
- Simplify with container components
- Serve the model with KServe/KFServing

So far most of the work is than but scattered...

# Buckets needed

datasets
mlflow
models
pipelines

# Refererences

https://learnopencv.com/ultralytics-yolov8/
https://learnopencv.com/train-yolov8-on-custom-dataset/
https://docs.ultralytics.com/integrations/mlflow/#how-to-use


https://rh-aiservices-bu.github.io/parasol-insurance/modules/04-05-model-serving.html