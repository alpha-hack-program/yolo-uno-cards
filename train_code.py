import torch
from ultralytics import YOLO, settings
import mlflow
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Environment variables
model_name = os.getenv("MODEL_NAME", "yolov8n")  # Replace with your model name if needed
image_size = os.getenv("IMAGE_SIZE", 640)  # Replace with your image size if needed
batch_size = os.getenv("BATCH_SIZE", 2)  # Replace with your batch size if needed
epochs = os.getenv("EPOCHS", 1)  # Replace with your number of epochs if needed

experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "YOLOv8n")  # Replace with your experiment name if needed
run_name = os.getenv("MLFLOW_RUN_NAME", "yolo-run")  # Replace with your run name if needed
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")  # Replace with your URI if needed

dataset_name = os.getenv("DATASET_NAME", "uno-cards")  # Replace with your dataset name if needed
dataset_root = os.getenv("DATASET_ROOT", "./datasets")  # Replace with your dataset path if needed
dataset_yaml = os.getenv("DATASET_YAML", "data.yaml")  # Replace with your dataset YAML file if needed

# Set up MLflow experiment and tracking URI
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# Check and set device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using device: MPS")

print(f"Using device: {device}")

# Update settings for MLflow integration
settings.update({"mlflow": True})

# Reset settings to default values
settings.reset()

# Print current run name
print(f"Current run name: {run_name}")

# Start an MLflow run for logging
with mlflow.start_run(run_name=run_name) as run:
    # Log initial parameters
    mlflow.log_param("model", model_name)  # or change based on model size
    mlflow.log_param("device", device)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("imgsz", image_size)

    # Load YOLO model and begin training
    model = YOLO(f'{model_name}.pt')

    # Check if {dataset_root}/{dataset_name}/{dataset_yaml} exists
    if not os.path.exists(f'{dataset_root}/{dataset_name}/{dataset_yaml}'):
        print(f"Dataset YAML file not found at {dataset_root}/{dataset_name}/{dataset_yaml}")
        exit(1)

    # Train model and log metrics per epoch
    results = model.train(data=f'{dataset_root}/{dataset_name}/{dataset_yaml}', epochs=epochs, imgsz=image_size, batch=batch_size, device=device)
    
    print("Training results: ")
    print(vars(results))

    # Log any available training metrics to MLflow
    mlflow.log_metric("train_loss", results.metrics['loss'])
    mlflow.log_metric("box_loss", results.box_loss)
    mlflow.log_metric("cls_loss", results.cls_loss)
    mlflow.log_metric("dfl_loss", results.dfl_loss)
    mlflow.log_metric("precision", results.metrics['precision'])
    mlflow.log_metric("recall", results.metrics['recall'])
    mlflow.log_metric("mAP_50", results.metrics['mAP_50'])
    mlflow.log_metric("mAP_50_95", results.metrics['mAP_50_95'])

    # Evaluate model performance on validation set
    validation_results = model.val()

    # Log validation metrics
    mlflow.log_metric("val_accuracy", validation_results.metrics['accuracy'])  # Replace with actual validation metric
    mlflow.log_metric("val_loss", validation_results.metrics['loss'])          # Replace as needed

    # Save model as artifact in PyTorch format
    model_path = f"model-{run_name}.pt"
    model.save(model_path)
    mlflow.log_artifact(model_path)

    # Export model to ONNX and log artifact
    onnx_path = model.export(format="onnx")
    mlflow.log_artifact(onnx_path)

    # Make a sample prediction
    results = model.predict('datasets/uno-cards/test/images/000090623_jpg.rf.f0956cd698e13eeb4614b8bbf89df3de.jpg')
    print("Prediction results: ", results)

# Print final training results
print("Training results: ")
print(results)
