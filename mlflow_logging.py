# mlflow_logging.py
import os
import torch
import mlflow
import mlflow.sklearn as skl
import mlflow.tensorflow as tf
import mlflow.pytorch as pytorch
from typing import Dict, Any
from urllib.parse import urlparse


def create_experiment(
    experiment_name: str,
    run_name: str,
    run_metrics: Dict[str, Any],
    model,
    TRACKING_URI: str,
    model_name: str = "pred_model",
    artifact_paths: Dict[str, str] = {},
    run_params: Dict[str, Any] = {},
    tag_dict: Dict[str, str] = {
        "tag1": "Linear Regression",
        "tag2": "House Rent Prediction",
    },
):
    try:
        # You can get your MLlfow tracking uri from your dagshub repo by opening "Remote" dropdown menu, go to "Experiments" tab and copy the MLflow experiment tracking uri and paste below
        mlflow.set_tracking_uri(str(TRACKING_URI))

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            # log params
            if run_params is not None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])

            # log metrics
            for metric, value in run_metrics.items():
                if isinstance(value, list):
                    # If the metric is a list, log each value as a separate step
                    for step, v in enumerate(value):
                        mlflow.log_metric(metric, v, step=step)
                else:
                    # If it's a single value, log it normally
                    mlflow.log_metric(metric, value)

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme

            # log artifacts
            for artifact_name, path in artifact_paths.items():
                if path and os.path.exists(path):
                    if tracking_url_type_store != "file":
                        mlflow.log_artifact(
                            path,
                            artifact_name
                        )
                elif path:
                    print(f"Warning: Artifact file not found: {path}")

            # log model
            if tracking_url_type_store != "file":
                # Check model type and log accordingly
                if hasattr(model, 'predict_proba'):  # sklearn model
                    skl.log_model(model, model_name)
                elif isinstance(model, torch.nn.Module):  # pytorch model
                    pytorch.log_model(model, model_name)
                elif hasattr(model, 'fit'): # tensorflow model  
                    tf.log_model(model, model_name)
                else:
                    print(f"Warning: Unknown model type - {type(model)}. Model not logged.")

            mlflow.set_tags(tag_dict)

        print(f"Run - {run_name} is logged to Experiment - {experiment_name}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
