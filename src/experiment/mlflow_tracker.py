"""
MLflow experiment tracker - records parameters, metrics, and artifacts.
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Wraps MLflow to provide a simple interface for logging
    experiment parameters, metrics, models, and artifacts.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local ./mlruns)
            artifact_location: Custom artifact storage path
        """
        self.experiment_name = experiment_name

        # Set tracking URI (default: local directory)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")

        mlflow.set_experiment(experiment_name)

        if artifact_location:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )

        self.run = None
        self.run_id = None

        logger.info(f"MLflowTracker initialized")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new experiment run.

        Args:
            run_name: Human-readable name for this run
            tags: Key-value tags to attach to the run
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self.run.info.run_id
        logger.info(f"Started run: {self.run_id}")

        return self.run

    def end_run(self):
        """End the current experiment run."""
        if self.run:
            mlflow.end_run()
            logger.info(f"Ended run: {self.run_id}")
            self.run = None
            self.run_id = None

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter name → value
        """
        for key, value in params.items():
            # MLflow only supports str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))

        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log numeric metrics.

        Args:
            metrics: Dictionary of metric name → value
            step: Training step or epoch number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

        logger.debug(f"Logged {len(metrics)} metrics at step {step}")

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log a PyTorch model as an artifact.

        Args:
            model: PyTorch module to log
            artifact_path: Path within the artifact store
            registered_model_name: Optional name for model registry
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged model to {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to the local file
            artifact_path: Destination path within the artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Data to log
            filename: Artifact filename
        """
        mlflow.log_dict(dictionary, filename)
        logger.info(f"Logged dict as {filename}")

    def log_figure(self, figure, filename: str):
        """
        Log a Matplotlib figure as an artifact.

        Args:
            figure: Matplotlib figure object
            filename: Artifact filename
        """
        mlflow.log_figure(figure, filename)
        logger.info(f"Logged figure: {filename}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tag name → value
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)

        logger.info(f"Set {len(tags)} tags")

    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """
        Return metadata about the current run.

        Returns:
            dict: Run information, or None if no run is active
        """
        if self.run is None:
            return None

        return {
            "run_id": self.run.info.run_id,
            "experiment_id": self.run.info.experiment_id,
            "status": self.run.info.status,
            "start_time": self.run.info.start_time,
            "artifact_uri": self.run.info.artifact_uri,
        }


class ExperimentLogger:
    """
    Convenience wrapper around MLflowTracker for the standard
    train → validate → test workflow.
    """

    def __init__(self, tracker: MLflowTracker):
        """
        Args:
            tracker: MLflowTracker instance to delegate to
        """
        self.tracker = tracker
        self.epoch_metrics = []

    def log_training_config(
        self,
        model_name: str,
        dataset_size: int,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        **kwargs
    ):
        """
        Log training hyperparameters.

        Args:
            model_name: Name of the base LLM
            dataset_size: Total number of training samples
            batch_size: Mini-batch size
            learning_rate: Initial learning rate
            num_epochs: Total training epochs
            **kwargs: Additional parameters to log
        """
        params = {
            "model_name": model_name,
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            **kwargs
        }
        self.tracker.log_params(params)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log per-epoch metrics.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            train_metrics: Additional training metrics
            val_metrics: Additional validation metrics
        """
        metrics = {"train_loss": train_loss}

        if val_loss is not None:
            metrics["val_loss"] = val_loss

        if train_metrics:
            for key, value in train_metrics.items():
                metrics[f"train_{key}"] = value

        if val_metrics:
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value

        self.tracker.log_metrics(metrics, step=epoch)
        self.epoch_metrics.append({"epoch": epoch, **metrics})

    def log_final_results(
        self,
        test_metrics: Dict[str, float],
        best_epoch: Optional[int] = None,
        best_val_loss: Optional[float] = None
    ):
        """
        Log final test-set results and best-model metadata.

        Args:
            test_metrics: Evaluation metrics on the test set
            best_epoch: Epoch at which the best model was saved
            best_val_loss: Validation loss of the best model
        """
        for key, value in test_metrics.items():
            self.tracker.log_metrics({f"test_{key}": value})

        if best_epoch is not None:
            self.tracker.log_params({"best_epoch": best_epoch})

        if best_val_loss is not None:
            self.tracker.log_metrics({"best_val_loss": best_val_loss})

        # Save full epoch history as JSON artifact
        self.tracker.log_dict(
            {"epochs": self.epoch_metrics},
            "epoch_metrics.json"
        )


def main():
    """Smoke test for MLflowTracker."""
    print("=== Test: MLflow Tracker ===")

    tracker = MLflowTracker("test_experiment")

    tracker.start_run(
        run_name="test_run",
        tags={"model": "gpt2", "task": "knowledge_probe"}
    )

    tracker.log_params({
        "model_name": "gpt2",
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 10
    })

    for epoch in range(5):
        tracker.log_metrics({
            "train_loss": 0.5 - epoch * 0.05,
            "val_loss": 0.6 - epoch * 0.04,
            "accuracy": 0.6 + epoch * 0.05
        }, step=epoch)

    tracker.log_dict(
        {"test_accuracy": 0.85, "test_f1": 0.82},
        "test_results.json"
    )

    info = tracker.get_run_info()
    print("\nRun Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    tracker.end_run()

    print("\nTest completed!")
    print(f"View results: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
