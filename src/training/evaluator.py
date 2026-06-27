"""
Probe evaluator - computes classification and regression metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Callable, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProbeEvaluator:
    """Evaluates a trained probe on a held-out dataset."""

    def __init__(
        self,
        probe: nn.Module,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            probe: Trained probe to evaluate
            device: Target device
            threshold: Decision boundary for binary classification
        """
        self.probe = probe
        self.device = device or torch.device("cpu")
        self.threshold = threshold

        self.probe.to(self.device)
        self.probe.eval()

        logger.info(f"ProbeEvaluator initialized on device: {self.device}")

    def evaluate(
        self,
        dataloader: DataLoader,
        extract_features_fn: Callable,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the probe on a full dataset.

        Args:
            dataloader: DataLoader for the evaluation set
            extract_features_fn: Function that takes a batch and returns features
            verbose: Show tqdm progress bar

        Returns:
            Dict: Evaluation metrics (MSE, accuracy, F1, ROC-AUC, etc.)
        """
        all_predictions = []
        all_labels = []

        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        with torch.no_grad():
            for batch in iterator:
                features = extract_features_fn(batch)

                if isinstance(features, list):
                    features = [f.to(self.device) for f in features]
                else:
                    features = features.to(self.device)

                labels = batch['label']
                predictions = self.probe(features)

                all_predictions.append(predictions.cpu())
                all_labels.append(labels)

        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        return self.compute_metrics(all_predictions, all_labels)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute regression and binary classification metrics.

        Args:
            predictions: Predicted scores of shape (n_samples,) or (n_samples, 1)
            labels: Ground-truth labels of shape (n_samples,) or (n_samples, 1)

        Returns:
            Dict: Metrics including MSE, MAE, RMSE, R², accuracy, precision,
                  recall, F1, ROC-AUC, and confusion matrix entries.
        """
        predictions = predictions.flatten()
        labels = labels.flatten()

        metrics = {}

        # Regression metrics
        metrics['mse'] = mean_squared_error(labels, predictions)
        metrics['mae'] = mean_absolute_error(labels, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])

        try:
            metrics['r2'] = r2_score(labels, predictions)
        except Exception:
            metrics['r2'] = 0.0

        # Binary classification metrics
        binary_predictions = (predictions >= self.threshold).astype(int)
        binary_labels = (labels >= self.threshold).astype(int)

        metrics['accuracy'] = accuracy_score(binary_labels, binary_predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels,
            binary_predictions,
            average='binary',
            zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        try:
            metrics['roc_auc'] = roc_auc_score(binary_labels, predictions)
        except Exception:
            metrics['roc_auc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(binary_labels, binary_predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)

        return metrics

    def evaluate_by_layer(
        self,
        dataloader: DataLoader,
        extract_features_fn: Callable,
        layer_indices: List[int],
        verbose: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate the probe separately for each specified layer.

        Args:
            dataloader: DataLoader for the evaluation set
            extract_features_fn: Function accepting (batch, layer_idx)
            layer_indices: Layer indices to evaluate
            verbose: Show per-layer progress

        Returns:
            Dict[int, Dict]: Metrics per layer index
        """
        results = {}

        for layer_idx in layer_indices:
            if verbose:
                print(f"\nEvaluating layer {layer_idx}...")

            def layer_extract_fn(batch):
                return extract_features_fn(batch, layer_idx)

            metrics = self.evaluate(dataloader, layer_extract_fn, verbose=False)
            results[layer_idx] = metrics

            if verbose:
                print(f"Layer {layer_idx} - MSE: {metrics['mse']:.4f}, "
                      f"Accuracy: {metrics['accuracy']:.4f}")

        return results

    def compute_calibration_error(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute the Expected Calibration Error (ECE).

        Args:
            predictions: Predicted probabilities of shape (n_samples,)
            labels: Ground-truth labels of shape (n_samples,)
            n_bins: Number of confidence bins

        Returns:
            float: ECE value
        """
        predictions = predictions.flatten()
        labels = labels.flatten()

        binary_labels = (labels >= self.threshold).astype(int)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        total_samples = len(predictions)

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])

            if mask.sum() > 0:
                bin_predictions = predictions[mask]
                bin_labels = binary_labels[mask]

                avg_confidence = bin_predictions.mean()
                avg_accuracy = bin_labels.mean()

                ece += abs(avg_confidence - avg_accuracy) * mask.sum() / total_samples

        return ece

    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in a formatted table.

        Args:
            metrics: Dictionary returned by compute_metrics()
        """
        print("\n=== Evaluation Metrics ===")

        print("\nRegression Metrics:")
        print(f"  MSE:  {metrics.get('mse', 0):.4f}")
        print(f"  MAE:  {metrics.get('mae', 0):.4f}")
        print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"  R²:   {metrics.get('r2', 0):.4f}")

        print("\nClassification Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
        print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")

        if 'true_positive' in metrics:
            print("\nConfusion Matrix:")
            print(f"  True Positive:  {metrics['true_positive']}")
            print(f"  False Positive: {metrics['false_positive']}")
            print(f"  True Negative:  {metrics['true_negative']}")
            print(f"  False Negative: {metrics['false_negative']}")


def main():
    """Smoke test for ProbeEvaluator."""
    from src.probes.linear_probe import LinearProbe

    print("=== Test: ProbeEvaluator ===")

    input_dim = 768
    probe = LinearProbe(input_dim)

    evaluator = ProbeEvaluator(probe, threshold=0.5)

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'features': torch.randn(input_dim),
                'label': torch.rand(1)
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    def extract_features(batch):
        return batch['features']

    print("\nEvaluating...")
    metrics = evaluator.evaluate(dataloader, extract_features, verbose=True)

    evaluator.print_metrics(metrics)

    print("\n=== Test: Calibration Error ===")
    predictions = np.random.rand(100)
    labels = np.random.rand(100)
    ece = evaluator.compute_calibration_error(predictions, labels)
    print(f"Expected Calibration Error: {ece:.4f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
