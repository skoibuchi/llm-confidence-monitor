"""
Visualization - plotting experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_layer_confidence(
    layer_scores: Dict[int, List[float]],
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confidence Scores by Layer"
):
    """
    Display per-layer confidence scores as a line chart.

    Args:
        layer_scores: Scores per layer {layer_idx: [scores]}
        labels: Labels for each sample (used in legend)
        save_path: Path to save the figure
        title: Chart title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot each sample
    num_samples = len(next(iter(layer_scores.values())))
    layers = sorted(layer_scores.keys())
    
    for sample_idx in range(num_samples):
        scores = [layer_scores[layer][sample_idx] for layer in layers]
        label = labels[sample_idx] if labels else f"Sample {sample_idx}"
        plt.plot(layers, scores, marker='o', label=label, alpha=0.7)
    
    plt.xlabel("Layer Index")
    plt.ylabel("Confidence Score")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confidence_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    threshold: float = 0.5,
    title: str = "Confidence Score Distribution"
):
    """
    Display the confidence score distribution as a histogram.

    Args:
        scores: Confidence scores (n_samples,)
        labels: Ground-truth labels (n_samples,)
        save_path: Path to save the figure
        threshold: Binary classification threshold
        title: Chart title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    axes[0].hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Overall Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution by label
    binary_labels = (labels >= threshold).astype(int)
    knows = scores[binary_labels == 1]
    not_knows = scores[binary_labels == 0]
    
    axes[1].hist(knows, bins=20, alpha=0.6, color='green', label='Knows (label ≥ 0.5)', edgecolor='black')
    axes[1].hist(not_knows, bins=20, alpha=0.6, color='red', label='Does not know (label < 0.5)', edgecolor='black')
    axes[1].axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    axes[1].set_xlabel("Confidence Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution by Label")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """
    Display the confusion matrix as a heatmap.

    Args:
        y_true: Ground-truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        save_path: Path to save the figure
        title: Chart title
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Does not know', 'Knows'],
        yticklabels=['Does not know', 'Knows'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve"
):
    """
    Display the ROC curve.

    Args:
        y_true: Ground-truth labels (n_samples,)
        y_scores: Confidence scores (n_samples,)
        save_path: Path to save the figure
        title: Chart title
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training History"
):
    """
    Display the training history.

    Args:
        history: Training history {'train_loss': [...], 'val_loss': [...]}
        save_path: Path to save the figure
        title: Chart title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_layer_comparison(
    layer_metrics: Dict[int, Dict[str, float]],
    metric_name: str = 'accuracy',
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Compare performance across layers.

    Args:
        layer_metrics: Per-layer evaluation metrics {layer_idx: {'accuracy': ..., 'mse': ...}}
        metric_name: Name of the metric to display
        save_path: Path to save the figure
        title: Chart title
    """
    layers = sorted(layer_metrics.keys())
    values = [layer_metrics[layer][metric_name] for layer in layers]
    
    plt.figure(figsize=(12, 6))
    plt.bar(layers, values, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel("Layer Index")
    plt.ylabel(metric_name.upper())
    plt.title(title or f"{metric_name.upper()} by Layer")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function for testing."""
    print("=== Test: Visualization Functions ===")

    # Dummy data
    np.random.seed(42)
    
    print("\n1. Layer Confidence Plot")
    layer_scores = {
        i: np.random.rand(3).tolist() for i in range(13)
    }
    labels = ["Question 1", "Question 2", "Question 3"]
    plot_layer_confidence(layer_scores, labels, title="Test: Layer Confidence")
    
    print("\n2. Confidence Distribution")
    scores = np.random.rand(200)
    labels_data = np.random.rand(200)
    plot_confidence_distribution(scores, labels_data, title="Test: Distribution")
    
    print("\n3. Confusion Matrix")
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    plot_confusion_matrix(y_true, y_pred, title="Test: Confusion Matrix")
    
    print("\n4. ROC Curve")
    y_true_roc = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    plot_roc_curve(y_true_roc, y_scores, title="Test: ROC Curve")
    
    print("\n5. Training History")
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.28]
    }
    plot_training_history(history, title="Test: Training History")
    
    print("\n6. Layer Comparison")
    layer_metrics = {
        i: {'accuracy': 0.5 + i * 0.03, 'mse': 0.3 - i * 0.01}
        for i in range(13)
    }
    plot_layer_comparison(layer_metrics, metric_name='accuracy', title="Test: Layer Comparison")
    
    print("\nAll visualization tests completed!")


if __name__ == "__main__":
    main()
