"""
Visualization script for the multi-layer integrated probe.
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_loader import ModelLoader
from src.models.hidden_extractor import HiddenStateExtractor
from src.probes.linear_probe import MultiLayerProbe
from src.data.dataset import KnowledgeDataset
from src.visualization.interactive_plotter import (
    plot_confidence_distribution_interactive,
    create_dashboard
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_multi_layer_probe(checkpoint_path: Path, device: str = "mps"):
    """Load a multi-layer probe from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect checkpoint format
    if 'probe_state_dict' in checkpoint:
        # Format saved by the trainer
        state_dict = checkpoint['probe_state_dict']

        # Infer probe config from state_dict
        if 'layer_weights' in state_dict:
            # Weighted aggregation
            num_layers = state_dict['layer_weights'].shape[0]
            aggregation = 'weighted'
        elif 'linear.weight' in state_dict:
            # Concat or Mean
            linear_weight_shape = state_dict['linear.weight'].shape
            # linear.weight: (output_dim, input_dim * num_layers) for concat
            # linear.weight: (output_dim, input_dim) for mean
            # Check if it is a multiple of 768
            input_features = linear_weight_shape[1]
            if input_features % 768 == 0:
                num_layers = input_features // 768
                aggregation = 'concat' if num_layers > 1 else 'mean'
            else:
                num_layers = 3  # default
                aggregation = 'mean'
        else:
            num_layers = 3
            aggregation = 'concat'

        input_dim = 768  # GPT-2 default

    else:
        # Legacy format (saved directly)
        input_dim = checkpoint.get('input_dim', 768)
        num_layers = checkpoint.get('num_layers', 3)
        aggregation = checkpoint.get('aggregation', 'concat')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    probe = MultiLayerProbe(
        input_dim=input_dim,
        num_layers=num_layers,
        aggregation=aggregation
    )
    
    probe.load_state_dict(state_dict)
    probe.to(device)
    probe.eval()
    
    logger.info(f"Loaded MultiLayerProbe:")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Num layers: {num_layers}")
    logger.info(f"  Aggregation: {aggregation}")
    
    return probe, checkpoint


def extract_scores(
    model_loader: ModelLoader,
    extractor: HiddenStateExtractor,
    probe: MultiLayerProbe,
    dataset: KnowledgeDataset,
    layers: list,
    device: str = "mps"
):
    """Extract scores using the multi-layer probe."""
    scores = []
    
    for i in range(len(dataset)):
        question, label = dataset[i]
        
        # Extract multi-layer hidden states
        hidden_states = extractor.extract_and_pool(question, layers=layers, pooling="last")

        # Stack into list
        features = [h.to(device) for h in hidden_states]

        # Predict score
        with torch.no_grad():
            score = probe(features).item()
        scores.append(score)
    
    return np.array(scores)


def plot_sample_predictions(
    questions: list,
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    num_samples: int = 20
):
    """Visualize per-sample predictions."""
    # Cap number of samples
    indices = np.random.choice(len(questions), min(num_samples, len(questions)), replace=False)
    
    fig = go.Figure()
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=list(range(len(indices))),
        y=predictions[indices],
        mode='markers',
        name='Prediction',
        marker=dict(size=12, color='blue', symbol='circle'),
        text=[questions[i][:50] + "..." if len(questions[i]) > 50 else questions[i] for i in indices],
        hovertemplate="<b>%{text}</b><br>" +
                     "Prediction: %{y:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    # Ground-truth labels
    fig.add_trace(go.Scatter(
        x=list(range(len(indices))),
        y=labels[indices],
        mode='markers',
        name='Ground Truth',
        marker=dict(size=12, color='red', symbol='x'),
        text=[questions[i][:50] + "..." if len(questions[i]) > 50 else questions[i] for i in indices],
        hovertemplate="<b>%{text}</b><br>" +
                     "Ground Truth: %{y:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    # Threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold (0.5)")
    
    fig.update_layout(
        title="Sample Predictions vs Ground Truth",
        xaxis_title="Sample Index",
        yaxis_title="Confidence Score",
        template='plotly_white',
        width=1200,
        height=600,
        hovermode='closest'
    )
    
    fig.write_html(str(save_path))
    logger.info(f"Sample predictions plot saved to {save_path}")


def plot_error_analysis(
    questions: list,
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Path
):
    """Visualize prediction error analysis."""
    errors = np.abs(predictions - labels)

    # Sort by largest error first
    sorted_indices = np.argsort(errors)[::-1]
    top_errors = sorted_indices[:20]
    
    fig = go.Figure()
    
    # Prepare custom data
    custom_data = [[predictions[i], labels[i]] for i in top_errors]
    
    fig.add_trace(go.Bar(
        x=list(range(len(top_errors))),
        y=errors[top_errors],
        text=[questions[i][:30] + "..." if len(questions[i]) > 30 else questions[i] for i in top_errors],
        customdata=custom_data,
        hovertemplate="<b>%{text}</b><br>" +
                     "Error: %{y:.3f}<br>" +
                     "Prediction: %{customdata[0]:.3f}<br>" +
                     "Ground Truth: %{customdata[1]:.3f}<br>" +
                     "<extra></extra>",
        marker_color='red'
    ))
    
    fig.update_layout(
        title="Top 20 Prediction Errors",
        xaxis_title="Sample Rank",
        yaxis_title="Absolute Error",
        template='plotly_white',
        width=1200,
        height=600
    )
    
    fig.write_html(str(save_path))
    logger.info(f"Error analysis plot saved to {save_path}")


def plot_confidence_scatter(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Path
):
    """Scatter plot of predictions vs. ground truth."""
    fig = go.Figure()

    # Color-code by knows / does not know
    knows_mask = labels >= 0.5
    
    fig.add_trace(go.Scatter(
        x=labels[knows_mask],
        y=predictions[knows_mask],
        mode='markers',
        name='Knows',
        marker=dict(size=8, color='green', opacity=0.6),
        hovertemplate="Ground Truth: %{x:.3f}<br>" +
                     "Prediction: %{y:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=labels[~knows_mask],
        y=predictions[~knows_mask],
        mode='markers',
        name='Does not know',
        marker=dict(size=8, color='red', opacity=0.6),
        hovertemplate="Ground Truth: %{x:.3f}<br>" +
                     "Prediction: %{y:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    # Diagonal line (perfect prediction)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        title="Prediction vs Ground Truth Scatter Plot",
        xaxis_title="Ground Truth Confidence",
        yaxis_title="Predicted Confidence",
        template='plotly_white',
        width=800,
        height=800
    )
    
    fig.write_html(str(save_path))
    logger.info(f"Confidence scatter plot saved to {save_path}")


def create_multi_layer_dashboard(
    questions: list,
    predictions: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    save_path: Path
):
    """Create a dashboard for the multi-layer probe."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction vs Ground Truth',
            'Confidence Distribution',
            'Sample Predictions',
            'Metrics Summary'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Scatter plot
    knows_mask = labels >= 0.5
    fig.add_trace(
        go.Scatter(
            x=labels[knows_mask],
            y=predictions[knows_mask],
            mode='markers',
            name='Knows',
            marker=dict(size=6, color='green', opacity=0.6),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=labels[~knows_mask],
            y=predictions[~knows_mask],
            mode='markers',
            name='Does not know',
            marker=dict(size=6, color='red', opacity=0.6),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Distribution
    binary_labels = (labels >= 0.5).astype(int)
    for label, color, name in [(1, 'green', 'Knows'), (0, 'red', 'Does not know')]:
        mask = binary_labels == label
        fig.add_trace(
            go.Histogram(
                x=predictions[mask],
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=20,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Sample predictions (first 10 samples)
    num_samples = min(10, len(questions))
    fig.add_trace(
        go.Scatter(
            x=list(range(num_samples)),
            y=predictions[:num_samples],
            mode='markers+lines',
            name='Prediction',
            marker=dict(size=10, color='blue')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_samples)),
            y=labels[:num_samples],
            mode='markers',
            name='Ground Truth',
            marker=dict(size=10, color='red', symbol='x')
        ),
        row=2, col=1
    )
    
    # 4. Metrics table
    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{metrics.get('accuracy', 0):.3f}"],
        ["F1 Score", f"{metrics.get('f1', 0):.3f}"],
        ["ROC AUC", f"{metrics.get('roc_auc', 0):.3f}"],
        ["MSE", f"{metrics.get('mse', 0):.4f}"],
        ["MAE", f"{metrics.get('mae', 0):.4f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=["<b>Metric</b>", "<b>Value</b>"],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[[row[0] for row in metrics_data[1:]],
                              [row[1] for row in metrics_data[1:]]],
                      fill_color='lavender',
                      align='left')
        ),
        row=2, col=2
    )
    
    # Update layout axes
    fig.update_xaxes(title_text="Ground Truth", row=1, col=1)
    fig.update_yaxes(title_text="Prediction", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    
    fig.update_layout(
        title_text="Multi-Layer Probe Analysis Dashboard",
        template='plotly_white',
        width=1400,
        height=1000,
        showlegend=True
    )
    
    fig.write_html(str(save_path))
    logger.info(f"Dashboard saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-layer probe results")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained multi-layer probe checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/raw/large_dataset.jsonl",
        help="Path to dataset"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,6,11",
        help="Comma-separated layer indices"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments",
        help="Output directory for visualizations (recommended: same folder as checkpoint)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to visualize"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device settings
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and dataset
    logger.info("Loading model and dataset...")
    model_loader = ModelLoader(args.model_name, device=device)
    model, tokenizer = model_loader.load()
    extractor = HiddenStateExtractor(model, tokenizer, device=device)
    
    dataset = KnowledgeDataset.load(args.dataset_path)
    
    # Limit sample count
    if len(dataset) > args.num_samples:
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        questions = [dataset.questions[i] for i in indices]
        labels = np.array([dataset.labels[i] for i in indices])
        dataset = KnowledgeDataset(questions, labels.tolist())
    
    # Parse layer list
    layers = [int(x) for x in args.layers.split(",")]
    logger.info(f"Using layers: {layers}")
    
    # Load probe
    logger.info(f"Loading checkpoint: {args.checkpoint_path}")
    probe, checkpoint = load_multi_layer_probe(Path(args.checkpoint_path), device)
    
    # Extract scores
    logger.info("Extracting predictions...")
    predictions = extract_scores(
        model_loader,
        extractor,
        probe,
        dataset,
        layers,
        device
    )
    
    labels_array = np.array(dataset.labels)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
    
    binary_preds = (predictions >= 0.5).astype(int)
    binary_labels = (labels_array >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(binary_labels, binary_preds),
        'f1': f1_score(binary_labels, binary_preds),
        'roc_auc': roc_auc_score(binary_labels, predictions),
        'mse': mean_squared_error(labels_array, predictions),
        'mae': mean_absolute_error(labels_array, predictions)
    }
    
    logger.info("\nMetrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. Sample predictions
    logger.info("1. Sample predictions...")
    plot_sample_predictions(
        dataset.questions,
        predictions,
        labels_array,
        output_dir / "sample_predictions.html",
        num_samples=20
    )
    
    # 2. Error analysis
    logger.info("2. Error analysis...")
    plot_error_analysis(
        dataset.questions,
        predictions,
        labels_array,
        output_dir / "error_analysis.html"
    )
    
    # 3. Scatter plot
    logger.info("3. Confidence scatter...")
    plot_confidence_scatter(
        predictions,
        labels_array,
        output_dir / "confidence_scatter.html"
    )
    
    # 4. Confidence distribution
    logger.info("4. Confidence distribution...")
    plot_confidence_distribution_interactive(
        predictions,
        labels_array,
        save_path=output_dir / "confidence_distribution.html",
        title=f"Confidence Distribution - Multi-Layer Probe"
    )
    
    # 5. Dashboard
    logger.info("5. Creating dashboard...")
    create_multi_layer_dashboard(
        dataset.questions,
        predictions,
        labels_array,
        metrics,
        output_dir / "dashboard.html"
    )
    
    logger.info(f"\nAll visualizations saved to: {output_dir}")
    logger.info("Open the HTML files in a web browser to view them.")

    # Display summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Layers: {layers}")
    print(f"Aggregation: {checkpoint.get('aggregation', 'unknown')}")
    print(f"Number of samples: {len(dataset)}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nGenerated files:")
    print(f"  - sample_predictions.html")
    print(f"  - error_analysis.html")
    print(f"  - confidence_scatter.html")
    print(f"  - confidence_distribution.html")
    print(f"  - dashboard.html")
    print("="*60)


if __name__ == "__main__":
    main()
