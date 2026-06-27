"""
Interactive visualization - advanced plotting using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_layer_confidence_interactive(
    layer_scores: Dict[int, List[float]],
    questions: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confidence Scores by Layer (Interactive)"
):
    """
    Display per-layer confidence scores as an interactive chart.

    Args:
        layer_scores: Scores per layer {layer_idx: [scores]}
        questions: Question text for each sample (shown on hover)
        save_path: Path to save the HTML file
        title: Chart title
    """
    fig = go.Figure()
    
    layers = sorted(layer_scores.keys())
    num_samples = len(next(iter(layer_scores.values())))
    
    # Plot each sample
    for sample_idx in range(num_samples):
        scores = [layer_scores[layer][sample_idx] for layer in layers]
        
        hover_text = questions[sample_idx] if questions else f"Sample {sample_idx}"
        
        fig.add_trace(go.Scatter(
            x=layers,
            y=scores,
            mode='lines+markers',
            name=f"Sample {sample_idx}",
            hovertemplate=f"<b>{hover_text}</b><br>" +
                         "Layer: %{x}<br>" +
                         "Confidence: %{y:.3f}<br>" +
                         "<extra></extra>",
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Confidence Score",
        hovermode='closest',
        template='plotly_white',
        width=1200,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Interactive plot saved to {save_path}")
    
    return fig


def plot_confidence_heatmap(
    layer_scores: Dict[int, List[float]],
    questions: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confidence Heatmap"
):
    """
    Display a confidence heatmap.

    Args:
        layer_scores: Scores per layer {layer_idx: [scores]}
        questions: Question text for each sample
        save_path: Path to save the HTML file
        title: Chart title
    """
    layers = sorted(layer_scores.keys())
    num_samples = len(next(iter(layer_scores.values())))
    
    # Convert data to matrix form
    data = np.array([[layer_scores[layer][i] for layer in layers] for i in range(num_samples)])

    # Truncate long question strings
    if questions:
        y_labels = [q[:50] + "..." if len(q) > 50 else q for q in questions]
    else:
        y_labels = [f"Sample {i}" for i in range(num_samples)]
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[f"Layer {l}" for l in layers],
        y=y_labels,
        colorscale='RdYlGn',
        colorbar=dict(title="Confidence"),
        hovertemplate="Layer: %{x}<br>" +
                     "Sample: %{y}<br>" +
                     "Confidence: %{z:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Question",
        template='plotly_white',
        width=1000,
        height=max(400, num_samples * 20),
        yaxis=dict(tickfont=dict(size=10))
    )
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Heatmap saved to {save_path}")
    
    return fig


def plot_confidence_distribution_interactive(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    threshold: float = 0.5,
    title: str = "Confidence Distribution (Interactive)"
):
    """
    Display the confidence distribution as an interactive chart.

    Args:
        scores: Confidence scores
        labels: Ground-truth labels
        save_path: Path to save the HTML file
        threshold: Decision threshold
        title: Chart title
    """
    # Create DataFrame
    binary_labels = (labels >= threshold).astype(int)
    df = pd.DataFrame({
        'Confidence': scores,
        'Label': ['Knows' if l == 1 else 'Does not know' for l in binary_labels]
    })
    
    # Create histogram
    fig = px.histogram(
        df,
        x='Confidence',
        color='Label',
        nbins=30,
        barmode='overlay',
        opacity=0.7,
        title=title,
        labels={'Confidence': 'Confidence Score', 'count': 'Frequency'},
        color_discrete_map={
            'Knows': 'green',
            'Does not know': 'red'
        }
    )
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top"
    )
    
    fig.update_layout(
        template='plotly_white',
        width=1000,
        height=600,
        hovermode='x unified'
    )
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Distribution plot saved to {save_path}")
    
    return fig


def plot_layer_comparison_interactive(
    layer_metrics: Dict[int, Dict[str, float]],
    metric_names: List[str] = ['accuracy', 'f1', 'roc_auc'],
    save_path: Optional[Path] = None,
    title: str = "Layer Performance Comparison"
):
    """
    Compare performance across layers (multiple metrics).

    Args:
        layer_metrics: Per-layer evaluation metrics
        metric_names: List of metric names to display
        save_path: Path to save the HTML file
        title: Chart title
    """
    layers = sorted(layer_metrics.keys())
    
    fig = go.Figure()
    
    for metric_name in metric_names:
        values = [layer_metrics[layer].get(metric_name, 0) for layer in layers]
        
        fig.add_trace(go.Scatter(
            x=layers,
            y=values,
            mode='lines+markers',
            name=metric_name.upper(),
            hovertemplate=f"<b>{metric_name.upper()}</b><br>" +
                         "Layer: %{x}<br>" +
                         "Value: %{y:.3f}<br>" +
                         "<extra></extra>",
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Score",
        template='plotly_white',
        width=1200,
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Comparison plot saved to {save_path}")
    
    return fig


def create_dashboard(
    layer_scores: Dict[int, List[float]],
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    layer_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    questions: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Create a comprehensive dashboard.

    Args:
        layer_scores: Scores per layer
        test_scores: Confidence scores for the test set
        test_labels: Labels for the test set
        layer_metrics: Per-layer evaluation metrics
        questions: Question text
        save_path: Path to save the HTML file
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Confidence by Layer',
            'Confidence Distribution',
            'Layer Performance',
            'Confidence Heatmap'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    layers = sorted(layer_scores.keys())
    num_samples = min(10, len(next(iter(layer_scores.values()))))  # cap at 10 samples

    # 1. Confidence by layer (limited samples)
    for sample_idx in range(num_samples):
        scores = [layer_scores[layer][sample_idx] for layer in layers]
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=scores,
                mode='lines+markers',
                name=f"Sample {sample_idx}",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Confidence distribution
    binary_labels = (test_labels >= 0.5).astype(int)
    for label, color, name in [(1, 'green', 'Knows'), (0, 'red', 'Does not know')]:
        mask = binary_labels == label
        fig.add_trace(
            go.Histogram(
                x=test_scores[mask],
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=20,
                showlegend=True
            ),
            row=1, col=2
        )
    
    # 3. Layer performance comparison
    if layer_metrics:
        for metric_name in ['accuracy', 'f1']:
            values = [layer_metrics[layer].get(metric_name, 0) for layer in layers]
            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=values,
                    mode='lines+markers',
                    name=metric_name.upper()
                ),
                row=2, col=1
            )
    
    # 4. Heatmap (limited samples)
    heatmap_data = np.array([[layer_scores[layer][i] for layer in layers] for i in range(num_samples)])
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=[f"L{l}" for l in layers],
            y=[f"S{i}" for i in range(num_samples)],
            colorscale='RdYlGn',
            showscale=True
        ),
        row=2, col=2
    )
    
    # Update layout axes
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    
    fig.update_layout(
        title_text="Knowledge Probe Analysis Dashboard",
        template='plotly_white',
        width=1400,
        height=1000,
        showlegend=True
    )
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Dashboard saved to {save_path}")
    
    return fig


def main():
    """Main function for testing."""
    print("=== Test: Interactive Visualization ===")

    # Dummy data
    np.random.seed(42)

    # Per-layer scores
    layer_scores = {
        i: np.random.rand(5).tolist() for i in range(13)
    }
    questions = [
        "What is the height of Mount Fuji?",
        "What is the chemical formula of water?",
        "What is the capital of Japan?",
        "What is the gravity on fictional planet X?",
        "Who will win the Nobel Prize in 2050?",
    ]
    
    print("\n1. Layer Confidence (Interactive)")
    fig1 = plot_layer_confidence_interactive(
        layer_scores,
        questions,
        save_path="test_layer_confidence_interactive.html"
    )
    
    print("\n2. Confidence Heatmap")
    fig2 = plot_confidence_heatmap(
        layer_scores,
        questions,
        save_path="test_heatmap.html"
    )
    
    print("\n3. Confidence Distribution (Interactive)")
    scores = np.random.rand(200)
    labels = np.random.rand(200)
    fig3 = plot_confidence_distribution_interactive(
        scores,
        labels,
        save_path="test_distribution_interactive.html"
    )
    
    print("\n4. Layer Comparison (Interactive)")
    layer_metrics = {
        i: {
            'accuracy': 0.5 + i * 0.03,
            'f1': 0.4 + i * 0.04,
            'roc_auc': 0.6 + i * 0.02
        }
        for i in range(13)
    }
    fig4 = plot_layer_comparison_interactive(
        layer_metrics,
        save_path="test_layer_comparison.html"
    )
    
    print("\n5. Dashboard")
    fig5 = create_dashboard(
        layer_scores,
        scores,
        labels,
        layer_metrics,
        questions,
        save_path="test_dashboard.html"
    )
    
    print("\nAll interactive visualizations created!")
    print("Open the HTML files in a web browser to view them.")


if __name__ == "__main__":
    main()
