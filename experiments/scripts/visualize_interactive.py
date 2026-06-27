"""
Interactive visualization script.
Visualizes trained probe results using Plotly.
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
from src.probes.linear_probe import LinearProbe, MultiLayerProbe
from src.data.dataset import KnowledgeDataset
from src.training.evaluator import ProbeEvaluator
from src.visualization.interactive_plotter import (
    plot_layer_confidence_interactive,
    plot_confidence_heatmap,
    plot_confidence_distribution_interactive,
    plot_layer_comparison_interactive,
    create_dashboard
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_probe(checkpoint_path: Path, device: str = "mps"):
    """Load a probe from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine probe type
    if 'num_layers' in checkpoint:
        # MultiLayerProbe
        probe = MultiLayerProbe(
            input_dim=checkpoint['input_dim'],
            num_layers=checkpoint['num_layers'],
            aggregation=checkpoint.get('aggregation', 'concat')
        )
    else:
        # LinearProbe
        probe = LinearProbe(input_dim=checkpoint['input_dim'])
    
    probe.load_state_dict(checkpoint['model_state_dict'])
    probe.to(device)
    probe.eval()
    
    return probe, checkpoint


def extract_layer_scores(
    model_loader: ModelLoader,
    extractor: HiddenStateExtractor,
    dataset: KnowledgeDataset,
    layers: list,
    device: str = "mps"
):
    """Extract confidence scores for each layer."""
    layer_scores = {}

    for layer_idx in layers:
        logger.info(f"Processing layer {layer_idx}...")

        # Search for checkpoint in multiple locations
        possible_paths = [
            Path(f"experiments/results/probe_layer_{layer_idx}.pt"),
            Path(f"results/experiments/probe_layer_{layer_idx}.pt"),
        ]
        
        # Also search for the latest multi-layer model
        results_dir = Path("results/experiments")
        if results_dir.exists():
            for exp_dir in sorted(results_dir.iterdir(), reverse=True):
                if exp_dir.is_dir():
                    possible_paths.append(exp_dir / "best_model.pt")
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            logger.warning(f"Checkpoint not found for layer {layer_idx}")
            logger.info(f"Searched paths: {[str(p) for p in possible_paths[:3]]}")
            continue
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        probe, _ = load_probe(checkpoint_path, device)
        
        # Compute scores
        scores = []
        for i in range(len(dataset)):
            question, label = dataset[i]
            
            # Extract hidden states
            hidden_states = extractor.extract_and_pool(question, layers=[layer_idx], pooling="last")
            features = hidden_states[0].to(device)
            
            # Predict score
            with torch.no_grad():
                score = probe(features).item()
            scores.append(score)
        
        layer_scores[layer_idx] = scores
    
    return layer_scores


def main():
    parser = argparse.ArgumentParser(description="Interactive visualization of probe results")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/knowledge_dataset.json",
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
        default=20,
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
        labels = [dataset.labels[i] for i in indices]
        dataset = KnowledgeDataset(questions, labels)
    
    # Parse layer list
    layers = [int(x) for x in args.layers.split(",")]
    logger.info(f"Analyzing layers: {layers}")
    
    # Extract per-layer scores
    logger.info("Extracting layer scores...")
    layer_scores = extract_layer_scores(
        model_loader,
        extractor,
        dataset,
        layers,
        device
    )
    
    if not layer_scores:
        logger.error("No layer scores extracted. Check checkpoint paths.")
        return
    
    # Prepare test data (use last layer)
    last_layer = max(layer_scores.keys())
    test_scores = np.array(layer_scores[last_layer])
    test_labels = np.array(dataset.labels)
    
    # Compute per-layer metrics
    logger.info("Computing layer metrics...")
    layer_metrics = {}
    for layer_idx in layers:
        if layer_idx not in layer_scores:
            continue
        
        checkpoint_path = Path(f"experiments/results/probe_layer_{layer_idx}.pt")
        probe, _ = load_probe(checkpoint_path, device)
        
        # Create evaluator
        evaluator = ProbeEvaluator(probe, device)

        # Extract features
        features_list = []
        for question in dataset.questions:
            hidden_states = extractor.extract_and_pool(question, layers=[layer_idx], pooling="last")
            features_list.append(hidden_states[0])
        
        features = torch.stack(features_list)
        labels_tensor = torch.tensor(dataset.labels, dtype=torch.float32)
        
        # Evaluate
        metrics = evaluator.evaluate(features, labels_tensor)
        layer_metrics[layer_idx] = metrics
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # 1. Per-layer confidence (interactive)
    logger.info("1. Layer confidence (interactive)...")
    plot_layer_confidence_interactive(
        layer_scores,
        questions=dataset.questions,
        save_path=output_dir / "layer_confidence_interactive.html",
        title=f"Confidence Scores by Layer - {args.model_name}"
    )
    
    # 2. Heatmap
    logger.info("2. Confidence heatmap...")
    plot_confidence_heatmap(
        layer_scores,
        questions=dataset.questions,
        save_path=output_dir / "confidence_heatmap.html",
        title=f"Confidence Heatmap - {args.model_name}"
    )
    
    # 3. Confidence distribution (interactive)
    logger.info("3. Confidence distribution (interactive)...")
    plot_confidence_distribution_interactive(
        test_scores,
        test_labels,
        save_path=output_dir / "confidence_distribution_interactive.html",
        title=f"Confidence Distribution - {args.model_name}"
    )
    
    # 4. Layer comparison (interactive)
    logger.info("4. Layer comparison (interactive)...")
    plot_layer_comparison_interactive(
        layer_metrics,
        save_path=output_dir / "layer_comparison_interactive.html",
        title=f"Layer Performance Comparison - {args.model_name}"
    )
    
    # 5. Dashboard
    logger.info("5. Creating dashboard...")
    create_dashboard(
        layer_scores,
        test_scores,
        test_labels,
        layer_metrics,
        questions=dataset.questions,
        save_path=output_dir / "dashboard.html"
    )
    
    logger.info(f"\nAll visualizations saved to: {output_dir}")
    logger.info("Open the HTML files in a web browser to view them.")
    
    # Display summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Layers analyzed: {layers}")
    print(f"Number of samples: {len(dataset)}")
    print(f"\nGenerated files:")
    print(f"  - layer_confidence_interactive.html")
    print(f"  - confidence_heatmap.html")
    print(f"  - confidence_distribution_interactive.html")
    print(f"  - layer_comparison_interactive.html")
    print(f"  - dashboard.html")
    print("="*60)


if __name__ == "__main__":
    main()
