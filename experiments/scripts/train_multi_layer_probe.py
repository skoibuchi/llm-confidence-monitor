"""
Training script for the multi-layer integrated probe.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
import json
from datetime import datetime
from tqdm import tqdm

from src.models.model_loader import ModelLoader
from src.models.hidden_extractor import HiddenStateExtractor
from src.data.dataset import KnowledgeProbeDataset
from src.probes.linear_probe import MultiLayerProbe
from src.training.trainer import ProbeTrainer
from src.training.evaluator import ProbeEvaluator
from src.visualization.plotter import (
    plot_training_history,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_roc_curve
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-layer knowledge probe")

    # Model settings
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model name (default: gpt2)")
    parser.add_argument("--layers", type=str, default="0,6,11",
                        help="Comma-separated layer indices (e.g., '0,6,11')")
    parser.add_argument("--aggregation", type=str, default="concat",
                        choices=["concat", "mean", "weighted"],
                        help="Aggregation method for multiple layers")
    
    # Data settings
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory containing train/val/test.jsonl")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--pooling", type=str, default="last",
                        choices=["last", "mean", "max", "cls"],
                        help="Pooling method")
    
    # Miscellaneous
    parser.add_argument("--output_dir", type=str, default="results/experiments",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (used as subdirectory prefix, e.g. 'gpt2_weighted'). "
                             "Defaults to 'multi_layer_{aggregation}'.")
    parser.add_argument("--cache_hidden_states", action="store_true",
                        help="Pre-compute all hidden states once and cache in memory. "
                             "Speeds up training significantly when num_epochs > 1 "
                             "at the cost of higher memory usage.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function."""
    args = parse_args()
    set_seed(args.seed)

    # Parse layer indices
    layer_indices = [int(x) for x in args.layers.split(',')]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = args.experiment_name if args.experiment_name else f"multi_layer_{args.aggregation}"
    exp_name = f"{base_name}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Layer Knowledge Probe Training")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Layers: {layer_indices}")
    print(f"Aggregation: {args.aggregation}")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    loader = ModelLoader(args.model_name)
    model, tokenizer = loader.load()

    model_info = loader.get_model_info()
    print(f"Model info: {model_info}")

    # Save config (include model architecture info for later use e.g. Gradio demo)
    config = vars(args).copy()
    config['layer_indices'] = layer_indices
    config['hidden_dim'] = model_info['hidden_size']
    config['num_layers'] = model_info['num_layers']
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create hidden state extractor
    extractor = HiddenStateExtractor(model, tokenizer)
    
    # Load datasets
    print(f"\nLoading datasets from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    
    train_dataset = KnowledgeProbeDataset(
        data_dir / "train.jsonl",
        tokenizer,
        return_text=False
    )
    val_dataset = KnowledgeProbeDataset(
        data_dir / "val.jsonl",
        tokenizer,
        return_text=False
    )
    test_dataset = KnowledgeProbeDataset(
        data_dir / "test.jsonl",
        tokenizer,
        return_text=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create multi-layer probe
    print(f"\nCreating multi-layer probe...")
    print(f"  Layers: {layer_indices}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Dropout: {args.dropout}")
    
    hidden_size = model_info['hidden_size']
    probe = MultiLayerProbe(
        input_dim=hidden_size,
        num_layers=len(layer_indices),
        aggregation=args.aggregation,
        dropout=args.dropout
    )
    
    # Optimizer and loss function
    optimizer = Adam(probe.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = ProbeTrainer(
        probe=probe,
        optimizer=optimizer,
        criterion=criterion,
        device=loader.device
    )
    
    def extract_features_from_batch(batch):
        """Extract and pool hidden states from a raw batch (runs the LLM)."""
        input_ids = batch['input_ids'].to(loader.device)
        attention_mask = batch['attention_mask'].to(loader.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            selected_layers = []
            for layer_idx in layer_indices:
                hidden = hidden_states[layer_idx]

                if args.pooling == "last":
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = hidden.shape[0]
                    features = hidden[torch.arange(batch_size, device=loader.device), seq_lengths]
                elif args.pooling == "mean":
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size())
                    sum_hidden = (hidden * mask_expanded).sum(dim=1)
                    sum_mask = mask_expanded.sum(dim=1)
                    features = sum_hidden / sum_mask
                elif args.pooling == "cls":
                    features = hidden[:, 0, :]
                else:
                    features = hidden[:, -1, :]

                # Cast to float32 to match probe dtype (model may use bfloat16/float16)
                selected_layers.append(features.to(torch.float32))

        return selected_layers

    # ── Hidden state cache (optional) ────────────────────────────────
    if args.cache_hidden_states:
        from torch.utils.data import TensorDataset

        def _build_cache(dataloader, desc):
            """Pre-compute hidden states for every sample in the dataloader."""
            all_features = [[] for _ in layer_indices]  # one list per layer
            all_labels = []

            print(f"  Pre-computing hidden states ({desc})...")
            for batch in tqdm(dataloader, desc=desc):
                feats = extract_features_from_batch(batch)  # List[Tensor(B, D)]
                for i, f in enumerate(feats):
                    all_features[i].append(f.cpu())
                all_labels.append(batch['label'])

            # Concatenate into single tensors: shape (N, hidden_size)
            all_features = [torch.cat(f, dim=0) for f in all_features]
            all_labels = torch.cat(all_labels, dim=0)
            return all_features, all_labels

        print("\nBuilding hidden state cache...")
        train_feats, train_labels = _build_cache(train_loader, "train")
        val_feats,   val_labels   = _build_cache(val_loader,   "val")
        print("  Cache built.")

        # Wrap as TensorDataset so the trainer can iterate normally
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = features  # List[Tensor(N, D)]
                self.labels = labels      # Tensor(N,)

            def __len__(self):
                return self.labels.shape[0]

            def __getitem__(self, idx):
                return {
                    'features': [f[idx] for f in self.features],
                    'label': self.labels[idx]
                }

        train_loader = DataLoader(
            CachedDataset(train_feats, train_labels),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            CachedDataset(val_feats, val_labels),
            batch_size=args.batch_size, shuffle=False
        )

        # With cached data the features are already tensors — just return them
        def extract_features(batch):
            return batch['features']

    else:
        # Without cache: run the LLM on every batch every epoch
        extract_features = extract_features_from_batch

    # Training
    print(f"\nTraining for {args.num_epochs} epochs...")
    history = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        extract_features_fn=extract_features,
        num_epochs=args.num_epochs,
        verbose=True,
        save_best=True,
        checkpoint_dir=output_dir
    )
    
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        history,
        save_path=output_dir / "training_history.png",
        title=f"Training History (Multi-Layer {args.aggregation})"
    )
    
    # Display layer weights (only for weighted aggregation)
    if args.aggregation == "weighted":
        layer_weights = probe.get_layer_weights()
        if layer_weights is not None:
            print("\nLearned layer weights:")
            for i, (layer_idx, weight) in enumerate(zip(layer_indices, layer_weights)):
                print(f"  Layer {layer_idx}: {weight:.4f}")
            
            # Save weights
            weights_dict = {str(layer_idx): float(weight) for layer_idx, weight in zip(layer_indices, layer_weights)}
            with open(output_dir / "layer_weights.json", "w") as f:
                json.dump(weights_dict, f, indent=2)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = ProbeEvaluator(probe, device=loader.device, threshold=0.5)
    metrics = evaluator.evaluate(test_loader, extract_features, verbose=True)
    
    # Print evaluation results
    evaluator.print_metrics(metrics)
    
    # Save evaluation results
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Collect predictions for visualization
    print("\nGenerating visualizations...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = extract_features(batch)
            predictions = probe(features)
            all_predictions.append(predictions.cpu())
            all_labels.append(batch['label'])
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy().flatten()
    all_labels = torch.cat(all_labels, dim=0).numpy().flatten()
    
    # Visualize
    plot_confidence_distribution(
        all_predictions,
        all_labels,
        save_path=output_dir / "confidence_distribution.png",
        title=f"Confidence Distribution (Multi-Layer {args.aggregation})"
    )
    
    binary_predictions = (all_predictions >= 0.5).astype(int)
    binary_labels = (all_labels >= 0.5).astype(int)
    
    plot_confusion_matrix(
        binary_labels,
        binary_predictions,
        save_path=output_dir / "confusion_matrix.png",
        title=f"Confusion Matrix (Multi-Layer {args.aggregation})"
    )
    
    plot_roc_curve(
        binary_labels,
        all_predictions,
        save_path=output_dir / "roc_curve.png",
        title=f"ROC Curve (Multi-Layer {args.aggregation})"
    )
    
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    # Cleanup
    loader.unload()


if __name__ == "__main__":
    main()
