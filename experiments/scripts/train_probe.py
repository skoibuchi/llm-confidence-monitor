"""
Probe training script - end-to-end experiment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
import json
from datetime import datetime

from src.models.model_loader import ModelLoader
from src.models.hidden_extractor import HiddenStateExtractor
from src.data.dataset import KnowledgeProbeDataset, create_sample_dataset, split_dataset
from src.probes.linear_probe import LinearProbe
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
    parser = argparse.ArgumentParser(description="Train knowledge probe")

    # Model settings
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model name (default: gpt2)")
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="Layer index to probe (-1 for last layer)")
    
    # Data settings
    parser.add_argument("--data_path", type=str, default="data/raw/sample_dataset.jsonl",
                        help="Path to dataset")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create sample dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for sample dataset")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--pooling", type=str, default="last",
                        choices=["last", "mean", "max", "cls"],
                        help="Pooling method")
    
    # Miscellaneous
    parser.add_argument("--output_dir", type=str, default="results/experiments",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (used as subdirectory prefix, e.g. 'gpt2_layer6'). "
                             "Defaults to a timestamp only.")
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

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.experiment_name}_{timestamp}" if args.experiment_name else timestamp
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Knowledge Probe Training")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_dir / 'config.json'}")
    
    # Create sample dataset if requested
    if args.create_sample:
        print(f"\nCreating sample dataset with {args.num_samples} samples...")
        create_sample_dataset(args.data_path, args.num_samples)

        # Split dataset
        print("Splitting dataset...")
        split_dataset(args.data_path, output_dir=Path(args.data_path).parent.parent / "processed")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    loader = ModelLoader(args.model_name)
    model, tokenizer = loader.load()
    
    model_info = loader.get_model_info()
    print(f"Model info: {model_info}")
    
    # Create hidden state extractor
    extractor = HiddenStateExtractor(model, tokenizer)
    
    # Load datasets
    print(f"\nLoading datasets...")
    data_dir = Path(args.data_path).parent.parent / "processed"
    
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
    
    # Create probe
    print(f"\nCreating probe for layer {args.layer_idx}...")
    hidden_size = model_info['hidden_size']
    probe = LinearProbe(hidden_size)
    
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
    
    # Define feature extraction function
    def extract_features(batch):
        """Extract features from a batch."""
        input_ids = batch['input_ids'].to(loader.device)
        attention_mask = batch['attention_mask'].to(loader.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Retrieve specified layer
            layer_idx = args.layer_idx if args.layer_idx >= 0 else len(hidden_states) - 1
            hidden = hidden_states[layer_idx]

            # Pooling
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
            features = features.to(torch.float32)

        return features
    
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
        save_path=output_dir / "training_history.png"
    )
    
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
        save_path=output_dir / "confidence_distribution.png"
    )
    
    binary_predictions = (all_predictions >= 0.5).astype(int)
    binary_labels = (all_labels >= 0.5).astype(int)
    
    plot_confusion_matrix(
        binary_labels,
        binary_predictions,
        save_path=output_dir / "confusion_matrix.png"
    )
    
    plot_roc_curve(
        binary_labels,
        all_predictions,
        save_path=output_dir / "roc_curve.png"
    )
    
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    # Cleanup
    loader.unload()


if __name__ == "__main__":
    main()
