"""
Probe trainer - manages the training loop for linear probes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Optional, Callable
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProbeTrainer:
    """Manages training, validation, and checkpointing for a probe."""

    def __init__(
        self,
        probe: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        scheduler: Optional[object] = None
    ):
        """
        Args:
            probe: The probe module to train
            optimizer: Optimizer instance
            criterion: Loss function
            device: Target device
            scheduler: Optional learning rate scheduler
        """
        self.probe = probe
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device("cpu")
        self.scheduler = scheduler

        self.probe.to(self.device)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        logger.info(f"ProbeTrainer initialized on device: {self.device}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        extract_features_fn: Callable,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            dataloader: DataLoader for the training set
            extract_features_fn: Function that takes a batch dict and
                returns features (Tensor or List[Tensor])
            verbose: Show tqdm progress bar

        Returns:
            Dict with key "loss" containing the mean epoch loss
        """
        self.probe.train()

        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

        for batch in iterator:
            features = extract_features_fn(batch)

            if isinstance(features, list):
                features = [f.to(self.device) for f in features]
            else:
                features = features.to(self.device)

            labels = batch['label'].to(self.device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            predictions = self.probe(features)
            loss = self.criterion(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if verbose:
                iterator.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'loss': avg_loss}

    def validate(
        self,
        dataloader: DataLoader,
        extract_features_fn: Callable,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run one validation pass.

        Args:
            dataloader: DataLoader for the validation set
            extract_features_fn: Feature extraction function
            verbose: Show tqdm progress bar

        Returns:
            Dict with key "loss" containing the mean validation loss
        """
        self.probe.eval()

        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(dataloader, desc="Validation") if verbose else dataloader

        with torch.no_grad():
            for batch in iterator:
                features = extract_features_fn(batch)

                if isinstance(features, list):
                    features = [f.to(self.device) for f in features]
                else:
                    features = features.to(self.device)

                labels = batch['label'].to(self.device)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)

                predictions = self.probe(features)
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                num_batches += 1

                if verbose:
                    iterator.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'loss': avg_loss}

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        extract_features_fn: Callable,
        num_epochs: int,
        verbose: bool = True,
        save_best: bool = True,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict[str, list]:
        """
        Train for multiple epochs with optional best-model checkpointing.

        Args:
            train_dataloader: DataLoader for the training set
            val_dataloader: DataLoader for the validation set (optional)
            extract_features_fn: Feature extraction function
            num_epochs: Number of epochs to train
            verbose: Show per-epoch progress
            save_best: Save a checkpoint whenever validation loss improves
            checkpoint_dir: Directory to write checkpoints to

        Returns:
            Dict: Training history with "train_loss" and "val_loss" lists
        """
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_results = self.train_epoch(
                train_dataloader,
                extract_features_fn,
                verbose=verbose
            )
            self.history['train_loss'].append(train_results['loss'])

            if verbose:
                print(f"Train Loss: {train_results['loss']:.4f}")

            if val_dataloader is not None:
                val_results = self.validate(
                    val_dataloader,
                    extract_features_fn,
                    verbose=verbose
                )
                self.history['val_loss'].append(val_results['loss'])

                if verbose:
                    print(f"Val Loss: {val_results['loss']:.4f}")

                if save_best and val_results['loss'] < best_val_loss:
                    best_val_loss = val_results['loss']
                    if checkpoint_dir is not None:
                        self.save_checkpoint(
                            checkpoint_dir / "best_model.pt",
                            epoch=epoch,
                            val_loss=best_val_loss
                        )
                        if verbose:
                            print(f"Saved best model (val_loss: {best_val_loss:.4f})")

            if self.scheduler is not None:
                if val_dataloader is not None:
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: Optional[int] = None,
        **kwargs
    ):
        """
        Save a checkpoint to disk.

        Args:
            path: File path for the checkpoint
            epoch: Current epoch number
            **kwargs: Additional metadata to store in the checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'probe_state_dict': self.probe.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """
        Load a checkpoint from disk.

        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.probe.load_state_dict(checkpoint['probe_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {path}")

        return checkpoint


def main():
    """Smoke test for ProbeTrainer."""
    from src.probes.linear_probe import LinearProbe

    print("=== Test: ProbeTrainer ===")

    input_dim = 768
    probe = LinearProbe(input_dim)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainer = ProbeTrainer(probe, optimizer, criterion)

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'features': torch.randn(input_dim),
                'label': torch.rand(1)
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    def extract_features(batch):
        return batch['features']

    print("\nTraining for 3 epochs...")
    history = trainer.train(
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        extract_features_fn=extract_features,
        num_epochs=3,
        verbose=True
    )

    print("\nTraining history:")
    print(f"Train losses: {[f'{loss:.4f}' for loss in history['train_loss']]}")
    print(f"Val losses:   {[f'{loss:.4f}' for loss in history['val_loss']]}")

    checkpoint_path = Path("results/experiments/test/test_checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path, epoch=3)
    print(f"\nCheckpoint saved to {checkpoint_path}")

    trainer.load_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
