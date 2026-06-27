"""
Linear probe - predicts a confidence score from hidden states.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """
    Single-layer linear probe: maps a hidden state vector to a
    confidence score in [0, 1] via a Linear layer followed by Sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        bias: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Dimension of the input hidden state
            output_dim: Output dimension (1 for a scalar confidence score)
            bias: Whether to include a bias term
            dropout: Dropout rate (0.0 = disabled)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

        logger.info(f"LinearProbe initialized: input_dim={input_dim}, output_dim={output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size, output_dim), range [0, 1]
        """
        if self.dropout is not None:
            x = self.dropout(x)

        logits = self.linear(x)
        confidence = self.sigmoid(logits)

        return confidence

    def predict_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence scores in eval mode (no gradient).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size, output_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_binary(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict binary labels by applying a threshold to the confidence score.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            threshold: Decision boundary (default: 0.5)

        Returns:
            torch.Tensor: Binary predictions (0 = does not know, 1 = knows)
        """
        confidence = self.predict_confidence(x)
        return (confidence >= threshold).float()

    def get_weights(self) -> torch.Tensor:
        """
        Return the linear layer's weight matrix.

        Returns:
            torch.Tensor: Weight matrix of shape (output_dim, input_dim)
        """
        return self.linear.weight.data

    def get_bias(self) -> Optional[torch.Tensor]:
        """
        Return the linear layer's bias vector.

        Returns:
            torch.Tensor or None: Bias vector of shape (output_dim,)
        """
        if self.linear.bias is not None:
            return self.linear.bias.data
        return None


class MultiLayerProbe(nn.Module):
    """
    Probe that integrates hidden states from multiple layers before
    predicting a confidence score.

    Supports three aggregation strategies:
    - concat:    concatenate all layer vectors, then apply a single linear layer
    - mean:      average the layer vectors, then apply a linear layer
    - weighted:  learnable weighted sum, then apply a linear layer
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        output_dim: int = 1,
        aggregation: str = "concat",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Hidden state dimension of each layer
            num_layers: Number of layers to integrate
            output_dim: Output dimension (1 for a scalar confidence score)
            aggregation: Aggregation method ("concat", "mean", "weighted")
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.aggregation = aggregation

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if aggregation == "concat":
            self.linear = nn.Linear(input_dim * num_layers, output_dim)
        elif aggregation == "mean":
            self.linear = nn.Linear(input_dim, output_dim)
        elif aggregation == "weighted":
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        self.sigmoid = nn.Sigmoid()

        logger.info(f"MultiLayerProbe initialized: num_layers={num_layers}, aggregation={aggregation}")

    def forward(self, layer_outputs: list) -> torch.Tensor:
        """
        Args:
            layer_outputs: List of hidden state tensors, one per layer,
                each of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size, output_dim)
        """
        if len(layer_outputs) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {len(layer_outputs)}"
            )

        if self.aggregation == "concat":
            x = torch.cat(layer_outputs, dim=-1)
        elif self.aggregation == "mean":
            x = torch.stack(layer_outputs, dim=0).mean(dim=0)
        elif self.aggregation == "weighted":
            weights = torch.softmax(self.layer_weights, dim=0)
            x = sum(w * h for w, h in zip(weights, layer_outputs))

        if self.dropout is not None:
            x = self.dropout(x)

        logits = self.linear(x)
        confidence = self.sigmoid(logits)

        return confidence

    def predict_confidence(self, layer_outputs: list) -> torch.Tensor:
        """Predict confidence scores in eval mode (no gradient)."""
        self.eval()
        with torch.no_grad():
            return self.forward(layer_outputs)

    def get_layer_weights(self) -> Optional[torch.Tensor]:
        """
        Return the learned layer importance weights (weighted aggregation only).

        Returns:
            torch.Tensor or None: Normalized weights of shape (num_layers,)
        """
        if self.aggregation == "weighted":
            return torch.softmax(self.layer_weights, dim=0).data
        return None


def main():
    """Smoke test for LinearProbe and MultiLayerProbe."""
    print("=== Test 1: LinearProbe ===")

    input_dim = 768  # GPT-2 hidden size
    probe = LinearProbe(input_dim)

    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    confidence = probe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {confidence.shape}")
    print(f"Confidence scores: {confidence.squeeze().tolist()}")

    binary = probe.predict_binary(x, threshold=0.5)
    print(f"Binary predictions: {binary.squeeze().tolist()}")

    print("\n=== Test 2: MultiLayerProbe (concat) ===")

    num_layers = 3
    multi_probe = MultiLayerProbe(input_dim, num_layers, aggregation="concat")

    layer_outputs = [torch.randn(batch_size, input_dim) for _ in range(num_layers)]

    confidence = multi_probe(layer_outputs)
    print(f"Number of layers: {num_layers}")
    print(f"Output shape: {confidence.shape}")
    print(f"Confidence scores: {confidence.squeeze().tolist()}")

    print("\n=== Test 3: MultiLayerProbe (weighted) ===")

    multi_probe_weighted = MultiLayerProbe(
        input_dim, num_layers, aggregation="weighted"
    )

    confidence = multi_probe_weighted(layer_outputs)
    print(f"Confidence scores: {confidence.squeeze().tolist()}")

    layer_weights = multi_probe_weighted.get_layer_weights()
    print(f"Layer weights: {layer_weights.tolist()}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
