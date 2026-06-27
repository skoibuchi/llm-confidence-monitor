"""
Confidence scorer - wraps a trained probe and interprets its output.
"""

import torch
from typing import List, Dict, Optional, Union
import logging

from src.probes.linear_probe import LinearProbe, MultiLayerProbe

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Wraps a trained LinearProbe or MultiLayerProbe and provides
    helper methods to compute and interpret confidence scores.
    """

    def __init__(
        self,
        probes: Union[Dict[int, LinearProbe], MultiLayerProbe],
        threshold: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            probes: Trained probe(s).
                - Dict[int, LinearProbe]: one probe per layer index
                - MultiLayerProbe: single probe that integrates multiple layers
            threshold: Decision boundary for binary classification
            device: Target device
        """
        self.probes = probes
        self.threshold = threshold
        self.device = device or torch.device("cpu")

        # Move probe(s) to device and set eval mode
        if isinstance(probes, dict):
            for probe in probes.values():
                probe.to(self.device)
                probe.eval()
        else:
            probes.to(self.device)
            probes.eval()

        logger.info(f"ConfidenceScorer initialized with threshold={threshold}")

    def score_single_layer(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute the confidence score for a single layer.

        Args:
            hidden_state: Hidden state tensor of shape (batch_size, hidden_dim)
            layer_idx: Layer index to use

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size, 1)
        """
        if not isinstance(self.probes, dict):
            raise ValueError("probes must be a dict for single layer scoring")

        if layer_idx not in self.probes:
            raise ValueError(f"No probe for layer {layer_idx}")

        probe = self.probes[layer_idx]
        hidden_state = hidden_state.to(self.device)

        with torch.no_grad():
            confidence = probe(hidden_state)

        return confidence

    def score_all_layers(
        self,
        hidden_states: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute confidence scores for every layer that has a probe.

        Args:
            hidden_states: List of hidden state tensors, one per layer,
                each of shape (batch_size, hidden_dim)

        Returns:
            List[torch.Tensor]: Confidence scores per layer
        """
        if not isinstance(self.probes, dict):
            raise ValueError("probes must be a dict for all layers scoring")

        scores = []
        for layer_idx, hidden_state in enumerate(hidden_states):
            if layer_idx in self.probes:
                score = self.score_single_layer(hidden_state, layer_idx)
                scores.append(score)

        return scores

    def score_multi_layer(
        self,
        hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute a single confidence score by integrating multiple layers.

        Args:
            hidden_states: List of hidden state tensors, one per layer,
                each of shape (batch_size, hidden_dim)

        Returns:
            torch.Tensor: Confidence score of shape (batch_size, 1)
        """
        if not isinstance(self.probes, MultiLayerProbe):
            raise ValueError("probes must be MultiLayerProbe for multi-layer scoring")

        hidden_states = [h.to(self.device) for h in hidden_states]

        with torch.no_grad():
            confidence = self.probes(hidden_states)

        return confidence

    def classify(
        self,
        confidence_score: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply a threshold to produce binary predictions.

        Args:
            confidence_score: Confidence scores of shape (batch_size, 1)
            threshold: Decision boundary (defaults to self.threshold)

        Returns:
            torch.Tensor: Binary predictions (0 = does not know, 1 = knows)
        """
        threshold = threshold if threshold is not None else self.threshold
        return (confidence_score >= threshold).float()

    def get_confidence_level(
        self,
        confidence_score: float
    ) -> str:
        """
        Map a scalar confidence score to a human-readable level string.

        Args:
            confidence_score: Scalar in [0, 1]

        Returns:
            str: Confidence level description
        """
        if confidence_score >= 0.9:
            return "Highly confident (knows)"
        elif confidence_score >= 0.7:
            return "Likely knows"
        elif confidence_score >= 0.5:
            return "Uncertain"
        elif confidence_score >= 0.3:
            return "Likely does not know"
        else:
            return "Does not know"

    def score_with_interpretation(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        layer_idx: Optional[int] = None
    ) -> Dict[str, Union[float, str, int]]:
        """
        Compute a confidence score and return it with human-readable metadata.

        Args:
            hidden_states: Hidden state tensor or list of tensors
            layer_idx: Layer index (required when using a single-layer dict probe)

        Returns:
            Dict with keys:
                - confidence_score (float)
                - binary_prediction (int: 0 or 1)
                - confidence_level (str)
                - knows (bool)
        """
        if isinstance(hidden_states, list):
            if isinstance(self.probes, MultiLayerProbe):
                confidence = self.score_multi_layer(hidden_states)
            else:
                confidence = self.score_single_layer(hidden_states[-1], len(hidden_states) - 1)
        else:
            if layer_idx is None:
                raise ValueError("layer_idx must be specified for single layer scoring")
            confidence = self.score_single_layer(hidden_states, layer_idx)

        confidence_value = confidence.item() if confidence.numel() == 1 else confidence.mean().item()

        binary = self.classify(confidence)
        binary_value = int(binary.item() if binary.numel() == 1 else binary.mode()[0].item())

        level = self.get_confidence_level(confidence_value)

        return {
            "confidence_score": confidence_value,
            "binary_prediction": binary_value,
            "confidence_level": level,
            "knows": binary_value == 1
        }


def main():
    """Smoke test for ConfidenceScorer."""
    print("=== Test 1: Single Layer Scoring ===")

    input_dim = 768
    num_layers = 13  # GPT-2 Small

    probes = {i: LinearProbe(input_dim) for i in range(num_layers)}
    scorer = ConfidenceScorer(probes, threshold=0.5)

    batch_size = 1
    hidden_state = torch.randn(batch_size, input_dim)

    score = scorer.score_single_layer(hidden_state, layer_idx=12)
    print(f"Layer 12 confidence: {score.item():.3f}")

    result = scorer.score_with_interpretation(hidden_state, layer_idx=12)
    print(f"Confidence score:  {result['confidence_score']:.3f}")
    print(f"Binary prediction: {result['binary_prediction']}")
    print(f"Confidence level:  {result['confidence_level']}")
    print(f"Knows:             {result['knows']}")

    print("\n=== Test 2: All Layers Scoring ===")

    hidden_states = [torch.randn(batch_size, input_dim) for _ in range(num_layers)]

    scores = scorer.score_all_layers(hidden_states)
    print(f"Number of layers: {len(scores)}")
    for i, score in enumerate(scores):
        print(f"  Layer {i}: {score.item():.3f}")

    print("\n=== Test 3: Multi-Layer Scoring ===")

    multi_probe = MultiLayerProbe(input_dim, num_layers=3, aggregation="concat")
    multi_scorer = ConfidenceScorer(multi_probe, threshold=0.5)

    hidden_states_3 = [torch.randn(batch_size, input_dim) for _ in range(3)]

    score = multi_scorer.score_multi_layer(hidden_states_3)
    print(f"Multi-layer confidence: {score.item():.3f}")

    result = multi_scorer.score_with_interpretation(hidden_states_3)
    print(f"Confidence level: {result['confidence_level']}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
