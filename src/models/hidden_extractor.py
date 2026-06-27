"""
Hidden state extractor - retrieves intermediate layer outputs from an LLM.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """
    Extracts hidden states from intermediate layers of an LLM.

    Uses the unified Hugging Face Transformers interface and works
    with most causal language models without model-specific code.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Hugging Face pretrained model
            tokenizer: Corresponding tokenizer
            device: Target device (defaults to the model's current device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        # Cache model dimensions
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size

        logger.info(f"HiddenStateExtractor initialized")
        logger.info(f"Number of layers: {self.num_layers}")
        logger.info(f"Hidden size: {self.hidden_size}")

    def extract_hidden_states(
        self,
        text: Union[str, List[str]],
        layers: Optional[List[int]] = None,
        return_attention_mask: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        Extract hidden states from the specified layers.

        Args:
            text: Input text or list of texts
            layers: Layer indices to extract (None = all layers)
            return_attention_mask: Also return the attention mask

        Returns:
            List[torch.Tensor]: Hidden states per layer,
                each of shape (batch_size, seq_len, hidden_size).
            Or Tuple[List[torch.Tensor], torch.Tensor] when return_attention_mask=True.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Forward pass without gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (num_layers + 1) tensors,
        # each of shape (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states

        # Filter to requested layers
        if layers is not None:
            hidden_states = [hidden_states[i] for i in layers]
        else:
            hidden_states = list(hidden_states)

        if return_attention_mask:
            return hidden_states, inputs.attention_mask
        else:
            return hidden_states

    def extract_and_pool(
        self,
        text: Union[str, List[str]],
        layers: Optional[List[int]] = None,
        pooling: str = "last"
    ) -> List[torch.Tensor]:
        """
        Extract hidden states and apply pooling to reduce the sequence dimension.

        Args:
            text: Input text or list of texts
            layers: Layer indices to extract
            pooling: Pooling strategy
                - "last": hidden state of the last non-padding token
                - "mean": mean over non-padding tokens
                - "max": max over non-padding tokens
                - "cls": first token ([CLS]-style)

        Returns:
            List[torch.Tensor]: Pooled hidden states per layer,
                each of shape (batch_size, hidden_size).
        """
        hidden_states, attention_mask = self.extract_hidden_states(
            text, layers, return_attention_mask=True
        )

        pooled_states = []
        for hidden in hidden_states:
            pooled = self._pool_hidden_state(hidden, attention_mask, pooling)
            # Cast to float32 so downstream probes always receive a consistent dtype
            # regardless of the model's native dtype (e.g. bfloat16 for LLaMA/TinyLlama)
            pooled_states.append(pooled.to(torch.float32))

        return pooled_states

    def _pool_hidden_state(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str
    ) -> torch.Tensor:
        """
        Apply pooling to a single layer's hidden state tensor.

        Args:
            hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            pooling: Pooling strategy

        Returns:
            torch.Tensor: (batch_size, hidden_size)
        """
        if pooling == "last":
            # Use the actual last token, not the padded position
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            batch_size = hidden_state.shape[0]
            pooled = hidden_state[
                torch.arange(batch_size, device=self.device),
                seq_lengths
            ]

        elif pooling == "mean":
            # Mean pooling, excluding padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
            sum_hidden = (hidden_state * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            pooled = sum_hidden / sum_mask

        elif pooling == "max":
            # Max pooling, ignoring padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
            hidden_state = hidden_state.clone()
            hidden_state[mask_expanded == 0] = -1e9
            pooled = hidden_state.max(dim=1)[0]

        elif pooling == "cls":
            # First token representation
            pooled = hidden_state[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        return pooled

    def extract_layer_range(
        self,
        text: Union[str, List[str]],
        start_layer: int,
        end_layer: int,
        pooling: Optional[str] = None
    ) -> List[torch.Tensor]:
        """
        Extract hidden states for a contiguous range of layers.

        Args:
            text: Input text or list of texts
            start_layer: First layer index (0-indexed, inclusive)
            end_layer: Last layer index (0-indexed, inclusive)
            pooling: Pooling strategy (None = no pooling)

        Returns:
            List[torch.Tensor]: Hidden states for the specified layers
        """
        layers = list(range(start_layer, end_layer + 1))

        if pooling is not None:
            return self.extract_and_pool(text, layers, pooling)
        else:
            return self.extract_hidden_states(text, layers)

    def get_layer_info(self) -> dict:
        """
        Return basic information about the model's layer structure.

        Returns:
            dict: Layer information
        """
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "device": str(self.device),
        }


def main():
    """Smoke test for HiddenStateExtractor."""
    from src.models.model_loader import ModelLoader

    loader = ModelLoader("gpt2")
    model, tokenizer = loader.load()

    extractor = HiddenStateExtractor(model, tokenizer)

    text = "Mount Fuji is 3776 meters tall."

    print("\n=== Test 1: Extract all layers ===")
    hidden_states = extractor.extract_hidden_states(text)
    print(f"Number of layers: {len(hidden_states)}")
    print(f"Shape of each layer: {hidden_states[0].shape}")

    print("\n=== Test 2: Extract specific layers ===")
    hidden_states = extractor.extract_hidden_states(text, layers=[0, 6, 12])
    print(f"Number of layers: {len(hidden_states)}")

    print("\n=== Test 3: Extract and pool (last token) ===")
    pooled_states = extractor.extract_and_pool(text, pooling="last")
    print(f"Number of layers: {len(pooled_states)}")
    print(f"Shape of each layer: {pooled_states[0].shape}")

    print("\n=== Test 4: Extract and pool (mean) ===")
    pooled_states = extractor.extract_and_pool(text, pooling="mean")
    print(f"Shape of each layer: {pooled_states[0].shape}")

    print("\n=== Test 5: Batch processing ===")
    texts = ["What is the height of Mount Fuji?", "What is the capital of Japan?"]
    pooled_states = extractor.extract_and_pool(texts, pooling="last")
    print(f"Batch size: {pooled_states[0].shape[0]}")
    print(f"Hidden size: {pooled_states[0].shape[1]}")

    print("\n=== Layer Info ===")
    info = extractor.get_layer_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\nAll tests passed!")

    loader.unload()


if __name__ == "__main__":
    main()
