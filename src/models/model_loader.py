"""
Model loader - manages LLM models and tokenizers.
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads an LLM model and tokenizer from Hugging Face.

    Automatically selects the best available device:
    MPS (Apple Silicon) > CUDA > CPU.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: Hugging Face model name (e.g. "gpt2", "TinyLlama/TinyLlama-1.1B")
            device: Device selection ("auto", "mps", "cuda", "cpu")
            cache_dir: Directory for model cache
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = self._get_device(device)

        logger.info(f"ModelLoader initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.config = None

    def _get_device(self, device: str) -> torch.device:
        """
        Determine the device to use.

        Args:
            device: Device specification

        Returns:
            torch.device: Selected device
        """
        if device == "auto":
            # Prefer MPS (Apple Silicon), then CUDA, then CPU
            if torch.backends.mps.is_available():
                logger.info("MPS (Metal Performance Shaders) is available")
                return torch.device("mps")
            elif torch.cuda.is_available():
                logger.info("CUDA is available")
                return torch.device("cuda")
            else:
                logger.info("Using CPU")
                return torch.device("cpu")
        else:
            return torch.device(device)

    def load(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load the model and tokenizer.

        Returns:
            Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer
        """
        logger.info(f"Loading model: {self.model_name}")

        # Load config
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=self.cache_dir
        )

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

        # Set pad token if missing (required for GPT-2 etc.)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded successfully")
        logger.info(f"Number of layers: {self.config.num_hidden_layers}")
        logger.info(f"Hidden size: {self.config.hidden_size}")

        return self.model, self.tokenizer

    def get_model_info(self) -> dict:
        """
        Return metadata about the loaded model.

        Returns:
            dict: Model information
        """
        if self.config is None:
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

        return {
            "model_name": self.model_name,
            "num_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "device": str(self.device),
            "model_type": self.config.model_type,
        }

    def unload(self):
        """Release the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear device memory cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        logger.info("Model unloaded from memory")


def main():
    """Smoke test for ModelLoader."""
    loader = ModelLoader("gpt2")
    model, tokenizer = loader.load()

    info = loader.get_model_info()
    print("\n=== Model Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")

    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(loader.device)

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\nTest input: {text}")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print("\nModel loaded successfully!")

    loader.unload()


if __name__ == "__main__":
    main()
