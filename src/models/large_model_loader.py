"""
Large model loader - supports TinyLlama, LLaMA-2, and other large models
with optional quantization for memory-constrained environments.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LargeModelLoader:
    """
    Loads large causal language models with optional quantization.

    Supports 4-bit and 8-bit quantization via bitsandbytes to allow
    running large models on M1 Macs and other memory-constrained devices.
    """

    SUPPORTED_MODELS = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "phi-2": "microsoft/phi-2",
    }

    def __init__(
        self,
        model_name: str,
        device: str = "mps",
        use_quantization: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[dict] = None
    ):
        """
        Args:
            model_name: Model name (key in SUPPORTED_MODELS or Hugging Face model ID)
            device: Target device ("mps", "cuda", "cpu")
            use_quantization: Enable quantization
            load_in_8bit: Load with 8-bit quantization
            load_in_4bit: Load with 4-bit quantization
            max_memory: Per-device memory limits for device_map="auto"
        """
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization or load_in_8bit or load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory

        # Resolve shorthand names to full Hugging Face model IDs
        self.model_id = self.SUPPORTED_MODELS.get(model_name, model_name)

        self.model = None
        self.tokenizer = None

        logger.info(f"LargeModelLoader initialized")
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Device: {device}")
        logger.info(f"Quantization: {self.use_quantization}")

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
        """
        logger.info(f"Loading model: {self.model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }

        # Configure quantization if requested
        if self.use_quantization:
            if self.load_in_4bit:
                logger.info("Loading with 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            elif self.load_in_8bit:
                logger.info("Loading with 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
        else:
            if self.max_memory:
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = self.max_memory

        # Load model with fallback to float32 on error
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Move to device manually when not using device_map
            if not self.use_quantization and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info("Model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Retrying with float32 precision...")

            model_kwargs["torch_dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            if not self.use_quantization:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Model loaded with fallback settings")

        return self.model, self.tokenizer

    def unload(self):
        """Release the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("Model unloaded")

    def get_model_info(self) -> dict:
        """
        Return metadata about the loaded model.

        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "not loaded"}

        return {
            "model_id": self.model_id,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "quantized": self.use_quantization,
        }


def main():
    """Smoke test for LargeModelLoader (uses TinyLlama as the lightest option)."""
    print("=== Test: Large Model Loader ===")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\nUsing device: {device}")

    print("\n1. Loading TinyLlama...")
    loader = LargeModelLoader("tinyllama", device=device)
    model, tokenizer = loader.load()

    info = loader.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n2. Test inference...")
    text = "The height of Mount Fuji is"
    inputs = tokenizer(text, return_tensors="pt")

    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Number of hidden states: {len(outputs.hidden_states)}")
    print(f"Hidden state shape: {outputs.hidden_states[0].shape}")

    loader.unload()
    print("\nTest completed!")


if __name__ == "__main__":
    main()
