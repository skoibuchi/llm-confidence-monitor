"""
Real-time inference engine - generates tokens and computes a confidence
score for each step using a trained probe.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Optional, Generator, Tuple
import logging

logger = logging.getLogger(__name__)


class RealtimeGenerator:
    """
    Generates text token by token and computes a probe confidence score
    at each step.  Yields (token, confidence, metadata) tuples so that
    callers can stream results incrementally.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        probe,
        layers: List[int],
        device: str = "mps"
    ):
        """
        Args:
            model: Causal language model
            tokenizer: Corresponding tokenizer
            probe: Trained probe (LinearProbe or MultiLayerProbe)
            layers: Layer indices to feed into the probe
            device: Target device string
        """
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.layers = layers
        self.device = device

        self.model.eval()
        self.probe.eval()

        logger.info(f"RealtimeGenerator initialized")
        logger.info(f"Layers: {layers}")
        logger.info(f"Device: {device}")

    def generate_with_confidence(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        do_sample: bool = True,
    ) -> Generator[Tuple[str, float, Dict], None, None]:
        """
        Generate tokens one at a time and yield confidence scores.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (only effective when do_sample=True)
            top_p: Top-p (nucleus) sampling threshold; 1.0 = disabled
            top_k: Top-k filtering; 0 = disabled
            repetition_penalty: Penalty applied to logits of already-generated tokens; 1.0 = disabled
            no_repeat_ngram_size: Forbid repeating n-grams of this length; 0 = disabled
            min_new_tokens: Suppress EOS until this many tokens have been generated
            do_sample: False = greedy decoding

        Yields:
            Tuple[str, float, Dict]: (token_text, confidence, metadata)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids: List[int] = []  # token IDs generated so far (used by various filters)

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Extract next-token logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, tuple):
                logits = outputs[0][:, -1, :]
            else:
                last_hidden = outputs.last_hidden_state[:, -1, :]
                logits = self.model.lm_head(last_hidden) if hasattr(self.model, 'lm_head') else last_hidden

            # ── repetition penalty ──────────────────────────────────
            if repetition_penalty != 1.0 and generated_ids:
                for token_id in set(generated_ids):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # ── no_repeat_ngram ─────────────────────────────────────
            if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size - 1:
                ngram_prefix = tuple(generated_ids[-(no_repeat_ngram_size - 1):])
                banned: List[int] = []
                for i in range(len(generated_ids) - (no_repeat_ngram_size - 1)):
                    if tuple(generated_ids[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                        banned.append(generated_ids[i + no_repeat_ngram_size - 1])
                for token_id in banned:
                    logits[0, token_id] = float('-inf')

            # ── min_new_tokens: suppress EOS ────────────────────────
            if step < min_new_tokens and self.tokenizer.eos_token_id is not None:
                logits[0, self.tokenizer.eos_token_id] = float('-inf')

            if do_sample:
                # ── temperature ─────────────────────────────────────
                logits = logits / temperature

                # ── top-k ───────────────────────────────────────────
                if top_k > 0:
                    topk_values = torch.topk(logits, top_k).values[:, -1, None]
                    logits = logits.masked_fill(logits < topk_values, float('-inf'))

                # ── top-p (nucleus) ─────────────────────────────────
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_prob = probs[0, next_token.item()].item()
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                # For greedy decoding, compute softmax over raw logits for the probability
                token_prob = torch.softmax(logits, dim=-1)[0, next_token.item()].item()

            # Decode the new token
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)

            if not token_text:
                # Fallback: convert token ID to string directly
                token_text = self.tokenizer.convert_ids_to_tokens(next_token[0].item())
                # GPT-2 uses 'Ġ' as a space prefix
                if isinstance(token_text, str) and token_text.startswith('Ġ'):
                    token_text = ' ' + token_text[1:]

            # Compute confidence from the probe
            confidence, layer_confidences = self._calculate_confidence(outputs.hidden_states)

            metadata = {
                'step': step,
                'token_id': next_token.item(),
                'token_prob': token_prob,
                'layer_confidences': layer_confidences,
                'logit_max': logits.max().item(),
                'logit_entropy': self._calculate_entropy(logits)
            }

            yield token_text, confidence, metadata

            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

    def _calculate_confidence(
        self,
        hidden_states: Tuple[torch.Tensor, ...]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Feed the last-token hidden states into the probe.

        Args:
            hidden_states: All layer hidden states from the model output

        Returns:
            Tuple[float, Dict]: (overall confidence, per-layer confidence dict)
        """
        selected_hidden_states = []
        layer_confidences = {}

        for layer_idx in self.layers:
            hidden = hidden_states[layer_idx][:, -1, :].to(torch.float32)  # last token, cast to float32
            selected_hidden_states.append(hidden)

        with torch.no_grad():
            if hasattr(self.probe, 'forward'):
                # MultiLayerProbe
                confidence = self.probe(selected_hidden_states).item()
            else:
                # LinearProbe (single layer)
                confidence = self.probe(selected_hidden_states[0]).item()

        for layer_idx in self.layers:
            layer_confidences[layer_idx] = confidence  # simplified

        return confidence, layer_confidences

    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute the entropy of the next-token distribution.

        Args:
            logits: Next-token logits

        Returns:
            float: Shannon entropy
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        return entropy

    def generate_full_response(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **kwargs
    ) -> Tuple[str, List[Dict]]:
        """
        Generate a full response and return per-token confidence info.

        Handles multi-byte tokenization (e.g. Japanese) by decoding the full
        token sequence each step and taking the diff from the previous step.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments forwarded to generate_with_confidence

        Returns:
            Tuple[str, List[Dict]]: (generated_text, list of per-token dicts)
        """
        tokens_info = []
        all_token_ids = []
        previous_text = prompt

        for token, confidence, metadata in self.generate_with_confidence(
            prompt, max_new_tokens, **kwargs
        ):
            all_token_ids.append(metadata['token_id'])

            # Decode incrementally to handle multi-byte tokens correctly
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = prompt_ids + all_token_ids
            current_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

            if current_text.startswith(previous_text):
                token_text = current_text[len(previous_text):]
            else:
                token_text = token  # fallback

            previous_text = current_text

            tokens_info.append({
                'token': token_text if token_text else token,
                'confidence': confidence,
                **metadata
            })

        generated_text = previous_text[len(prompt):] if previous_text.startswith(prompt) else previous_text
        return generated_text, tokens_info


def main():
    """Smoke test for RealtimeGenerator."""
    from src.models.model_loader import ModelLoader
    from src.probes.linear_probe import MultiLayerProbe

    print("=== Test: Realtime Generator ===")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\nUsing device: {device}")

    print("\nLoading model...")
    loader = ModelLoader("gpt2", device=device)
    model, tokenizer = loader.load()

    print("Creating probe (untrained)...")
    probe = MultiLayerProbe(
        input_dim=768,
        num_layers=3,
        aggregation="weighted"
    ).to(device)

    print("Creating generator...")
    generator = RealtimeGenerator(
        model=model,
        tokenizer=tokenizer,
        probe=probe,
        layers=[0, 6, 11],
        device=device
    )

    prompt = "The height of Mount Fuji is"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating with confidence scores:")
    print("-" * 60)

    for token, confidence, metadata in generator.generate_with_confidence(
        prompt,
        max_new_tokens=20,
        temperature=0.7
    ):
        if confidence > 0.7:
            color = "\033[92m"  # green
        elif confidence > 0.4:
            color = "\033[93m"  # yellow
        else:
            color = "\033[91m"  # red
        reset = "\033[0m"

        print(f"{color}Token: {token:15s} Confidence: {confidence:.3f}{reset}")

    print("-" * 60)

    print("\n\nGenerating full response...")
    full_text, tokens_info = generator.generate_full_response(
        prompt,
        max_new_tokens=20
    )

    print(f"\nGenerated text: {full_text}")
    print(f"\nAverage confidence: {sum(t['confidence'] for t in tokens_info) / len(tokens_info):.3f}")

    print("\nTest completed!")


if __name__ == "__main__":
    main()
