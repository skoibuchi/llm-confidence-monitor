"""
Gradio web interface - real-time confidence display.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
import torch
import pandas as pd
from typing import List, Dict
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.linear_probe import MultiLayerProbe
from src.inference.realtime_generator import RealtimeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceDemo:
    """Confidence demo application."""

    def __init__(
        self,
        model_name: str = "gpt2",
        checkpoint_path: str = None,
        layers: List[int] = [0, 6, 11],
        hidden_dim: int = 768
    ):
        """
        Args:
            model_name: Model name
            checkpoint_path: Path to the probe checkpoint
            layers: Layer indices to use
            hidden_dim: Hidden state dimensionality
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.layers = layers
        self.hidden_dim = hidden_dim
        
        # Select compute device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and probe
        self._load_models()
    
    def _load_models(self):
        """Load model and probe."""
        logger.info("Loading models...")

        # Load LLM for text generation
        logger.info(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load probe
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Instantiate probe
            self.probe = MultiLayerProbe(
                input_dim=self.hidden_dim,
                num_layers=len(self.layers),
                aggregation="weighted"
            ).to(self.device)

            # Restore weights
            if 'probe_state_dict' in checkpoint:
                self.probe.load_state_dict(checkpoint['probe_state_dict'])
            else:
                self.probe.load_state_dict(checkpoint)

            self.probe.eval()
            logger.info("Probe loaded successfully")
        else:
            # Untrained dummy probe (for demo without checkpoint)
            self.probe = MultiLayerProbe(
                input_dim=self.hidden_dim,
                num_layers=len(self.layers),
                aggregation="weighted"
            ).to(self.device)
            logger.warning("Using dummy probe (not trained)")
        
        # Wrap model and probe in the streaming generator
        self.generator = RealtimeGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            probe=self.probe,
            layers=self.layers,
            device=self.device
        )
        
        logger.info("Models loaded successfully")
    
    def generate_with_confidence(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Generate text with confidence scores.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens
            temperature: Sampling temperature
            top_p: Top-p threshold

        Returns:
            Tuple: (generated text, token info DataFrame, confidence plot)
        """
        # Generate full response (non-streaming)
        generated_text, tokens_info = self.generator.generate_full_response(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Build DataFrame
        df = pd.DataFrame(tokens_info)
        df = df[['token', 'confidence', 'step']]
        df['confidence'] = df['confidence'].round(3)

        # Build confidence plot
        confidence_plot = self._create_confidence_plot(tokens_info)

        # Build color-coded HTML
        colored_text = self._create_colored_text(tokens_info)
        
        return colored_text, df, confidence_plot
    
    def generate_with_confidence_streaming(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
    ):
        """
        Generate text with confidence scores (streaming).

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            do_sample: False = greedy decoding
            top_k: Top-k filtering; 0 = disabled
            repetition_penalty: Repetition penalty; 1.0 = disabled
            no_repeat_ngram_size: Forbid repeating n-grams of this size; 0 = disabled
            min_new_tokens: Minimum tokens to generate before allowing EOS

        Yields:
            Tuple: (color-coded HTML, token table HTML, confidence plot)
        """
        import html as html_module

        tokens_info = []
        colored_html = (
            "<div style='"
            "font-size:16px; line-height:2; font-family:monospace;"
            "min-height:140px; max-height:260px; overflow-y:auto;"
            "padding:0.75rem 1rem;"
            "background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;"
            "'>"
        )

        # Generate tokens one by one
        for token, confidence, metadata in self.generator.generate_with_confidence(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
        ):
            # Append token info
            tokens_info.append({
                'token': token,
                'confidence': confidence,
                'token_prob': metadata.get('token_prob', 0.0),
                'step': metadata['step']
            })
            
            # HTML escape
            token_escaped = html_module.escape(token)
            if not token_escaped.strip():
                token_escaped = "␣"
            
            # Choose color based on confidence
            if confidence > 0.7:
                color = "#4CAF50"  # green
                bg_color = "#E8F5E9"
            elif confidence > 0.4:
                color = "#FF9800"  # orange
                bg_color = "#FFF3E0"
            else:
                color = "#F44336"  # red
                bg_color = "#FFEBEE"

            # Append to HTML
            colored_html += f'<span style="color: {color}; background-color: {bg_color}; padding: 2px 4px; margin: 2px; border-radius: 3px;" title="Confidence: {confidence:.3f}">{token_escaped}</span>'
            
            # Create table HTML
            table_html = self._create_tokens_table(tokens_info)

            # Create confidence plot
            confidence_plot = self._create_confidence_plot(tokens_info)

            # Yield incrementally
            yield colored_html + "</div>", table_html, confidence_plot
    
    def _create_tokens_table(self, tokens_info: List[Dict]) -> str:
        """Create a fixed-height scrollable HTML table for token details."""
        import html as html_module
        rows = ""
        for info in tokens_info:
            token_escaped = html_module.escape(info['token']).replace(" ", "&nbsp;")
            c = info['confidence']
            p = info.get('token_prob', 0.0)
            if c > 0.7:
                badge_color = "#4CAF50"
            elif c > 0.4:
                badge_color = "#FF9800"
            else:
                badge_color = "#F44336"
            rows += (
                f"<tr>"
                f"<td style='padding:4px 8px; font-family:monospace; border-bottom:1px solid #f0f0f0;'>{token_escaped}</td>"
                f"<td style='padding:4px 8px; border-bottom:1px solid #f0f0f0;'>"
                f"<span style='display:inline-block; width:8px; height:8px; border-radius:50%; background:{badge_color}; margin-right:5px; vertical-align:middle;'></span>"
                f"{c:.3f}</td>"
                f"<td style='padding:4px 8px; border-bottom:1px solid #f0f0f0; color:#57606a;'>{p:.3f}</td>"
                f"<td style='padding:4px 8px; border-bottom:1px solid #f0f0f0; color:#57606a;'>{info['step']}</td>"
                f"</tr>"
            )
        return (
            "<div style='"
            "height:260px; overflow-y:auto;"
            "border:1px solid #e5e7eb; border-radius:8px;"
            "background:#ffffff;"
            "'>"
            "<table style='width:100%; border-collapse:collapse; font-size:13px;'>"
            "<thead><tr style='position:sticky; top:0; background:#f7f8fa;'>"
            "<th style='padding:6px 8px; text-align:left; border-bottom:2px solid #e5e7eb; font-weight:600;'>Token</th>"
            "<th style='padding:6px 8px; text-align:left; border-bottom:2px solid #e5e7eb; font-weight:600;'>Probe confidence</th>"
            "<th style='padding:6px 8px; text-align:left; border-bottom:2px solid #e5e7eb; font-weight:600;'>Token prob</th>"
            "<th style='padding:6px 8px; text-align:left; border-bottom:2px solid #e5e7eb; font-weight:600;'>Step</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table></div>"
        )

    def _create_colored_text(self, tokens_info: List[Dict]) -> str:
        """Create color-coded HTML text based on confidence scores."""
        import html as html_module

        html = (
            "<div style='"
            "font-size:16px; line-height:2; font-family:monospace;"
            "min-height:140px; max-height:260px; overflow-y:auto;"
            "padding:0.75rem 1rem;"
            "background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;"
            "'>"
        )

        for info in tokens_info:
            token = info['token']
            confidence = info['confidence']

            # HTML-escape special characters for correct display
            token_escaped = html_module.escape(token)

            # Show a placeholder for blank tokens
            if not token_escaped.strip():
                token_escaped = "␣"  # visualise whitespace

            # Choose color based on confidence
            if confidence > 0.7:
                color = "#4CAF50"  # green
                bg_color = "#E8F5E9"
            elif confidence > 0.4:
                color = "#FF9800"  # orange
                bg_color = "#FFF3E0"
            else:
                color = "#F44336"  # red
                bg_color = "#FFEBEE"
            
            html += f'<span style="color: {color}; background-color: {bg_color}; padding: 2px 4px; margin: 2px; border-radius: 3px;" title="Confidence: {confidence:.3f}">{token_escaped}</span>'
        
        html += "</div>"
        return html
    
    def _create_confidence_plot(self, tokens_info: List[Dict]):
        """Create a confidence score plot."""
        import plotly.graph_objects as go
        
        steps = [info['step'] for info in tokens_info]
        confidences = [info['confidence'] for info in tokens_info]
        tokens = [info['token'] for info in tokens_info]
        
        fig = go.Figure()
        
        # Line chart for confidence scores
        fig.add_trace(go.Scatter(
            x=steps,
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            text=tokens,
            hovertemplate='<b>Token:</b> %{text}<br>' +
                         '<b>Step:</b> %{x}<br>' +
                         '<b>Confidence:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Threshold lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Threshold (0.5)")
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                     annotation_text="High Confidence (0.7)")
        
        fig.update_layout(
            title="Confidence Score per Token",
            xaxis_title="Token Step",
            yaxis_title="Confidence Score",
            yaxis_range=[0, 1],
            template='plotly_white',
            height=400
        )
        
        return fig


def get_model_config(model_name: str) -> Dict:
    """
    Return configuration for the given model name.

    Args:
        model_name: Model name

    Returns:
        Dict: Model config (layers, hidden_dim, examples)
    """
    configs = {
        "gpt2": {
            "layers": [0, 6, 11],
            "hidden_dim": 768,
            "num_layers": 12,
            "examples": [
                ["The capital of the United States is", 30, 0.7],
                ["Mount Everest is the highest mountain in", 30, 0.7],
                ["The Earth orbits around the", 20, 0.7],
                ["The capital of the fictional planet Zorg is", 30, 0.7],
            ]
        },
        "rinna/japanese-gpt2-medium": {
            "layers": [0, 12, 23],
            "hidden_dim": 1024,
            "num_layers": 24,
            "examples": [
                ["富士山の高さは", 30, 0.7],
                ["日本の首都は", 20, 0.7],
                ["水の化学式は", 20, 0.7],
                ["架空の惑星Xの重力は", 30, 0.7],
            ]
        },
        "cyberagent/open-calm-small": {
            "layers": [0, 8, 15],
            "hidden_dim": 1024,
            "num_layers": 16,
            "examples": [
                ["東京タワーの高さは", 30, 0.7],
                ["日本の人口は", 20, 0.7],
                ["光の速度は", 20, 0.7],
                ["架空の国アトランティスの首都は", 30, 0.7],
            ]
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "layers": [0, 11, 21],
            "hidden_dim": 2048,
            "num_layers": 22,
            "examples": [
                ["The capital of Japan is", 30, 0.7],
                ["The height of Mount Fuji is", 30, 0.7],
                ["The speed of light is", 20, 0.7],
                ["The population of Mars in 2050 will be", 30, 0.7],
            ]
        }
    }
    
    # Fallback config for unrecognised model names
    default_config = {
        "layers": [0, 6, 11],
        "hidden_dim": 768,
        "num_layers": 12,
        "examples": [
            ["The capital of the United States is", 30, 0.7],
            ["Mount Everest is located in", 30, 0.7],
        ]
    }
    
    return configs.get(model_name, default_config)


def create_demo_interface(model_name: str = "gpt2", checkpoint_path: str = None):
    """Create the Gradio interface."""

    # Get model-specific config
    model_config = get_model_config(model_name)

    # Initialize demo app
    demo_app = ConfidenceDemo(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        layers=model_config['layers'],
        hidden_dim=model_config['hidden_dim']
    )
    
    # Gradio interface
    with gr.Blocks(title="LLM Confidence Monitor") as demo:

        # ── Header ──────────────────────────────────────────────────
        gr.HTML("""
        <div style="border-bottom:1px solid #e5e7eb; padding-bottom:0.75rem; margin-bottom:1rem;">
          <h1 style="font-size:1.4rem; font-weight:700; margin:0 0 0.25rem;">LLM Confidence Monitor</h1>
          <p style="margin:0 0 0.6rem; color:#57606a; font-size:0.9rem;">
            Displays the <strong>confidence score</strong> of each token generated by the LLM in real-time streaming.
          </p>
          <div style="display:flex; gap:1.5rem; font-size:0.8rem; color:#374151;">
            <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#4CAF50;margin-right:5px;vertical-align:middle;"></span><strong>High</strong> &nbsp;≥ 0.7</span>
            <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#FF9800;margin-right:5px;vertical-align:middle;"></span><strong>Medium</strong> &nbsp;0.4–0.7</span>
            <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#F44336;margin-right:5px;vertical-align:middle;"></span><strong>Low</strong> &nbsp;&lt; 0.4</span>
          </div>
        </div>
        """)

        # ── Prompt ───────────────────────────────────────────────────
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter a question (e.g. The height of Mount Fuji is)",
            lines=5,
            max_lines=10,
        )

        # ── Basic parameters ─────────────────────────────────────────
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="Max tokens",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                label="Temperature",
            )

        # ── Advanced parameters ──────────────────────────────────────
        with gr.Accordion("Advanced parameters", open=False):
            with gr.Row():
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top-p",
                    info="Restrict sampling to tokens whose cumulative probability ≤ top_p. 1.0 = disabled."
                )
                top_k = gr.Slider(
                    minimum=0, maximum=200, value=0, step=10,
                    label="Top-k",
                    info="Keep only the top-k highest-probability tokens. 0 = disabled."
                )
            with gr.Row():
                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.0, step=0.05,
                    label="Repetition penalty",
                    info="Penalise tokens that have already been generated. 1.0 = disabled."
                )
                no_repeat_ngram_size = gr.Slider(
                    minimum=0, maximum=5, value=0, step=1,
                    label="No-repeat n-gram size",
                    info="Forbid repeating any n-gram of this length. 0 = disabled."
                )
            with gr.Row():
                min_new_tokens = gr.Slider(
                    minimum=0, maximum=50, value=0, step=5,
                    label="Min new tokens",
                    info="Suppress EOS until at least this many tokens have been generated."
                )
                do_sample = gr.Checkbox(
                    value=True,
                    label="Sampling",
                    info="Uncheck for greedy decoding (deterministic output)."
                )

        # ── Generate button ──────────────────────────────────────────
        generate_btn = gr.Button("Generate", variant="primary", size="lg")

        # gr.Examples(
        #     examples=model_config['examples'],
        #     inputs=[prompt_input, max_tokens, temperature],
        #     label="Examples",
        # )

        # ── Output area ──────────────────────────────────────────────
        gr.HTML('<p style="font-size:0.85rem; font-weight:600; color:#374151; margin:0.5rem 0 0.25rem;">Generated text (color-coded)</p>')
        colored_output = gr.HTML(
            value=(
                "<div style='"
                "font-size:16px; line-height:2; font-family:monospace;"
                "min-height:140px; max-height:260px; overflow-y:auto;"
                "padding:0.75rem 1rem;"
                "background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;"
                "'></div>"
            ),
            elem_classes=["output-box"],
        )
        confidence_plot = gr.Plot(label="Confidence over time")
        gr.HTML('<p style="font-size:0.85rem; font-weight:600; color:#374151; margin:0.75rem 0 0.25rem;">Token details</p>')
        tokens_table = gr.HTML(
            value=(
                "<div style='"
                "height:260px; overflow-y:auto;"
                "border:1px solid #e5e7eb; border-radius:8px;"
                "background:#ffffff;"
                "'></div>"
            ),
        )

        # Event handler (streaming)
        generate_btn.click(
            fn=demo_app.generate_with_confidence_streaming,
            inputs=[
                prompt_input, max_tokens, temperature,
                top_p, do_sample,
                top_k, repetition_penalty, no_repeat_ngram_size, min_new_tokens,
            ],
            outputs=[colored_output, tokens_table, confidence_plot]
        )

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    return demo, theme


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (e.g., gpt2, rinna/japanese-gpt2-medium, cyberagent/open-calm-small)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained probe checkpoint"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting demo with model: {args.model}")
    if args.checkpoint:
        logger.info(f"Using checkpoint: {args.checkpoint}")

    # Build and launch the Gradio interface
    demo, theme = create_demo_interface(
        model_name=args.model,
        checkpoint_path=args.checkpoint
    )

    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
        theme=theme,
    )


if __name__ == "__main__":
    main()
