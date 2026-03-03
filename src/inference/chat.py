#!/usr/bin/env python3
"""Interactive chat mode for max_llm trained models.

Loads a checkpoint and provides a simple REPL for generating text.

Usage:
    python -m src.inference.chat \
    --config config/milestones/<experiment>.toml \
    --checkpoint outputs/<run>/checkpoint.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.config.experiment import ExperimentConfig
from src.inference.sampler import sample_token
from src.inference.utils import (
    create_tokenizer_from_data_config,
    load_checkpoint_into_model,
    resolve_device,
)
from src.models.learning_model import AttentionLM, BaseLearningModel, DecoderLM, SimpleLM
from src.tokenizer import Tokenizer


def load_checkpoint_model(
    config_path: str,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> tuple[BaseLearningModel, Tokenizer, ExperimentConfig, torch.device]:
    """Load model from checkpoint.

    Args:
        config_path: Path to experiment config
        checkpoint_path: Path to checkpoint.pt, or None for random init
        device: "auto", "cuda", or "cpu"

    Returns:
        (model, tokenizer, config, device_obj)
    """
    # Load config
    config = ExperimentConfig.from_toml(config_path)

    # Determine device
    device_obj = resolve_device(device)
    print(f"📍 Device: {device_obj}")

    # Honour the attention backend specified in the training config so that
    # flash/xformers models don't crash at inference time (no AMP context →
    # fp32 tensors → flash requires fp16/bf16).
    attn_backend = getattr(config.training, "attention_backend", "standard")

    # Create model based on model_type
    model: BaseLearningModel
    if config.model.model_type == "simple_lm":
        model = SimpleLM.from_config(config.model).to(device_obj)
    elif config.model.model_type == "attention_lm":
        model = AttentionLM.from_config(config.model, attention_backend=attn_backend).to(device_obj)
    elif config.model.model_type == "decoder_lm":
        model = DecoderLM.from_config(config.model, attention_backend=attn_backend).to(device_obj)
    else:
        raise ValueError(
            f"Unknown model_type: {config.model.model_type}. "
            "Supported types: simple_lm, attention_lm, decoder_lm"
        )

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            print(f"📂 Loading checkpoint: {checkpoint_file}")
            load_checkpoint_into_model(model, str(checkpoint_file), device_obj)
            print("✓ Checkpoint loaded")
            print("✓ Model ready")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_file}")
            print("⚠ Using random weights")
    else:
        print("⚠ No checkpoint provided; using random weights")

    # Create tokenizer
    tokenizer = create_tokenizer_from_data_config(config.data)

    model.eval()

    return model, tokenizer, config, device_obj


def chat_mode(
    model: BaseLearningModel,
    tokenizer: Tokenizer,
    config: ExperimentConfig,
    device: torch.device,
    max_tokens: int | None = None,
) -> None:
    """Interactive chat REPL.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: Config
        device: Device object
        max_tokens: Override max_new_tokens from config
    """
    temperature = config.inference.temperature
    top_p = config.inference.top_p
    top_k = config.inference.top_k
    max_new_tokens = max_tokens or config.inference.max_new_tokens

    print("\n🤖 Chat Mode Ready")
    print(
        f"   Model: hidden_size={config.model.hidden_size}, "
        f"layers={config.model.num_layers}, "
        f"vocab={config.model.vocab_size}"
    )
    print(f"   Data: {config.data.dataset_path}")
    print(
        f"   Sampling: temp={temperature}, top_p={top_p}, top_k={top_k}, "
        f"max_tokens={max_new_tokens}"
    )
    print("\n💬 Type prompts below (Ctrl+C to exit):\n")

    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                # Handle piped input
                break

            if not prompt:
                continue

            # Tokenize prompt
            try:
                prompt_tokens = tokenizer.encode(prompt)
            except Exception as e:
                print(f"❌ Tokenization error: {e}")
                continue

            if not prompt_tokens:
                print("❌ Prompt produced no tokens")
                continue

            # Generate
            try:
                tokens = list(prompt_tokens)

                # Use autocast so flash/xformers backends receive the expected dtype
                # (fp16/bf16) even without an explicit AMP training loop.
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.no_grad()
                )
                with torch.no_grad(), autocast_ctx:
                    for _ in range(max_new_tokens):
                        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(
                            0
                        )
                        logits = model(input_ids)[0, -1]
                        next_token = sample_token(
                            logits, temperature=temperature, top_p=top_p, top_k=top_k
                        )
                        tokens.append(next_token)

                # Decode
                output_text = tokenizer.decode(tokens)

                print(f"🤖 {output_text}\n")
            except Exception as e:
                print(f"❌ Generation error: {e}\n")
                import traceback

                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with trained LM")
    parser.add_argument("--config", required=True, help="Path to experiment config (TOML)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_new_tokens")

    args = parser.parse_args()

    # Load model
    model, tokenizer, config, device = load_checkpoint_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Chat mode
    chat_mode(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
