#!/usr/bin/env python3
"""Interactive chat mode for max_llm trained models.

Loads a checkpoint and provides a simple REPL for generating text.

Usage:
    python -m src.inference.chat \
        --config config/experiment_curriculum.toml \
        --checkpoint outputs/curriculum-alternating/checkpoint.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from src.config.experiment import ExperimentConfig
from src.models.learning_model import SimpleLM
from src.tokenizer import TokenizerFactory


def load_checkpoint_model(
    config_path: str,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> tuple:
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
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"📍 Device: {device_obj}")

    # Create model
    model = SimpleLM.from_config(config.model).to(device_obj)

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            print(f"📂 Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device_obj)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                for key in ("model_state", "state_dict", "model"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        model.load_state_dict(checkpoint[key], strict=False)
                        print("✓ Checkpoint loaded")
                        break
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    print("✓ Checkpoint loaded")
            print("✓ Model ready")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_file}")
            print("⚠ Using random weights")
    else:
        print("⚠ No checkpoint provided; using random weights")

    # Create tokenizer
    tokenizer_kwargs: dict[str, Any] = {"mode": config.data.tokenizer_mode}
    if config.data.tokenizer_mode == "codepoint":
        tokenizer_kwargs["vocab_size"] = config.data.tokenizer_vocab_size
    tokenizer = TokenizerFactory.create(config.data.tokenizer_name, **tokenizer_kwargs)

    model.eval()

    return model, tokenizer, config, device_obj


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    """Sample next token from logits."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(
            logits < min_value, torch.tensor(float("-inf"), device=logits.device), logits
        )

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = torch.where(
            cutoff, torch.tensor(float("-inf"), device=logits.device), sorted_logits
        )
        logits = torch.empty_like(logits).scatter(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def chat_mode(
    model: SimpleLM,
    tokenizer: TokenizerFactory,
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

                with torch.no_grad():
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
