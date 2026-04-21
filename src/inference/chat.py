#!/usr/bin/env python3
"""Interactive chat REPL for max_llm trained models.

Usage:
    python -m src.inference.chat \
    --config config/milestones/<experiment>.toml \
    --checkpoint outputs/<run>/checkpoint.pt
"""

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
from src.models.learning_model import LearningModel
from src.tokenizer import Tokenizer


def load_checkpoint_model(
    config_path: str,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> tuple[LearningModel, Tokenizer, ExperimentConfig, torch.device]:
    """Return (model, tokenizer, config, device) loaded from config_path and optional checkpoint."""
    config = ExperimentConfig.from_toml(config_path)
    device_obj = resolve_device(device)
    print(f"📍 Device: {device_obj}")

    # Honour training attention backend, but force a CPU-safe backend.
    attn_backend = getattr(config.training, "attention_backend", "standard")
    if device_obj.type == "cpu" and attn_backend != "standard":
        print(
            f"⚠ Attention backend '{attn_backend}' is not supported on CPU; "
            "falling back to 'standard'"
        )
        attn_backend = "standard"
    model = LearningModel.from_config(config.model, attention_backend=attn_backend).to(device_obj)

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

    tokenizer = create_tokenizer_from_data_config(config.data)
    model.eval()
    return model, tokenizer, config, device_obj


def chat_mode(
    model: LearningModel,
    tokenizer: Tokenizer,
    config: ExperimentConfig,
    device: torch.device,
    max_tokens: int | None = None,
) -> None:
    """Run the interactive generation REPL until Ctrl+C."""
    temperature = config.inference.temperature
    top_p = config.inference.top_p
    top_k = config.inference.top_k
    repetition_penalty = config.inference.repetition_penalty
    repetition_window = config.inference.repetition_window
    max_new_tokens = max_tokens or config.inference.max_new_tokens
    max_seq_len = config.model.max_seq_length

    print("\n🤖 Chat Mode Ready")
    print(
        f"   Model: hidden_size={config.model.hidden_size}, layers={config.model.num_layers}, vocab={config.model.vocab_size}"
    )
    print(f"   Data: {config.data.dataset_path}")
    print(
        f"   Sampling: temp={temperature}, top_p={top_p}, top_k={top_k}, "
        f"rep_penalty={repetition_penalty}, max_tokens={max_new_tokens}"
    )
    print("\n💬 Type prompts below (Ctrl+C to exit):\n")

    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                break

            if not prompt:
                continue

            try:
                prompt_tokens = tokenizer.encode(prompt)
            except Exception as e:
                print(f"❌ Tokenization error: {e}")
                continue

            if not prompt_tokens:
                print("❌ Prompt produced no tokens")
                continue

            try:
                tokens = list(prompt_tokens)
                if len(tokens) > max_seq_len:
                    tokens = tokens[-max_seq_len:]

                # autocast so flash/xformers backends receive fp16/bf16
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.no_grad()
                )
                print("🤖 ", end="", flush=True)
                gen_start = len(tokens)
                prev_decoded_len = 0
                with torch.no_grad(), autocast_ctx:
                    for _ in range(max_new_tokens):
                        input_ids = torch.tensor(
                            tokens[-max_seq_len:], dtype=torch.long, device=device
                        ).unsqueeze(0)
                        next_token = sample_token(
                            model(input_ids)[0, -1],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            repetition_window=repetition_window,
                            context=tokens,
                        )
                        tokens.append(next_token)
                        decoded = tokenizer.decode(tokens[gen_start:])
                        print(decoded[prev_decoded_len:], end="", flush=True)
                        prev_decoded_len = len(decoded)

                print("\n")
            except Exception as e:
                print(f"❌ Generation error: {e}\n")
                import traceback

                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat with trained LM")
    parser.add_argument("--config", required=True, help="Path to experiment config (TOML)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_new_tokens")
    args = parser.parse_args()

    model, tokenizer, config, device = load_checkpoint_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    chat_mode(
        model=model, tokenizer=tokenizer, config=config, device=device, max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
