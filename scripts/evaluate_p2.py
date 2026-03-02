#!/usr/bin/env python3
"""Evaluate Phase 2 model: loss, perplexity, generation samples.

Usage:
    python scripts/evaluate_p2.py --config config/milestones/<experiment>.toml \\
        --checkpoint outputs/<run>/checkpoint.pt \\
        --prompt "Hello world" \\
        --num-samples 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment import ExperimentConfig
from src.inference.sampler import sample_token
from src.inference.utils import create_tokenizer_from_data_config, load_checkpoint_into_model
from src.models.learning_model import SimpleLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 model")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment TOML config")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint file")
    parser.add_argument("--prompt", default="Hello", type=str, help="Text prompt for generation")
    parser.add_argument("--num-samples", default=3, type=int, help="Number of generation samples")
    parser.add_argument("--device", default="cpu", type=str, help="Device: cpu or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Load config
    config = ExperimentConfig.from_toml(args.config)
    device = torch.device(args.device)

    # Load model
    print("[Phase 2 Model Evaluation]\n")
    print(f"Config      : {args.config.name}")
    print(f"Checkpoint  : {args.checkpoint.name}")
    print(f"Device      : {device}")
    print(
        f"Model arch  : hidden_size={config.model.hidden_size}, vocab_size={config.model.vocab_size}"
    )
    print()

    model = SimpleLM.from_config(config.model).to(device)
    model.eval()
    load_checkpoint_into_model(model, str(args.checkpoint), device)

    # Load tokenizer
    tokenizer = create_tokenizer_from_data_config(config.data)

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print("[Prompt Analysis]")
    print(f"Text        : '{args.prompt}'")
    print(f"Tokens      : {len(prompt_tokens)} tokens")
    print(f"Token IDs   : {prompt_tokens[:20]}{'...' if len(prompt_tokens) > 20 else ''}")
    print()

    # Generate samples
    print(f"[Generation Samples (n={args.num_samples})]")
    print(
        f"Sampling    : temp={config.inference.temperature}, "
        f"top_p={config.inference.top_p}, top_k={config.inference.top_k}"
    )
    print(f"Max tokens  : {config.inference.max_new_tokens}")
    print()

    with torch.no_grad():
        for i in range(args.num_samples):
            tokens = list(prompt_tokens)
            for _ in range(config.inference.max_new_tokens):
                input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
                logits = model(input_ids)[0, -1]
                next_token = sample_token(
                    logits,
                    temperature=config.inference.temperature,
                    top_p=config.inference.top_p,
                    top_k=config.inference.top_k,
                )
                tokens.append(next_token)

            generated_text = tokenizer.decode(tokens)
            print(f"Sample {i+1}:")
            print(f"  {generated_text[:300]}{'...' if len(generated_text) > 300 else ''}")
            print()

    print("[Evaluation Complete]")


if __name__ == "__main__":
    main()
