"""Inference entrypoint for max-llm.

Usage:
    python -m src.inference.run --config config/experiment.toml --prompt "Hello"
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from src.config.experiment import ExperimentConfig
from src.models.learning_model import SimpleLM
from src.tokenizer import TokenizerFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a max-llm model")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (.pt or .pth)")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override device")
    parser.add_argument("--max-new-tokens", type=int, help="Override max_new_tokens")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--top-p", type=float, help="Override top_p")
    parser.add_argument("--top-k", type=int, help="Override top_k")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(model: SimpleLM, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                model.load_state_dict(checkpoint[key], strict=False)
                return
        model.load_state_dict(checkpoint, strict=False)
        return
    raise ValueError("Unsupported checkpoint format")


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
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


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)
    inference = config.inference

    device_choice = args.device if args.device is not None else inference.device
    device = resolve_device(device_choice)

    max_new_tokens = args.max_new_tokens or inference.max_new_tokens
    temperature = args.temperature if args.temperature is not None else inference.temperature
    top_p = args.top_p if args.top_p is not None else inference.top_p
    top_k = args.top_k if args.top_k is not None else inference.top_k

    tokenizer_kwargs: dict[str, Any] = {"mode": config.data.tokenizer_mode}
    if config.data.tokenizer_mode == "codepoint":
        tokenizer_kwargs["vocab_size"] = config.data.tokenizer_vocab_size
    tokenizer = TokenizerFactory.create(config.data.tokenizer_name, **tokenizer_kwargs)

    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("Prompt produced no tokens; check tokenizer settings.")

    model = SimpleLM.from_config(config.model).to(device)
    model.eval()

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    else:
        print("Warning: no checkpoint provided; using random weights.")

    tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)[0, -1]
            next_token = sample_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            tokens.append(next_token)

    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
