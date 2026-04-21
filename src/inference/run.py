"""Inference entrypoint for max-llm.

Usage:
    python -m src.inference.run --config config/milestones/<experiment>.toml --prompt "Hello"
"""

import argparse

import torch

from src.config.experiment import ExperimentConfig
from src.inference.sampler import sample_token
from src.inference.utils import (
    create_tokenizer_from_data_config,
    load_checkpoint_into_model,
    resolve_device,
)
from src.models.learning_model import LearningModel


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
    repetition_penalty = inference.repetition_penalty
    repetition_window = inference.repetition_window

    tokenizer = create_tokenizer_from_data_config(config.data)

    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("Prompt produced no tokens; check tokenizer settings.")

    attn_backend = getattr(config.training, "attention_backend", "standard")
    if device.type == "cpu" and attn_backend != "standard":
        attn_backend = "standard"
    model = LearningModel.from_config(config.model, attention_backend=attn_backend).to(device)
    model.eval()

    if args.checkpoint:
        load_checkpoint_into_model(model, args.checkpoint, device)
    else:
        print("Warning: no checkpoint provided; using random weights.")

    tokens = list(prompt_tokens)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.no_grad()
    )
    with torch.no_grad(), autocast_ctx:
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)[0, -1]
            next_token = sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                context=tokens,
            )
            tokens.append(next_token)

    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
