"""Shared scoring utilities for MCQ benchmark evaluation."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _fit_context_window(
    prompt_tokens: list[int],
    choice_tokens: list[int],
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Trim left context to fit model context while preserving completion tokens.

    We prefer keeping full completion tokens and trim prompt prefix first.
    """
    if max_seq_len <= 1:
        return prompt_tokens[-1:], choice_tokens[:1]

    total_len = len(prompt_tokens) + len(choice_tokens)
    if total_len <= max_seq_len:
        return prompt_tokens, choice_tokens

    keep_prompt = max(0, max_seq_len - len(choice_tokens))
    if keep_prompt == 0:
        # Completion itself exceeds context; keep tail of completion.
        return [], choice_tokens[-max_seq_len:]
    return prompt_tokens[-keep_prompt:], choice_tokens


def score_choice_logprob(
    model: nn.Module,
    tokenizer: TokenizerLike,
    prompt: str,
    choice: str,
    max_seq_len: int,
    length_normalize: bool,
    device: torch.device,
) -> float:
    """Return log-likelihood score for choice conditioned on prompt.

    If `length_normalize` is True, returns average token log-probability.
    """
    prompt_tokens = tokenizer.encode(prompt)
    choice_tokens = tokenizer.encode(choice)
    if not choice_tokens:
        return float("-inf")

    prompt_tokens, choice_tokens = _fit_context_window(prompt_tokens, choice_tokens, max_seq_len)
    full = prompt_tokens + choice_tokens
    if len(full) < 2:
        return float("-inf")

    input_ids = torch.tensor(full[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
        logits = model(input_ids)[0]  # (T-1, V)
        log_probs = F.log_softmax(logits, dim=-1)

    # Target positions associated with completion tokens only.
    start = max(0, len(prompt_tokens) - 1)
    if start >= targets.shape[0]:
        return float("-inf")

    target_slice = targets[start:]
    token_log_probs = log_probs[start:].gather(1, target_slice.unsqueeze(1)).squeeze(1)
    score_sum = float(token_log_probs.sum().item())

    if not length_normalize:
        return score_sum
    denom = max(1, target_slice.shape[0])
    return score_sum / denom


def safe_accuracy(num_correct: int, num_examples: int) -> float:
    """Return accuracy with safe zero-example handling."""
    if num_examples <= 0:
        return float("nan")
    return num_correct / num_examples


def is_valid_accuracy(value: float) -> bool:
    """True if value is a finite number in [0, 1]."""
    return math.isfinite(value) and 0.0 <= value <= 1.0
