"""Token sampling strategies for text generation.

Implements various sampling methods:
- Greedy (argmax)
- Temperature sampling
- Top-k filtering
- Top-p (nucleus) sampling
"""

from __future__ import annotations

import torch


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> int:
    """Sample next token from logits with various strategies.

    Args:
        logits: Logits for next token (vocab_size,)
        temperature: Temperature > 0 (lower = more deterministic)
                    If <= 0, uses greedy sampling (argmax)
        top_p: Nucleus sampling threshold (0.0 = disabled)
              Keeps tokens whose cumulative probability >= top_p
        top_k: Top-k filtering (0 = disabled)
              Keeps only the k most likely tokens

    Returns:
        Sampled token ID

    Examples:
        Greedy sampling:
            >>> sample_token(logits, temperature=0.0)

        Temperature sampling:
            >>> sample_token(logits, temperature=0.8)

        Top-k sampling:
            >>> sample_token(logits, temperature=0.8, top_k=50)

        Nucleus (top-p) sampling:
            >>> sample_token(logits, temperature=0.8, top_p=0.9)

        Combined:
            >>> sample_token(logits, temperature=0.8, top_k=50, top_p=0.9)
    """
    # Greedy sampling
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(
            logits < min_value,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

    # Top-p (nucleus) filtering
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        # Create cutoff mask (keep tokens until cumulative > top_p)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False  # Always keep at least one token

        # Apply cutoff and unsort
        sorted_logits = torch.where(
            cutoff,
            torch.tensor(float("-inf"), device=logits.device),
            sorted_logits,
        )
        logits = torch.empty_like(logits).scatter(0, sorted_indices, sorted_logits)

    # Multinomial sampling
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
