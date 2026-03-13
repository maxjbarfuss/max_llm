"""Token sampling strategies: greedy, temperature, top-k, top-p (nucleus)."""

import torch


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> int:
    """Sample next token from logits (vocab_size,).

    temperature <= 0: greedy (argmax).
    top_k > 0: keep only the k most likely tokens before sampling.
    0 < top_p < 1: nucleus sampling — keep tokens until cumulative prob >= top_p.
    """
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        logits = torch.where(
            logits < values[-1], torch.tensor(float("-inf"), device=logits.device), logits
        )

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False  # always keep at least one token
        sorted_logits = torch.where(
            cutoff, torch.tensor(float("-inf"), device=logits.device), sorted_logits
        )
        logits = torch.empty_like(logits).scatter(0, sorted_indices, sorted_logits)

    return int(torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item())
