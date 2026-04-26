"""Attention mask helpers shared by attention backends."""

import torch


def document_causal_bias(
    document_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return additive attention bias that blocks cross-document attention.

    Args:
        document_ids: ``(B, T)`` integer ids. Positions with id < 0 are ignored
            positions; they produce finite attention rows but are blocked as keys.
        dtype: Floating dtype for the additive bias.

    Returns:
        ``(B, 1, T, T)`` additive mask with ``-inf`` where a query may not
        attend to a key from another packed document.
    """
    if document_ids.ndim != 2:
        raise ValueError(f"document_ids must be (B, T), got {tuple(document_ids.shape)}")

    valid = document_ids >= 0
    same_doc = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)
    valid_keys = valid.unsqueeze(1)
    valid_queries = valid.unsqueeze(2)

    # Invalid/padded queries are ignored by the loss; allow finite rows to
    # avoid all--inf softmax rows while still preventing them as keys.
    allowed = (same_doc & valid_queries & valid_keys) | ~valid_queries
    bias = torch.zeros(
        (*document_ids.shape, document_ids.shape[1]),
        device=document_ids.device,
        dtype=dtype,
    )
    return bias.masked_fill(~allowed, float("-inf")).unsqueeze(1)
