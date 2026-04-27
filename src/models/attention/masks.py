"""Attention mask helpers shared by attention backends."""

import torch


def cu_seqlens_from_document_ids(
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Compute cumulative sequence lengths for Flash varlen attention.

    Each batch row is split into segments at every change in ``document_ids``
    (including transitions to/from invalid ids ``< 0``). Padded positions form
    their own segments so they only attend to themselves; the training-side
    loss mask discards their predictions.

    Args:
        document_ids: ``(B, T)`` integer ids per packed token.

    Returns:
        Tuple ``(cu_seqlens, max_seqlen)`` where ``cu_seqlens`` is an
        ``int32`` tensor of shape ``(num_segments + 1,)`` and ``max_seqlen``
        is a Python int safe upper bound (the row sequence length ``T``).
    """
    if document_ids.ndim != 2:
        raise ValueError(f"document_ids must be (B, T), got {tuple(document_ids.shape)}")

    B, T = document_ids.shape
    device = document_ids.device
    if T == 0:
        return torch.zeros(1, dtype=torch.int32, device=device), 0

    # Mark segment starts: position 0 of every row, and any in-row id change.
    starts = torch.zeros(B, T, dtype=torch.bool, device=device)
    starts[:, 0] = True
    if T > 1:
        starts[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]

    flat_starts = starts.flatten()
    start_idx = flat_starts.nonzero(as_tuple=True)[0].to(torch.int32)
    total = torch.tensor([B * T], device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([start_idx, total])
    return cu_seqlens, T


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
