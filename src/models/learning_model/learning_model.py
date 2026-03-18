"""Universal learning model: token embeddings + optional transformer layers + LM head.

Supports Phase 2 (embedding-only, num_layers=0) through Phase 4+ (transformer-based).
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from typing_extensions import Self

from src.config.model import ModelConfig
from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.norm import make_norm
from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.learned_position import LearnedPositionEmbedding
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding
from src.models.transformer.transformer_block import TransformerBlock


def _build_pos_modules(
    pos_type: str,
    head_dim: int,
    max_seq_len: int,
    num_heads: int,
    rope_base: int | None,
    rel_pos_num_buckets: int,
    d_model: int,
) -> tuple[
    LearnedPositionEmbedding | None,
    RotaryEmbedding | AdditiveRoPE | None,
    ALiBi | RelativePositionBias | None,
]:
    """Construct positional encoding modules for a given pos_type."""
    pos_emb: LearnedPositionEmbedding | None = (
        LearnedPositionEmbedding(max_seq_len, d_model) if pos_type == "learned" else None
    )
    rope: RotaryEmbedding | AdditiveRoPE | None
    if pos_type == "rope":
        rope = RotaryEmbedding(head_dim, max_seq_len, rope_base or 10000)
    elif pos_type == "add_rope":
        rope = AdditiveRoPE(head_dim, max_seq_len, rope_base or 10000)
    else:
        rope = None
    attn_bias: ALiBi | RelativePositionBias | None
    if pos_type == "alibi":
        attn_bias = ALiBi(num_heads)
    elif pos_type == "rel_pos":
        attn_bias = RelativePositionBias(num_heads, rel_pos_num_buckets)
    else:
        attn_bias = None
    return pos_emb, rope, attn_bias


class LearningModel(nn.Module):
    """Universal language model supporting Phase 2–7 configurations.

    Architecture:
        1. Token embedding + Learned positional embedding
        2. Stack of N transformer blocks (pre-norm attention + FFN with residuals)
           - When num_layers=0: skips transformer blocks (Phase 2 embedding-only mode)
           - When num_layers>0: applies N transformer blocks (Phase 3+ transformer mode)
        3. Optional final layer norm (skipped when num_layers=0)
        4. LM head (weight-tied to token embedding)

    Weight tying: lm_head shares its weight matrix with token_embedding,
    halving the parameter count and tying input/output representations.

    Args:
        vocab_size: Size of vocabulary.
        d_model: Model dimension (embedding and hidden size).
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_kv_heads: K/V heads for GQA/MQA (None = MHA, 1 = MQA, N = GQA).
        dropout: Dropout probability (default: 0.0).
        ff_expansion_ratio: Expansion ratio for FFN hidden dimension (default: 4).
        max_seq_len: Maximum sequence length (default: 2048).
        attention_backend: Attention backend to use (default: "flash").
            Options: "flash", "sage", "xformers", "standard"
        pos_type: Positional encoding type. One of "learned", "rope", "add_rope",
            "alibi", "rel_pos" (default: "learned").
        rel_pos_num_buckets: Number of relative position buckets (used when
            pos_type == "rel_pos", default: 32).
        rope_base: Frequency base for RoPE/AddRoPE (required when
            pos_type in {"rope", "add_rope"}).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        ff_expansion_ratio: int = 4,
        intermediate_size: int | None = None,
        ffn_type: str = "gelu",
        max_seq_len: int = 2048,
        attention_backend: str = "flash",
        embedding_dim: int | None = None,
        share_layer_weights: bool = False,
        norm_type: str = "layer",
        rope_base: int | None = None,
        pos_type: str = "learned",
        rel_pos_num_buckets: int = 32,
        attn_type: str = "mha",
        swa_window_size: int = 256,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.share_layer_weights = share_layer_weights

        # Factorized embeddings: use smaller embedding_dim if specified
        self.embedding_dim = embedding_dim or d_model
        self.use_factorized = embedding_dim is not None and embedding_dim < d_model

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, self.embedding_dim)

        # Back-compat: rope_base set but pos_type still "learned" → promote to "rope"
        if rope_base is not None and pos_type == "learned":
            pos_type = "rope"

        head_dim = d_model // num_heads
        pos_emb, rope, attn_bias = _build_pos_modules(
            pos_type, head_dim, max_seq_len, num_heads, rope_base, rel_pos_num_buckets, d_model
        )
        self.position_embedding = pos_emb

        # Projection layer for factorized embeddings
        self.embedding_projection: nn.Linear | None
        if self.use_factorized:
            self.embedding_projection = nn.Linear(self.embedding_dim, d_model)
        else:
            self.embedding_projection = None

        # Transformer blocks
        # Optional cross-layer parameter sharing: reuse one block N times.
        def _make_block() -> TransformerBlock:
            return TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                ff_expansion_ratio=ff_expansion_ratio,
                intermediate_size=intermediate_size,
                ffn_type=ffn_type,
                attention_backend=attention_backend,
                num_layers=num_layers,
                norm_type=norm_type,
                rope=rope,
                attn_bias=attn_bias,
                attn_type=attn_type,
                window_size=swa_window_size,
            )

        if self.share_layer_weights:
            self.blocks = nn.ModuleList([_make_block()])
        else:
            self.blocks = nn.ModuleList([_make_block() for _ in range(num_layers)])

        if self.share_layer_weights:
            assert len(self.blocks) == 1, "share_layer_weights=True must create exactly one block"
        else:
            assert (
                len(self.blocks) == num_layers
            ), "share_layer_weights=False must create one block per layer"

        # Final norm (skipped for num_layers=0 to support Phase 2 embedding-only mode)
        self.final_norm: nn.Module | None = (
            make_norm(norm_type, d_model) if num_layers > 0 else None
        )

        # LM head with weight tying (only when not using factorized embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if not self.use_factorized:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"LearningModel expects 2-D input (batch, seq_len), got shape {x.shape}"
        assert x.dtype == torch.long, f"LearningModel expects dtype=torch.long, got {x.dtype}"
        B, T = x.shape

        # Token and position embeddings
        tok_emb = self.token_embedding(x)  # (B, T, embedding_dim)

        # Project embeddings to d_model if using factorized embeddings
        if self.use_factorized:
            assert self.embedding_projection is not None
            tok_emb = self.embedding_projection(tok_emb)  # (B, T, d_model)

        # Learned position embedding (skipped for RoPE/ALiBi/RelPos variants)
        if self.position_embedding is not None:
            h = tok_emb + self.position_embedding(x)  # (B, T, d_model)
        else:
            h = tok_emb

        h = self._apply_transformer_blocks(h)

        # Final layer norm (skipped for num_layers=0 to support Phase 2 MLP-only models)
        if self.final_norm is not None:
            h = self.final_norm(h)

        # LM head
        logits = self.lm_head(h)

        assert logits.shape == (
            B,
            T,
            self.vocab_size,
        ), f"LearningModel output shape mismatch: expected {(B, T, self.vocab_size)}, got {logits.shape}"
        return logits

    def _apply_transformer_blocks(self, h: torch.Tensor) -> torch.Tensor:
        """Apply transformer stack, honoring optional cross-layer parameter sharing."""
        if self.share_layer_weights:
            shared_block = self.blocks[0]
            for _ in range(self.num_layers):
                h = shared_block(h)
            return h

        for block in self.blocks:
            h = block(h)
        return h

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load state dict and re-establish weight tying after restore."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if not self.use_factorized:
            self.lm_head.weight = self.token_embedding.embedding.weight
        return result

    @classmethod
    def from_config(cls, config: ModelConfig, attention_backend: str = "flash") -> Self:
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            intermediate_size=config.intermediate_size,
            ffn_type=config.ffn_type,
            max_seq_len=config.max_seq_length,
            attention_backend=attention_backend,
            embedding_dim=config.embedding_dim,
            share_layer_weights=config.share_layer_weights,
            norm_type=config.norm_type,
            rope_base=config.rope_base,
            pos_type=config.pos_type,
            rel_pos_num_buckets=config.rel_pos_num_buckets,
            attn_type=config.attn_type,
            swa_window_size=config.swa_window_size,
        )
