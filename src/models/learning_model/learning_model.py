"""Universal learning model: token embeddings + optional transformer layers + LM head.

Supports Phase 2 (embedding-only, num_layers=0) through Phase 4+ (transformer-based).
"""

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import Self

from src.config.model import ModelConfig
from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.kv_cache import LayerKVCache, ModelKVCache
from src.models.norm import make_norm
from src.models.position.add_rope import AdditiveRoPE
from src.models.position.alibi import ALiBi
from src.models.position.learned_position import LearnedPositionEmbedding
from src.models.position.rel_pos_bias import RelativePositionBias
from src.models.position.rope import RotaryEmbedding
from src.models.residual.attn_residual import AttnResidual
from src.models.transformer.transformer_block import TransformerBlock


def _build_pos_modules(
    pos_type: str,
    head_dim: int,
    max_seq_len: int,
    num_heads: int,
    rope_base: int | None,
    rel_pos_num_buckets: int,
    d_model: int,
    rope_scaling_factor: float = 1.0,
    rope_low_freq_factor: float = 1.0,
    rope_high_freq_factor: float = 4.0,
    rope_original_max_seq_len: int | None = None,
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
        rope = RotaryEmbedding(
            head_dim,
            max_seq_len,
            rope_base or 10000,
            scaling_factor=rope_scaling_factor,
            low_freq_factor=rope_low_freq_factor,
            high_freq_factor=rope_high_freq_factor,
            original_max_seq_len=rope_original_max_seq_len,
        )
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
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 4.0,
        rope_original_max_seq_len: int | None = None,
        mla_latent_dim: int | None = None,
        attn_type: str = "mha",
        swa_window_size: int = 256,
        res_type: str = "standard",
        attn_res_num_blocks: int = 8,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.share_layer_weights = share_layer_weights
        self.gradient_checkpointing = False
        self.gradient_checkpointing_mode = "full"
        self.gradient_checkpointing_interval = 1

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
            pos_type,
            head_dim,
            max_seq_len,
            num_heads,
            rope_base,
            rel_pos_num_buckets,
            d_model,
            rope_scaling_factor=rope_scaling_factor,
            rope_low_freq_factor=rope_low_freq_factor,
            rope_high_freq_factor=rope_high_freq_factor,
            rope_original_max_seq_len=rope_original_max_seq_len,
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
                mla_latent_dim=mla_latent_dim,
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

        # Attention Residuals (optional depth-wise attention over layer outputs)
        self.res_type = res_type
        self.attn_res_num_blocks = attn_res_num_blocks
        self.attn_res: AttnResidual | None = None
        if res_type in ("full_attn", "block_attn") and num_layers > 0:
            # 2 sublayers per transformer block (attn + ffn), +1 for final aggregation
            self.attn_res = AttnResidual(num_sublayers=2 * num_layers, d_model=d_model)

        # Final norm (skipped for num_layers=0 to support Phase 2 embedding-only mode)
        self.final_norm: nn.Module | None = (
            make_norm(norm_type, d_model) if num_layers > 0 else None
        )

        # LM head with weight tying (only when not using factorized embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if not self.use_factorized:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: "ModelKVCache | None" = None,
    ) -> torch.Tensor:

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
            pos_offset = kv_caches.length if kv_caches is not None else 0
            h = tok_emb + self.position_embedding(x, pos_offset=pos_offset)  # (B, T, d_model)
        else:
            h = tok_emb

        h = self._apply_transformer_blocks(h, kv_caches=kv_caches)

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

    def _apply_transformer_blocks(
        self,
        h: torch.Tensor,
        kv_caches: "ModelKVCache | None" = None,
    ) -> torch.Tensor:
        """Dispatch to the appropriate residual variant."""
        if self.res_type == "full_attn":
            return self._apply_blocks_full_attn_res(h, kv_caches=kv_caches)
        if self.res_type == "block_attn":
            return self._apply_blocks_block_attn_res(h, kv_caches=kv_caches)
        return self._apply_blocks_standard(h, kv_caches=kv_caches)

    def _apply_blocks_standard(
        self,
        h: torch.Tensor,
        kv_caches: "ModelKVCache | None" = None,
    ) -> torch.Tensor:
        """Standard pre-norm residual stack (original behaviour)."""
        if self.share_layer_weights:
            shared_block = cast(TransformerBlock, self.blocks[0])
            for i in range(self.num_layers):
                kv = kv_caches[i] if kv_caches is not None else None
                h = self._run_block(shared_block, h, i, kv_cache=kv)
            return h
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            h = self._run_block(cast(TransformerBlock, block), h, i, kv_cache=kv)
        return h

    def _should_checkpoint(self, block_idx: int, kv_cache: LayerKVCache | None = None) -> bool:
        return (
            self.training
            and self.gradient_checkpointing
            and kv_cache is None
            and self.num_layers > 0
            and block_idx % self.gradient_checkpointing_interval == 0
        )

    @staticmethod
    def _checkpoint_tensor_fn(fn: Any, *args: torch.Tensor) -> torch.Tensor:
        return checkpoint(fn, *args, use_reentrant=False)

    def _run_block(
        self,
        block: TransformerBlock,
        h: torch.Tensor,
        block_idx: int,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        if not self._should_checkpoint(block_idx, kv_cache):
            return block(h, kv_cache=kv_cache)
        if self.gradient_checkpointing_mode == "ffn":
            h = block._apply_attention_residual(h, kv_cache=None)  # noqa: SLF001
            return self._checkpoint_tensor_fn(block._apply_feedforward_residual, h)  # noqa: SLF001
        return self._checkpoint_tensor_fn(lambda tensor: block(tensor, kv_cache=None), h)

    def _run_attn_only(
        self,
        block: TransformerBlock,
        h: torch.Tensor,
        block_idx: int,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        if not self._should_checkpoint(block_idx, kv_cache):
            return block.apply_attn_only(h, kv_cache=kv_cache)
        return self._checkpoint_tensor_fn(lambda tensor: block.apply_attn_only(tensor), h)

    def _run_ffn_only(
        self, block: TransformerBlock, h: torch.Tensor, block_idx: int
    ) -> torch.Tensor:
        if not self._should_checkpoint(block_idx):
            return block.apply_ffn_only(h)
        return self._checkpoint_tensor_fn(block.apply_ffn_only, h)

    def _apply_blocks_full_attn_res(
        self,
        h: torch.Tensor,
        kv_caches: "ModelKVCache | None" = None,
    ) -> torch.Tensor:
        """Full Attention Residuals: each sublayer attends over all prior outputs."""
        assert self.attn_res is not None
        ar = self.attn_res

        values: list[torch.Tensor] = [h]
        sublayer_idx = 0

        for block_idx in range(self.num_layers):
            block = cast(
                TransformerBlock,
                self.blocks[0] if self.share_layer_weights else self.blocks[block_idx],
            )
            kv = kv_caches[block_idx] if kv_caches is not None else None
            # Attention sublayer
            h_in = ar(sublayer_idx, values)
            values.append(self._run_attn_only(block, h_in, block_idx, kv_cache=kv))
            sublayer_idx += 1

            # FFN sublayer
            h_in = ar(sublayer_idx, values)
            values.append(self._run_ffn_only(block, h_in, block_idx))
            sublayer_idx += 1

        return ar(sublayer_idx, values)

    def _apply_blocks_block_attn_res(
        self,
        h: torch.Tensor,
        kv_caches: "ModelKVCache | None" = None,
    ) -> torch.Tensor:
        """Block Attention Residuals: attend over N block-level summaries.

        Layers are grouped into N = attn_res_num_blocks groups of S = L/N blocks.
        Within each group, sublayer outputs are accumulated via simple addition
        (standard intra-block residual). Across groups, softmax attention over the
        N block-level summaries is used to compute each sublayer's input.

        num_layers must be divisible by attn_res_num_blocks.
        """
        assert self.attn_res is not None
        N = self.attn_res_num_blocks
        assert self.num_layers % N == 0, (
            f"num_layers ({self.num_layers}) must be divisible by "
            f"attn_res_num_blocks ({N}) for res_type='block_attn'"
        )
        ar = self.attn_res
        S = self.num_layers // N  # transformer blocks per group
        # b_0 = token embedding; block_sums grows to [b_0, b_1, ..., b_N]
        block_sums: list[torch.Tensor] = [h]
        sublayer_idx = 0

        for group_n in range(N):
            partial_b: torch.Tensor | None = None  # accumulates within-group outputs

            for rel_i in range(S):
                block_idx = group_n * S + rel_i
                block = cast(
                    TransformerBlock,
                    self.blocks[0] if self.share_layer_weights else self.blocks[block_idx],
                )
                kv = kv_caches[block_idx] if kv_caches is not None else None
                # --- Attention sublayer ---
                # First sublayer of the group: sources = [b_0, ..., b_{n-1}]
                # Subsequent sublayers: also include the partial block sum
                if rel_i == 0 and partial_b is None:
                    sources = block_sums
                else:
                    assert partial_b is not None
                    sources = [*block_sums, partial_b]

                h_in = ar(sublayer_idx, sources)
                delta_attn = self._run_attn_only(block, h_in, block_idx, kv_cache=kv)
                partial_b = delta_attn if partial_b is None else partial_b + delta_attn
                sublayer_idx += 1

                # --- FFN sublayer ---
                assert partial_b is not None
                sources = [*block_sums, partial_b]
                h_in = ar(sublayer_idx, sources)
                delta_ffn = self._run_ffn_only(block, h_in, block_idx)
                partial_b = partial_b + delta_ffn
                sublayer_idx += 1

            assert partial_b is not None
            block_sums.append(partial_b)

        # Final aggregation: attend over all N+1 block summaries
        return ar(sublayer_idx, block_sums)

    def gradient_checkpointing_enable(self, mode: str = "full", interval: int = 1) -> None:
        """Enable activation checkpointing for training-time transformer blocks."""
        if mode not in {"full", "ffn"}:
            raise ValueError("gradient checkpointing mode must be 'full' or 'ffn'")
        if interval < 1:
            raise ValueError("gradient checkpointing interval must be >= 1")
        self.gradient_checkpointing = True
        self.gradient_checkpointing_mode = mode
        self.gradient_checkpointing_interval = interval

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing."""
        self.gradient_checkpointing = False

    def make_kv_cache(
        self,
        max_seq_len: int | None = None,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "ModelKVCache":
        """Allocate a KV-cache for autoregressive generation.

        Args:
            max_seq_len: Maximum context length (defaults to model.max_seq_len).
            batch_size:  Batch size for the cache buffers.
            dtype:       Buffer dtype (use bfloat16 to match training precision).
            device:      Target device (defaults to the model's first parameter device).

        Returns:
            A ModelKVCache containing one LayerKVCache per transformer layer.
        """
        from src.models.attention.causal_mha import CausalMultiHeadAttention  # noqa: PLC0415
        from src.models.attention.multihead_latent_attention import (  # noqa: PLC0415
            MultiHeadLatentAttention,
        )
        from src.models.kv_cache import LayerKVCache, ModelKVCache  # noqa: PLC0415

        _max_seq_len = max_seq_len if max_seq_len is not None else self.max_seq_len
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        def _layer_cache(attn: nn.Module) -> LayerKVCache | None:
            if isinstance(attn, (CausalMultiHeadAttention, MultiHeadLatentAttention)):
                return LayerKVCache(
                    max_seq_len=_max_seq_len,
                    num_kv_heads=attn.num_kv_heads,
                    head_dim=attn.head_dim,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )
            return None

        if self.share_layer_weights:
            attn: nn.Module = self.blocks[0].attention  # type: ignore[assignment]
            layers: list[LayerKVCache | None] = [_layer_cache(attn) for _ in range(self.num_layers)]
        else:
            layers = [_layer_cache(cast(TransformerBlock, b).attention) for b in self.blocks]

        return ModelKVCache(layers)

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
            rope_scaling_factor=config.rope_scaling_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_original_max_seq_len=config.rope_original_max_seq_len,
            mla_latent_dim=config.mla_latent_dim,
            attn_type=config.attn_type,
            swa_window_size=config.swa_window_size,
            res_type=config.res_type,
            attn_res_num_blocks=config.attn_res_num_blocks,
        )
