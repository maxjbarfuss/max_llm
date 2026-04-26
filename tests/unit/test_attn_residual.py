"""Tests for Attention Residuals (AttnRes) implementation.

Covers:
  - AttnResidual module: shape, zero-init uniform weights, gradient flow
  - TransformerBlock: apply_attn_only / apply_ffn_only sublayer methods
  - LearningModel: full_attn and block_attn produce correct shapes and gradients
  - ModelConfig: res_type validation
"""

from __future__ import annotations

import pytest
import torch

from src.config.model import ModelConfig
from src.models.learning_model.learning_model import LearningModel
from src.models.residual.attn_residual import AttnResidual
from src.models.transformer.transformer_block import TransformerBlock

# ---------------------------------------------------------------------------
# AttnResidual unit tests
# ---------------------------------------------------------------------------


class TestAttnResidual:
    def test_output_shape(self) -> None:
        B, T, d = 2, 8, 64
        ar = AttnResidual(num_sublayers=4, d_model=d)
        sources = [torch.randn(B, T, d) for _ in range(3)]
        out = ar(0, sources)
        assert out.shape == (B, T, d)

    def test_single_source(self) -> None:
        B, T, d = 1, 4, 32
        ar = AttnResidual(num_sublayers=2, d_model=d)
        src = torch.randn(B, T, d)
        out = ar(0, [src])
        # With a single source, softmax weight = 1.0, so output == source
        assert torch.allclose(out, src, atol=1e-5)

    def test_zero_init_uniform_weights(self) -> None:
        """Zero-init queries → uniform softmax → output = mean of sources."""
        B, T, d = 1, 4, 64
        ar = AttnResidual(num_sublayers=4, d_model=d)
        n_src = 4
        sources = [torch.randn(B, T, d) for _ in range(n_src)]
        out = ar(0, sources)
        expected = torch.stack(sources, dim=-2).mean(dim=-2)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradient_flow(self) -> None:
        B, T, d = 1, 4, 32
        ar = AttnResidual(num_sublayers=2, d_model=d)
        sources = [torch.randn(B, T, d, requires_grad=True) for _ in range(3)]
        out = ar(0, sources)
        out.sum().backward()
        for src in sources:
            assert src.grad is not None

    def test_queries_receive_gradient(self) -> None:
        B, T, d = 1, 4, 32
        ar = AttnResidual(num_sublayers=4, d_model=d)
        sources = [torch.randn(B, T, d) for _ in range(3)]
        out = ar(1, sources)
        out.sum().backward()
        assert ar.queries.grad is not None
        # Only the used query (index 1) should have a non-zero gradient
        assert ar.queries.grad[1].abs().sum() > 0

    def test_num_queries(self) -> None:
        """Should have num_sublayers + 1 queries (extra for final aggregation)."""
        ar = AttnResidual(num_sublayers=6, d_model=32)
        assert ar.queries.shape == (7, 32)

    def test_forward_stacked_matches_list_path(self) -> None:
        """Stacked fixed-shape path should match list path on valid sources."""
        B, T, d = 2, 8, 64
        ar = AttnResidual(num_sublayers=4, d_model=d)
        sources = [torch.randn(B, T, d) for _ in range(3)]

        # Provide extra padded slots with arbitrary values; they must be masked out.
        padded = torch.randn(B, T, 5, d)
        for i, src in enumerate(sources):
            padded[:, :, i, :] = src

        out_list = ar(0, sources)
        out_stacked = ar.forward_stacked(0, padded, valid_sources=len(sources))
        assert torch.allclose(out_list, out_stacked, atol=1e-5)

    def test_list_path_does_not_stack_source_tensors(self, monkeypatch) -> None:
        """List path should avoid allocating a large (B,T,n_src,d) tensor."""
        B, T, d = 2, 8, 64
        ar = AttnResidual(num_sublayers=4, d_model=d)
        sources = [torch.randn(B, T, d) for _ in range(3)]
        original_stack = torch.stack

        def guarded_stack(
            tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
            dim: int = 0,
            *,
            out: torch.Tensor | None = None,
        ) -> torch.Tensor:
            tensor_list = list(tensors)
            if tensor_list and isinstance(tensor_list[0], torch.Tensor):
                assert tensor_list[0].ndim != 3, "source tensors should not be stacked"
            if out is None:
                return original_stack(tensor_list, dim=dim)
            return original_stack(tensor_list, dim=dim, out=out)

        monkeypatch.setattr(torch, "stack", guarded_stack)
        out = ar(0, sources)
        assert out.shape == (B, T, d)


# ---------------------------------------------------------------------------
# TransformerBlock sublayer methods
# ---------------------------------------------------------------------------


class TestTransformerBlockSublayers:
    def _block(self, d: int = 64, h: int = 4) -> TransformerBlock:
        return TransformerBlock(d_model=d, num_heads=h, attention_backend="standard")

    def test_apply_attn_only_shape(self) -> None:
        block = self._block()
        x = torch.randn(2, 8, 64)
        out = block.apply_attn_only(x)
        assert out.shape == x.shape

    def test_apply_ffn_only_shape(self) -> None:
        block = self._block()
        x = torch.randn(2, 8, 64)
        out = block.apply_ffn_only(x)
        assert out.shape == x.shape

    def test_attn_only_equals_residual_delta(self) -> None:
        """apply_attn_only should equal forward_attn - x (just the attn delta)."""
        block = self._block()
        block.eval()
        x = torch.randn(1, 4, 64)
        delta = block.apply_attn_only(x)
        with_residual = block._apply_attention_residual(x)
        assert torch.allclose(with_residual, x + delta, atol=1e-6)

    def test_ffn_only_equals_residual_delta(self) -> None:
        block = self._block()
        block.eval()
        x = torch.randn(1, 4, 64)
        delta = block.apply_ffn_only(x)
        with_residual = block._apply_feedforward_residual(x)
        assert torch.allclose(with_residual, x + delta, atol=1e-6)

    def test_attn_only_gradient(self) -> None:
        block = self._block()
        x = torch.randn(1, 4, 64, requires_grad=True)
        block.apply_attn_only(x).sum().backward()
        assert x.grad is not None

    def test_ffn_only_gradient(self) -> None:
        block = self._block()
        x = torch.randn(1, 4, 64, requires_grad=True)
        block.apply_ffn_only(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# LearningModel AttnRes forward passes
# ---------------------------------------------------------------------------


def _make_model(
    res_type: str,
    num_layers: int = 4,
    attn_res_num_blocks: int = 2,
    d_model: int = 64,
    num_heads: int = 4,
) -> LearningModel:
    return LearningModel(
        vocab_size=256,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=32,
        attention_backend="standard",
        pos_type="learned",
        res_type=res_type,
        attn_res_num_blocks=attn_res_num_blocks,
    )


class TestLearningModelAttnRes:
    @pytest.mark.parametrize("res_type", ["standard", "full_attn", "block_attn"])
    def test_output_shape(self, res_type: str) -> None:
        model = _make_model(res_type)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    @pytest.mark.parametrize("res_type", ["full_attn", "block_attn"])
    def test_gradient_flow(self, res_type: str) -> None:
        model = _make_model(res_type)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        logits.sum().backward()
        # All parameters should have gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    @pytest.mark.parametrize("res_type", ["full_attn", "block_attn"])
    def test_attn_res_module_exists(self, res_type: str) -> None:
        model = _make_model(res_type)
        assert model.attn_res is not None

    def test_standard_has_no_attn_res(self) -> None:
        model = _make_model("standard")
        assert model.attn_res is None

    @pytest.mark.parametrize("res_type", ["full_attn", "block_attn"])
    def test_zero_init_queries(self, res_type: str) -> None:
        """All query vectors must start at zero per the paper's requirement."""
        model = _make_model(res_type)
        assert model.attn_res is not None
        assert torch.all(model.attn_res.queries == 0.0)

    def test_full_attn_num_queries(self) -> None:
        """Full AttnRes should have 2*num_layers + 1 queries."""
        num_layers = 6
        model = _make_model("full_attn", num_layers=num_layers)
        assert model.attn_res is not None
        assert model.attn_res.queries.shape[0] == 2 * num_layers + 1

    def test_block_attn_num_blocks_divisibility(self) -> None:
        """block_attn requires num_layers % attn_res_num_blocks == 0."""
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(
                hidden_size=64,
                vocab_size=256,
                max_seq_length=32,
                num_layers=5,
                num_heads=4,
                res_type="block_attn",
                attn_res_num_blocks=3,
            )

    def test_block_attn_various_block_counts(self) -> None:
        """Block AttnRes should work for any valid block count."""
        for n_blocks in [1, 2, 4]:
            model = _make_model("block_attn", num_layers=4, attn_res_num_blocks=n_blocks)
            x = torch.randint(0, 256, (1, 8))
            out = model(x)
            assert out.shape == (1, 8, 256)

    @pytest.mark.parametrize("res_type", ["full_attn", "block_attn"])
    def test_different_outputs_than_standard(self, res_type: str) -> None:
        """AttnRes should produce different outputs than standard residuals."""
        torch.manual_seed(0)
        model_std = _make_model("standard")
        torch.manual_seed(0)
        model_ar = _make_model(res_type)

        x = torch.randint(0, 256, (1, 8))
        # Copy weights from standard to AttnRes model (blocks, embeddings, etc.)
        # but keep default zero queries → outputs will still differ due to averaging vs summing
        with torch.no_grad():
            out_std = model_std(x)
            out_ar = model_ar(x)
        assert not torch.allclose(out_std, out_ar)


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfigResType:
    def _base_kwargs(self) -> dict:
        return {
            "hidden_size": 64,
            "vocab_size": 256,
            "max_seq_length": 32,
            "num_layers": 4,
            "num_heads": 4,
        }

    def test_standard_is_default(self) -> None:
        cfg = ModelConfig(**self._base_kwargs())
        assert cfg.res_type == "standard"

    def test_full_attn_valid(self) -> None:
        cfg = ModelConfig(**self._base_kwargs(), res_type="full_attn")
        assert cfg.res_type == "full_attn"

    def test_block_attn_valid(self) -> None:
        cfg = ModelConfig(**self._base_kwargs(), res_type="block_attn", attn_res_num_blocks=2)
        assert cfg.res_type == "block_attn"

    def test_invalid_res_type(self) -> None:
        with pytest.raises(ValueError, match="res_type"):
            ModelConfig(**self._base_kwargs(), res_type="bad_value")

    def test_block_attn_invalid_num_blocks(self) -> None:
        with pytest.raises(ValueError, match="attn_res_num_blocks"):
            ModelConfig(**self._base_kwargs(), res_type="block_attn", attn_res_num_blocks=0)

    def test_from_config_full_attn(self) -> None:
        cfg = ModelConfig(**self._base_kwargs(), res_type="full_attn")
        model = LearningModel.from_config(cfg, attention_backend="standard")
        assert model.res_type == "full_attn"
        assert model.attn_res is not None

    def test_from_config_block_attn(self) -> None:
        cfg = ModelConfig(**self._base_kwargs(), res_type="block_attn", attn_res_num_blocks=2)
        model = LearningModel.from_config(cfg, attention_backend="standard")
        assert model.res_type == "block_attn"
        assert model.attn_res is not None
