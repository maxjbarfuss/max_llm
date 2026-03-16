"""Tests for RMSNorm module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.norm.rms_norm import RMSNorm


class TestRMSNormShape:
    def test_output_shape_3d(self) -> None:
        """Output shape must match input shape."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == (2, 8, 64)

    def test_output_shape_2d(self) -> None:
        """Works on 2-D tensors (batch, d_model)."""
        norm = RMSNorm(d_model=32)
        x = torch.randn(4, 32)
        assert norm(x).shape == (4, 32)

    def test_output_shape_4d(self) -> None:
        """Works on arbitrary leading dimensions."""
        norm = RMSNorm(d_model=16)
        x = torch.randn(2, 3, 5, 16)
        assert norm(x).shape == (2, 3, 5, 16)

    def test_output_dtype_float32(self) -> None:
        norm = RMSNorm(d_model=64)
        x = torch.randn(1, 8, 64)
        assert norm(x).dtype == torch.float32

    def test_wrong_last_dim_raises(self) -> None:
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 8, 32)
        with pytest.raises(AssertionError):
            norm(x)


class TestRMSNormNumerics:
    def test_unit_weight_normalizes_rms(self) -> None:
        """With weight=1, each output vector should have RMS ≈ 1."""
        norm = RMSNorm(d_model=128)
        # Force weight to 1 (it already is, but be explicit)
        nn.init.ones_(norm.weight)
        x = torch.randn(4, 16, 128) * 5.0  # large scale to stress-test
        y = norm(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_no_mean_subtraction(self) -> None:
        """RMSNorm should NOT center the mean (unlike LayerNorm)."""
        norm = RMSNorm(d_model=64)
        nn.init.ones_(norm.weight)
        # Constant positive input: mean == input, so LayerNorm would produce zeros
        x = torch.full((1, 1, 64), 3.0)
        y = norm(x)
        # All elements should equal 1.0 (since mean(x²)=9, RMS=3, 3/3*1=1)
        assert torch.allclose(y, torch.ones_like(y), atol=1e-5)

    def test_weight_scaling(self) -> None:
        """Output should scale linearly with weight parameter."""
        norm = RMSNorm(d_model=32)
        nn.init.constant_(norm.weight, 2.0)
        x = torch.randn(2, 4, 32)

        norm1 = RMSNorm(d_model=32)
        nn.init.ones_(norm1.weight)

        # y2 should be 2 * y1 when weight doubles
        y1 = norm1(x)
        y2 = norm(x)
        assert torch.allclose(y2, 2.0 * y1, atol=1e-5)

    def test_scale_invariance(self) -> None:
        """Scaling input by a positive constant should not change output (given weight=1)."""
        norm = RMSNorm(d_model=64)
        nn.init.ones_(norm.weight)
        x = torch.randn(2, 8, 64)
        y1 = norm(x)
        y2 = norm(x * 5.0)
        assert torch.allclose(y1, y2, atol=1e-5)

    def test_near_zero_input_stable(self) -> None:
        """Near-zero input should not cause NaN/Inf (eps guard)."""
        norm = RMSNorm(d_model=32, eps=1e-6)
        x = torch.zeros(1, 4, 32)
        y = norm(x)
        assert torch.isfinite(y).all()


class TestRMSNormParameters:
    def test_weight_initialized_to_ones(self) -> None:
        norm = RMSNorm(d_model=64)
        assert torch.allclose(norm.weight, torch.ones(64))

    def test_weight_is_learnable(self) -> None:
        norm = RMSNorm(d_model=32)
        assert norm.weight.requires_grad

    def test_no_bias_parameter(self) -> None:
        norm = RMSNorm(d_model=64)
        assert not hasattr(norm, "bias") or not isinstance(
            getattr(norm, "bias", None), nn.Parameter
        )

    def test_param_count(self) -> None:
        """Should have exactly d_model trainable parameters (the gain vector)."""
        norm = RMSNorm(d_model=128)
        params = list(norm.parameters())
        assert len(params) == 1
        assert params[0].shape == (128,)


class TestRMSNormGradient:
    def test_gradient_flows(self) -> None:
        norm = RMSNorm(d_model=32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        loss = norm(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_weight_gradient(self) -> None:
        norm = RMSNorm(d_model=32)
        x = torch.randn(1, 4, 32)
        loss = norm(x).sum()
        loss.backward()
        assert norm.weight.grad is not None


class TestRMSNormVsLayerNorm:
    def test_differs_from_layernorm(self) -> None:
        """RMSNorm and LayerNorm should produce different outputs on typical input."""
        d = 64
        rms = RMSNorm(d_model=d)
        ln = nn.LayerNorm(d)
        # Align biases: LayerNorm has bias=0 by default, gain=1 by default
        nn.init.ones_(rms.weight)
        nn.init.ones_(ln.weight)
        nn.init.zeros_(ln.bias)

        torch.manual_seed(0)
        x = torch.randn(2, 8, d)
        y_rms = rms(x)
        y_ln = ln(x)
        # They should differ (RMSNorm skips mean centering)
        assert not torch.allclose(y_rms, y_ln, atol=1e-4)


class TestRMSNormInTransformerBlock:
    def test_transformer_block_uses_rms_norm(self) -> None:
        """TransformerBlock with norm_type='rms' should use RMSNorm layers."""
        from src.models.transformer.transformer_block import TransformerBlock

        block = TransformerBlock(
            d_model=64, num_heads=4, attention_backend="standard", norm_type="rms"
        )
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)

    def test_transformer_block_default_uses_layernorm(self) -> None:
        """TransformerBlock default norm_type='layer' should use LayerNorm."""
        from src.models.transformer.transformer_block import TransformerBlock

        block = TransformerBlock(d_model=64, num_heads=4, attention_backend="standard")
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)

    def test_transformer_block_rms_forward(self) -> None:
        """TransformerBlock with RMSNorm should produce correct output shape."""
        from src.models.transformer.transformer_block import TransformerBlock

        block = TransformerBlock(
            d_model=64, num_heads=4, attention_backend="standard", norm_type="rms"
        )
        x = torch.randn(2, 8, 64)
        assert block(x).shape == (2, 8, 64)

    def test_transformer_block_invalid_norm_type(self) -> None:
        from src.models.transformer.transformer_block import TransformerBlock

        with pytest.raises(AssertionError):
            TransformerBlock(d_model=64, num_heads=4, norm_type="unknown")


class TestRMSNormInLearningModel:
    def test_learning_model_uses_rms_norm(self) -> None:
        """LearningModel with norm_type='rms' should use RMSNorm throughout."""
        from src.models.learning_model.learning_model import LearningModel

        model = LearningModel(
            vocab_size=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            attention_backend="standard",
            norm_type="rms",
        )
        assert isinstance(model.final_norm, RMSNorm)
        for block in model.blocks:
            assert isinstance(block.norm1, RMSNorm)  # type: ignore[union-attr]
            assert isinstance(block.norm2, RMSNorm)  # type: ignore[union-attr]

    def test_learning_model_default_uses_layernorm(self) -> None:
        from src.models.learning_model.learning_model import LearningModel

        model = LearningModel(
            vocab_size=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            attention_backend="standard",
        )
        assert isinstance(model.final_norm, nn.LayerNorm)


class TestModelConfigNormType:
    def test_default_norm_type_is_layer(self) -> None:
        from src.config.model import ModelConfig

        cfg = ModelConfig(hidden_size=64, vocab_size=128, max_seq_length=64)
        assert cfg.norm_type == "layer"

    def test_rms_norm_type_accepted(self) -> None:
        from src.config.model import ModelConfig

        cfg = ModelConfig(hidden_size=64, vocab_size=128, max_seq_length=64, norm_type="rms")
        assert cfg.norm_type == "rms"

    def test_invalid_norm_type_rejected(self) -> None:
        from src.config.model import ModelConfig

        with pytest.raises(ValueError, match="norm_type"):
            ModelConfig(hidden_size=64, vocab_size=128, max_seq_length=64, norm_type="batchnorm")
