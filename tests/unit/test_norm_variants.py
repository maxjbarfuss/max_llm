"""Tests for FlashNorm, DyT, and CRMSNorm normalization variants."""

import pytest
import torch

from src.models.norm import CRMSNorm, DyT, FlashNorm, make_norm

# ---------------------------------------------------------------------------
# FlashNorm
# ---------------------------------------------------------------------------


class TestFlashNorm:
    def test_output_shape(self):
        m = FlashNorm(64)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_no_parameters(self):
        m = FlashNorm(64)
        assert sum(p.numel() for p in m.parameters()) == 0

    def test_unit_rms(self):
        """Output should have RMS ≈ 1 along last dim."""
        m = FlashNorm(128)
        x = torch.randn(4, 16, 128) * 5
        y = m(x)
        rms = y.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_dtype_preserved(self):
        m = FlashNorm(32)
        x = torch.randn(2, 4, 32).to(torch.bfloat16)
        assert m(x).dtype == torch.bfloat16

    def test_scale_invariance(self):
        """FlashNorm output should be identical regardless of input scale."""
        m = FlashNorm(64)
        x = torch.randn(2, 8, 64)
        y1 = m(x)
        y2 = m(x * 10)
        assert torch.allclose(y1, y2, atol=1e-5)


# ---------------------------------------------------------------------------
# DyT
# ---------------------------------------------------------------------------


class TestDyT:
    def test_output_shape(self):
        m = DyT(64)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_parameter_count(self):
        m = DyT(64)
        params = dict(m.named_parameters())
        assert "alpha" in params
        assert "weight" in params
        assert params["alpha"].shape == ()  # scalar
        assert params["weight"].shape == (64,)

    def test_alpha_init(self):
        m = DyT(64, alpha_init=0.5)
        assert m.alpha.item() == pytest.approx(0.5)

    def test_weight_init(self):
        m = DyT(64)
        assert torch.allclose(m.weight, torch.ones(64))

    def test_bounded_output(self):
        """tanh ensures output is in (-weight_max, +weight_max)."""
        m = DyT(64)
        x = torch.randn(8, 16, 64) * 100  # very large inputs
        y = m(x)
        # weight initialised to 1 → output in (-1, 1)
        assert y.abs().max().item() < 1.0 + 1e-5

    def test_gradient_flows_alpha(self):
        m = DyT(32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        m(x).sum().backward()
        assert m.alpha.grad is not None
        assert m.weight.grad is not None

    def test_dtype_preserved(self):
        m = DyT(32)
        x = torch.randn(2, 4, 32).to(torch.bfloat16)
        assert m(x).dtype == torch.bfloat16

    def test_custom_alpha_init(self):
        m = DyT(64, alpha_init=0.1)
        assert m.alpha.item() == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# CRMSNorm
# ---------------------------------------------------------------------------


class TestCRMSNorm:
    def test_output_shape(self):
        m = CRMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_parameter_count(self):
        m = CRMSNorm(64)
        params = list(m.parameters())
        assert len(params) == 1  # just weight
        assert params[0].shape == (64,)

    def test_weight_init(self):
        m = CRMSNorm(64)
        assert torch.allclose(m.weight, torch.ones(64))

    def test_zero_mean_output(self):
        """Centering step should drive output mean close to zero."""
        m = CRMSNorm(128)
        x = torch.randn(4, 16, 128) + 5  # large mean offset
        y = m(x)
        assert y.mean(-1).abs().max().item() < 1e-4

    def test_unit_rms_of_centered(self):
        """After centering, the RMS of the output should be ≈ 1."""
        m = CRMSNorm(128)
        # Set weight to ones so scale = 1
        x = torch.randn(4, 16, 128)
        y = m(x)
        y_c = y - y.mean(-1, keepdim=True)
        rms = y_c.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_gradient_flows(self):
        m = CRMSNorm(32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert m.weight.grad is not None

    def test_dtype_preserved(self):
        m = CRMSNorm(32)
        x = torch.randn(2, 4, 32).to(torch.bfloat16)
        assert m(x).dtype == torch.bfloat16

    def test_differs_from_rms_norm(self):
        """CRMSNorm and RMSNorm should produce different outputs on mean-shifted input."""
        from src.models.norm import RMSNorm

        crms = CRMSNorm(64)
        rms = RMSNorm(64)
        # Copy weights so scale is identical
        with torch.no_grad():
            rms.weight.copy_(crms.weight)
        x = torch.randn(4, 8, 64) + 3  # non-zero mean
        assert not torch.allclose(crms(x), rms(x), atol=1e-3)


# ---------------------------------------------------------------------------
# make_norm factory
# ---------------------------------------------------------------------------


class TestMakeNorm:
    @pytest.mark.parametrize("norm_type", ["layer", "rms", "flash", "dyt", "crms"])
    def test_all_types_instantiate(self, norm_type):
        m = make_norm(norm_type, 64)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_flash_returns_flash_norm(self):
        assert isinstance(make_norm("flash", 64), FlashNorm)

    def test_dyt_returns_dyt(self):
        assert isinstance(make_norm("dyt", 64), DyT)

    def test_crms_returns_crms_norm(self):
        assert isinstance(make_norm("crms", 64), CRMSNorm)
