"""Tests for SwiGLU, ReLU², and xIELU feed-forward network variants."""

import pytest
import torch

from src.models.feedforward import make_ffn, swiglu_intermediate_size
from src.models.feedforward.relu2_ffn import ReLU2FFN
from src.models.feedforward.swiglu import SwiGLU
from src.models.feedforward.xielu_ffn import xIELU, xIELUFFN

D = 128
B, T = 2, 16


# ---------------------------------------------------------------------------
# swiglu_intermediate_size helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_model", [256, 512, 1024, 2048])
def test_swiglu_intermediate_size_multiple_of_256(d_model: int) -> None:
    assert swiglu_intermediate_size(d_model) % 256 == 0


def test_swiglu_intermediate_size_1024() -> None:
    # int(1024 * 8/3) = 2730 → rounds up to 2816
    assert swiglu_intermediate_size(1024) == 2816


# ---------------------------------------------------------------------------
# make_ffn factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ffn_type", ["gelu", "swiglu", "relu2", "xielu"])
def test_make_ffn_forward_shape(ffn_type: str) -> None:
    ffn = make_ffn(ffn_type, D, D * 4)
    x = torch.randn(B, T, D)
    assert ffn(x).shape == (B, T, D)


def test_make_ffn_invalid_type() -> None:
    with pytest.raises(AssertionError):
        make_ffn("relu3", D, D * 4)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


class TestSwiGLU:
    def test_output_shape(self) -> None:
        ffn = SwiGLU(D, swiglu_intermediate_size(D))
        assert ffn(torch.randn(B, T, D)).shape == (B, T, D)

    def test_no_bias(self) -> None:
        ffn = SwiGLU(D, D * 3)
        assert ffn.gate_proj.bias is None
        assert ffn.up_proj.bias is None
        assert ffn.down_proj.bias is None

    def test_three_projections(self) -> None:
        ffn = SwiGLU(D, D * 3)
        assert hasattr(ffn, "gate_proj")
        assert hasattr(ffn, "up_proj")
        assert hasattr(ffn, "down_proj")

    def test_parameter_count(self) -> None:
        hidden = D * 3
        ffn = SwiGLU(D, hidden)
        # gate + up + down, no bias
        expected = D * hidden + D * hidden + hidden * D
        assert sum(p.numel() for p in ffn.parameters()) == expected

    def test_gradient_flows(self) -> None:
        ffn = SwiGLU(D, D * 3)
        x = torch.randn(B, T, D, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None

    def test_bad_input_ndim_raises(self) -> None:
        ffn = SwiGLU(D, D * 3)
        with pytest.raises(AssertionError):
            ffn(torch.randn(B, D))

    def test_bad_input_dim_raises(self) -> None:
        ffn = SwiGLU(D, D * 3)
        with pytest.raises(AssertionError):
            ffn(torch.randn(B, T, D + 1))

    def test_gating_differs_from_ungated(self) -> None:
        """Gate and up projections should multiply — output differs from just up."""
        torch.manual_seed(0)
        ffn = SwiGLU(D, D * 3)
        x = torch.randn(B, T, D)
        # If gating were ignored (gate=1), output = down(up(x)); verify actual differs
        out = ffn(x)
        fake = ffn.down_proj(ffn.up_proj(x))
        assert not torch.allclose(out, fake)


# ---------------------------------------------------------------------------
# ReLU²
# ---------------------------------------------------------------------------


class TestReLU2FFN:
    def test_output_shape(self) -> None:
        ffn = ReLU2FFN(D, D * 4)
        assert ffn(torch.randn(B, T, D)).shape == (B, T, D)

    def test_non_negative_pre_output(self) -> None:
        """After first linear + ReLU², activations are always non-negative."""
        ffn = ReLU2FFN(D, D * 4)
        x = torch.randn(B, T, D)
        hidden = ffn.linear1(x).relu().pow(2)
        assert (hidden >= 0).all()

    def test_sparsity(self) -> None:
        """ReLU² should produce ~50% zeros on random input (dead neurons)."""
        ffn = ReLU2FFN(D, D * 4)
        x = torch.randn(B * 4, T * 4, D)
        hidden = ffn.linear1(x).relu().pow(2)
        sparsity = (hidden == 0).float().mean().item()
        assert sparsity > 0.3, f"Expected >30% sparsity, got {sparsity:.2%}"

    def test_gradient_flows(self) -> None:
        ffn = ReLU2FFN(D, D * 4)
        x = torch.randn(B, T, D, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None

    def test_parameter_count(self) -> None:
        hidden = D * 4
        ffn = ReLU2FFN(D, hidden)
        # linear1 weight+bias + linear2 weight+bias
        expected = D * hidden + hidden + hidden * D + D
        assert sum(p.numel() for p in ffn.parameters()) == expected


# ---------------------------------------------------------------------------
# xIELU activation
# ---------------------------------------------------------------------------


class TestxIELU:
    def test_output_shape(self) -> None:
        act = xIELU()
        x = torch.randn(B, T, D)
        assert act(x).shape == x.shape

    def test_two_trainable_params(self) -> None:
        act = xIELU()
        assert sum(p.numel() for p in act.parameters()) == 2

    def test_continuity_at_zero(self) -> None:
        """Function should be continuous at x=0 (both branches agree)."""
        act = xIELU()
        eps = 1e-5
        pos = act(torch.tensor([eps]))
        neg = act(torch.tensor([-eps]))
        assert abs(pos.item() - neg.item()) < 1e-3

    def test_positive_branch_is_quadratic(self) -> None:
        """For large positive x, dominant term is α_p * x²."""
        act = xIELU()
        x_large = torch.tensor([10.0])
        x_small = torch.tensor([5.0])
        # Ratio of outputs should be ~4 (quadratic scaling) for large positive x
        ratio = act(x_large).item() / act(x_small).item()
        assert 3.5 < ratio < 4.5, f"Expected ~4x ratio (quadratic), got {ratio:.2f}"

    def test_negative_branch_bounded(self) -> None:
        """Very negative inputs should not explode (exp is clamped)."""
        act = xIELU()
        x = torch.tensor([-100.0, -1000.0])
        out = act(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self) -> None:
        act = xIELU()
        x = torch.randn(B, T, D, requires_grad=True)
        act(x).sum().backward()
        assert x.grad is not None

    def test_params_are_learnable(self) -> None:
        act = xIELU()
        x = torch.randn(4, 4, requires_grad=True)
        act(x).sum().backward()
        for p in act.parameters():
            assert p.grad is not None

    def test_init_values(self) -> None:
        """At init, softplus(alpha_p) ≈ 0.8, β + softplus(alpha_n) ≈ 0.8."""
        import torch.nn.functional as F

        act = xIELU()
        alpha_p = F.softplus(act.alpha_p).item()
        alpha_n = (act.beta + F.softplus(act.alpha_n)).item()
        assert abs(alpha_p - 0.8) < 0.01
        assert abs(alpha_n - 0.8) < 0.01

    def test_matches_eager_reference(self) -> None:
        """Custom autograd Function output and grads match an eager reference."""
        import torch.nn.functional as F

        torch.manual_seed(0)
        act = xIELU()
        x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
        # Cast params to f64 for the comparison.
        act.alpha_p.data = act.alpha_p.data.to(torch.float64)
        act.alpha_n.data = act.alpha_n.data.to(torch.float64)

        # Reference (eager) implementation.
        def ref(xv: torch.Tensor, ap: torch.Tensor, an: torch.Tensor) -> torch.Tensor:
            alpha_p = F.softplus(ap)
            alpha_n = act.beta + F.softplus(an)
            pos = alpha_p * xv * xv + act.beta * xv
            neg_x = torch.clamp_max(xv, act.eps)
            neg = alpha_n * torch.expm1(neg_x) - alpha_n * xv + act.beta * xv
            return torch.where(xv > 0, pos, neg)

        # Forward equivalence.
        x_ref = x.detach().clone().requires_grad_(True)
        ap_ref = act.alpha_p.detach().clone().requires_grad_(True)
        an_ref = act.alpha_n.detach().clone().requires_grad_(True)
        y_ref = ref(x_ref, ap_ref, an_ref)
        y = act(x)
        assert torch.allclose(y, y_ref, atol=1e-10, rtol=1e-10)

        # Backward equivalence (use a non-trivial upstream grad).
        g = torch.randn_like(y)
        y.backward(g)
        y_ref.backward(g)
        assert torch.allclose(x.grad, x_ref.grad, atol=1e-10, rtol=1e-10)
        assert torch.allclose(act.alpha_p.grad, ap_ref.grad, atol=1e-10, rtol=1e-10)
        assert torch.allclose(act.alpha_n.grad, an_ref.grad, atol=1e-10, rtol=1e-10)

    def test_gradcheck(self) -> None:
        """torch.autograd.gradcheck against numerical gradients."""
        torch.manual_seed(0)
        act = xIELU()
        act.alpha_p.data = act.alpha_p.data.to(torch.float64)
        act.alpha_n.data = act.alpha_n.data.to(torch.float64)
        x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        # gradcheck needs (Tensor, ...) inputs to a function.
        from src.models.feedforward.xielu_ffn import _XIELUFunction

        assert torch.autograd.gradcheck(
            lambda xv, ap, an: _XIELUFunction.apply(xv, ap, an, act.beta, act.eps),
            (x, act.alpha_p, act.alpha_n),
            eps=1e-6,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# xIELUFFN
# ---------------------------------------------------------------------------


class TestxIELUFFN:
    def test_output_shape(self) -> None:
        ffn = xIELUFFN(D, D * 4)
        assert ffn(torch.randn(B, T, D)).shape == (B, T, D)

    def test_has_activation_params(self) -> None:
        """xIELU adds 2 params on top of the linear layers."""
        ffn = xIELUFFN(D, D * 4)
        hidden = D * 4
        linear_params = D * hidden + hidden + hidden * D + D  # weight+bias x2
        total = sum(p.numel() for p in ffn.parameters())
        assert total == linear_params + 2  # +2 for xIELU scalars

    def test_gradient_flows(self) -> None:
        ffn = xIELUFFN(D, D * 4)
        x = torch.randn(B, T, D, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Integration with LearningModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ffn_type", ["gelu", "swiglu", "relu2", "xielu"])
def test_learning_model_ffn_variants(ffn_type: str) -> None:
    from src.models.learning_model.learning_model import LearningModel

    intermediate = swiglu_intermediate_size(D) if ffn_type == "swiglu" else D * 4
    model = LearningModel(
        vocab_size=256,
        d_model=D,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        attention_backend="standard",
        ffn_type=ffn_type,
        intermediate_size=intermediate,
    )
    x = torch.randint(0, 256, (B, 16))
    assert model(x).shape == (B, 16, 256)


def test_learning_model_from_config_swiglu() -> None:
    from src.config.model import ModelConfig
    from src.models.feedforward.swiglu import swiglu_intermediate_size
    from src.models.learning_model.learning_model import LearningModel

    config = ModelConfig(
        hidden_size=D,
        vocab_size=256,
        max_seq_length=32,
        num_layers=2,
        num_heads=4,
        norm_type="rms",
        ffn_type="swiglu",
        intermediate_size=swiglu_intermediate_size(D),
    )
    model = LearningModel.from_config(config, attention_backend="standard")
    x = torch.randint(0, 256, (1, 16))
    assert model(x).shape == (1, 16, 256)
