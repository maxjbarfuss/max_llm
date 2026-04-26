"""Unit tests for optimizer utilities."""

from typing import cast

import pytest
import torch
import torch.nn as nn

from src.training.optimizer import (
    MuonAdamWOptimizer,
    MuonOptimizer,
    configure_muon_optimizer,
    configure_optimizer_param_groups,
    zeropower_via_newtonschulz5,
)


class SimpleModel(nn.Module):
    """Simple model with various parameter types for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, 64)
        self.linear1 = nn.Linear(64, 128, bias=True)
        self.layernorm1 = nn.LayerNorm(128)
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.layernorm2 = nn.LayerNorm(64)
        self.output = nn.Linear(64, 10, bias=True)


@pytest.fixture
def model() -> nn.Module:
    """Create a simple model for testing."""
    return SimpleModel()


def test_param_groups_structure(model: nn.Module) -> None:
    """Test that parameter groups have the correct structure."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Should return exactly 2 groups: decay and no-decay
    assert len(param_groups) == 2

    # Both groups should have required keys
    for group in param_groups:
        assert "params" in group
        assert "weight_decay" in group
        assert "lr" in group
        assert "betas" in group
        assert "eps" in group


def test_param_groups_decay_values(model: nn.Module) -> None:
    """Test that decay and no-decay groups have correct weight decay values."""
    weight_decay = 0.1
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=weight_decay,
        learning_rate=1e-3,
    )

    # First group should have weight decay
    assert param_groups[0]["weight_decay"] == weight_decay

    # Second group should have no weight decay
    assert param_groups[1]["weight_decay"] == 0.0


def test_bias_excluded_from_decay(model: nn.Module) -> None:
    """Test that bias parameters are excluded from weight decay."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    # Get all bias parameter IDs
    bias_ids = {id(p) for n, p in model.named_parameters() if n.endswith(".bias")}

    # Get parameters in decay group
    decay_param_ids = {id(p) for p in param_groups[0]["params"]}

    # Bias parameters should not be in decay group
    assert len(bias_ids & decay_param_ids) == 0

    # Bias parameters should be in no-decay group
    no_decay_param_ids = {id(p) for p in param_groups[1]["params"]}
    assert bias_ids.issubset(no_decay_param_ids)


def test_layernorm_excluded_from_decay(model: nn.Module) -> None:
    """Test that LayerNorm parameters are excluded from weight decay."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    # Get all LayerNorm parameter IDs
    layernorm_ids = {
        id(p)
        for n, p in model.named_parameters()
        if "layernorm" in n.lower() or "layer_norm" in n.lower()
    }

    # Get parameters in decay group
    decay_param_ids = {id(p) for p in param_groups[0]["params"]}

    # LayerNorm parameters should not be in decay group
    assert len(layernorm_ids & decay_param_ids) == 0

    # LayerNorm parameters should be in no-decay group
    no_decay_param_ids = {id(p) for p in param_groups[1]["params"]}
    assert layernorm_ids.issubset(no_decay_param_ids)


def test_embeddings_included_in_decay(model: nn.Module) -> None:
    """Test that embedding parameters are included in weight decay."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    # Get embedding parameter ID
    embedding_weight_id = id(cast(SimpleModel, model).embedding.weight)

    # Get parameters in decay group
    decay_param_ids = {id(p) for p in param_groups[0]["params"]}

    # Embedding weight should be in decay group (2D parameter, not bias)
    assert embedding_weight_id in decay_param_ids


def test_linear_weights_included_in_decay(model: nn.Module) -> None:
    """Test that linear layer weights are included in weight decay."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    # Get linear weight parameter IDs (exclude bias)
    linear_weight_ids = {
        id(p)
        for n, p in model.named_parameters()
        if n.endswith(".weight") and "linear" in n.lower()
    }

    # Get parameters in decay group
    decay_param_ids = {id(p) for p in param_groups[0]["params"]}

    # All linear weights should be in decay group
    assert linear_weight_ids.issubset(decay_param_ids)


def test_all_params_accounted_for(model: nn.Module) -> None:
    """Test that all model parameters are in exactly one group."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    # Get all parameter IDs from model
    all_param_ids = {id(p) for p in model.parameters() if p.requires_grad}

    # Get all parameter IDs from groups
    grouped_param_ids = {id(p) for group in param_groups for p in group["params"]}

    # Should have exact same parameters
    assert all_param_ids == grouped_param_ids


def test_no_duplicate_params(model: nn.Module) -> None:
    """Test that no parameter appears in multiple groups."""
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    decay_param_ids = {id(p) for p in param_groups[0]["params"]}
    no_decay_param_ids = {id(p) for p in param_groups[1]["params"]}

    # Groups should be disjoint
    assert len(decay_param_ids & no_decay_param_ids) == 0


def test_1d_params_excluded_from_decay() -> None:
    """Test that 1D parameters are excluded from weight decay."""

    class Model1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_2d = nn.Parameter(torch.randn(10, 10))
            self.weight_1d = nn.Parameter(torch.randn(10))
            self.scalar = nn.Parameter(torch.randn(1))

    model = Model1D()
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    decay_param_ids = {id(p) for p in param_groups[0]["params"]}
    no_decay_param_ids = {id(p) for p in param_groups[1]["params"]}

    # 2D parameter should be in decay group
    assert id(model.weight_2d) in decay_param_ids

    # 1D and scalar parameters should be in no-decay group
    assert id(model.weight_1d) in no_decay_param_ids
    assert id(model.scalar) in no_decay_param_ids


def test_hyperparameters_set_correctly(model: nn.Module) -> None:
    """Test that hyperparameters are set correctly in both groups."""
    lr = 1e-3
    betas = (0.9, 0.95)
    eps = 1e-7
    weight_decay = 0.2

    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=betas,
        eps=eps,
    )

    # Check decay group
    assert param_groups[0]["lr"] == lr
    assert param_groups[0]["betas"] == betas
    assert param_groups[0]["eps"] == eps
    assert param_groups[0]["weight_decay"] == weight_decay

    # Check no-decay group
    assert param_groups[1]["lr"] == lr
    assert param_groups[1]["betas"] == betas
    assert param_groups[1]["eps"] == eps
    assert param_groups[1]["weight_decay"] == 0.0


def test_frozen_params_excluded() -> None:
    """Test that frozen parameters (requires_grad=False) are excluded."""

    class ModelWithFrozen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = nn.Linear(10, 10)
            self.trainable = nn.Linear(10, 10)
            # Freeze first layer
            for param in self.frozen.parameters():
                param.requires_grad = False

    model = ModelWithFrozen()
    param_groups = configure_optimizer_param_groups(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
    )

    # Get all grouped parameter IDs
    grouped_param_ids = {id(p) for group in param_groups for p in group["params"]}

    # Frozen parameters should not be in any group
    frozen_param_ids = {id(p) for p in model.frozen.parameters()}
    assert len(frozen_param_ids & grouped_param_ids) == 0

    # Trainable parameters should all be in groups
    trainable_param_ids = {id(p) for p in model.trainable.parameters()}
    assert trainable_param_ids.issubset(grouped_param_ids)


class TestZeropowerNewtonSchulz5:
    """Tests for Newton-Schulz orthogonalization helper."""

    def test_preserves_shape_dtype_and_finiteness(self) -> None:
        gradient = torch.randn(8, 4, dtype=torch.bfloat16)

        orthogonalized = zeropower_via_newtonschulz5(gradient, steps=4)

        assert orthogonalized.shape == gradient.shape
        assert orthogonalized.dtype == gradient.dtype
        assert torch.isfinite(orthogonalized).all()

    def test_rectangular_result_has_near_orthonormal_rows(self) -> None:
        gradient = torch.randn(4, 8, dtype=torch.float32)

        orthogonalized = zeropower_via_newtonschulz5(gradient, steps=8)
        baseline = gradient / gradient.norm()
        baseline_gram = baseline @ baseline.T
        gram = orthogonalized @ orthogonalized.T

        assert torch.linalg.norm(gram - torch.eye(4)) < torch.linalg.norm(
            baseline_gram - torch.eye(4)
        )

    def test_batched_result_matches_per_matrix_result(self) -> None:
        generator = torch.Generator().manual_seed(0)
        gradients = torch.randn(3, 4, 8, dtype=torch.float32, generator=generator)

        batched = zeropower_via_newtonschulz5(gradients, steps=5)
        per_matrix = torch.stack(
            [zeropower_via_newtonschulz5(gradient, steps=5) for gradient in gradients]
        )

        torch.testing.assert_close(batched, per_matrix, atol=1e-5, rtol=1e-5)


class TestMuonOptimizer:
    """Tests for the Muon optimizer step behavior."""

    def test_updates_parameter_and_creates_fp32_momentum_state(self) -> None:
        parameter = nn.Parameter(torch.randn(4, 4, dtype=torch.float16))
        optimizer = MuonOptimizer([parameter], lr=0.1, momentum=0.9, ns_steps=4)
        parameter.grad = torch.randn_like(parameter)
        before = parameter.detach().clone()

        optimizer.step()

        assert not torch.allclose(parameter, before)
        assert optimizer.state[parameter]["momentum_buffer"].dtype == torch.float32

    def test_zero_grad_clears_gradient(self) -> None:
        parameter = nn.Parameter(torch.randn(4, 4))
        optimizer = MuonOptimizer([parameter], lr=0.1)
        parameter.grad = torch.randn_like(parameter)

        optimizer.zero_grad()

        assert parameter.grad is None


class ToyMuonModel(nn.Module):
    """Small module exposing parameter names used by Muon routing."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(32, 8)
        self.position_embedding = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8, bias=True)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.lm_head.weight = self.token_embedding.weight


class TestConfigureMuonOptimizer:
    """Tests for Muon/AdamW parameter routing."""

    def test_routes_only_2d_non_embedding_weights_to_muon(self) -> None:
        model = ToyMuonModel()

        optimizer = configure_muon_optimizer(
            model,
            muon_lr=0.02,
            adamw_lr=1e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        muon_params = {
            id(param) for group in optimizer.muon.param_groups for param in group["params"]
        }
        adamw_decay_params = {
            id(param)
            for group in optimizer.adamw.param_groups
            if group["weight_decay"] > 0
            for param in group["params"]
        }
        adamw_no_decay_params = {
            id(param)
            for group in optimizer.adamw.param_groups
            if group["weight_decay"] == 0
            for param in group["params"]
        }

        assert id(model.proj.weight) in muon_params
        assert id(model.token_embedding.weight) in adamw_decay_params
        assert id(model.position_embedding.weight) in adamw_decay_params
        assert id(model.proj.bias) in adamw_no_decay_params
        assert id(model.norm.weight) in adamw_no_decay_params
        assert id(model.norm.bias) in adamw_no_decay_params
        assert id(model.token_embedding.weight) not in muon_params
        assert id(model.token_embedding.weight) not in (
            {id(param) for group in optimizer.muon.param_groups for param in group["params"]}
            & {id(param) for group in optimizer.adamw.param_groups for param in group["params"]}
        )

    def test_marks_param_groups_with_is_muon_flag(self) -> None:
        model = ToyMuonModel()

        optimizer = configure_muon_optimizer(
            model,
            muon_lr=0.02,
            adamw_lr=1e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        assert all(group.get("is_muon") is True for group in optimizer.muon.param_groups)
        assert all(group.get("is_muon") is False for group in optimizer.adamw.param_groups)


class TestMuonAdamWOptimizer:
    """Tests for the composite Muon + AdamW optimizer wrapper."""

    def test_step_and_state_dict_delegate_to_inner_optimizers(self) -> None:
        muon_param = nn.Parameter(torch.randn(4, 4))
        adamw_param = nn.Parameter(torch.randn(4))
        muon = MuonOptimizer([muon_param], lr=0.02)
        adamw = torch.optim.AdamW([adamw_param], lr=1e-3)
        optimizer = MuonAdamWOptimizer(muon=muon, adamw=adamw)

        muon_param.grad = torch.randn_like(muon_param)
        adamw_param.grad = torch.randn_like(adamw_param)
        before_muon = muon_param.detach().clone()
        before_adamw = adamw_param.detach().clone()

        optimizer.step()
        state_dict = optimizer.state_dict()

        assert not torch.allclose(muon_param, before_muon)
        assert not torch.allclose(adamw_param, before_adamw)
        assert set(state_dict) == {"muon", "adamw"}

    def test_load_state_dict_restores_both_inner_optimizers(self) -> None:
        muon_param = nn.Parameter(torch.randn(4, 4))
        adamw_param = nn.Parameter(torch.randn(4))
        optimizer = MuonAdamWOptimizer(
            muon=MuonOptimizer([muon_param], lr=0.02),
            adamw=torch.optim.AdamW([adamw_param], lr=1e-3),
        )
        muon_param.grad = torch.randn_like(muon_param)
        adamw_param.grad = torch.randn_like(adamw_param)
        optimizer.step()
        saved = optimizer.state_dict()

        restored = MuonAdamWOptimizer(
            muon=MuonOptimizer([muon_param], lr=0.5),
            adamw=torch.optim.AdamW([adamw_param], lr=0.5),
        )
        restored.load_state_dict(saved)

        assert restored.muon.param_groups[0]["lr"] == pytest.approx(0.02)
        assert restored.adamw.param_groups[0]["lr"] == pytest.approx(1e-3)
