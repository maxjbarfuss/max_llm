"""Unit tests for the Phase 2/3 training loop."""

import csv
import math

import numpy as np
import pytest
import torch

import src.training.loop as loop
from src.config.model import ModelConfig
from src.models.learning_model import LearningModel
from src.training.loop import (
    _restore_filtered_gradients,
    apply_generalization_gradient_filter,
    compute_chunked_lm_loss,
    compute_loss_with_smoothing,
    optimizer_step,
    train,
    train_step,
)
from src.training.train import create_simple_loaders


def _make_model() -> LearningModel:
    config = ModelConfig(
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        vocab_size=128,
        max_seq_length=64,
        mla_latent_dim=64,
        rope_base=10000,
        intermediate_size=None,
        num_experts=1,
        experts_per_token=1,
        moe_frequency=0,
        gru_hidden_size=None,
        dropout=0.0,
    )
    return LearningModel.from_config(config, attention_backend="standard")


def _make_batch(batch_size: int = 4, seq_len: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 128, (batch_size, seq_len))
    y = torch.randint(0, 128, (batch_size, seq_len))
    return x, y


def _make_optimizer(model: LearningModel) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


class TestTrainStep:
    """train_step correctness."""

    def test_returns_finite_scalar(self):
        """train_step returns a finite Python float."""
        model = _make_model()
        x, y = _make_batch()
        loss = train_step(model, x, y, _make_optimizer(model))
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_loss_is_positive(self):
        """Cross-entropy loss is positive for random predictions."""
        model = _make_model()
        x, y = _make_batch()
        loss = train_step(model, x, y, _make_optimizer(model))
        assert loss > 0.0

    def test_parameters_change_after_step(self):
        """Model parameters are updated after a training step + optimizer step."""
        model = _make_model()
        optimizer = _make_optimizer(model)
        params_before = [p.clone() for p in model.parameters()]
        x, y = _make_batch()
        train_step(model, x, y, optimizer)
        optimizer_step(optimizer, model=model)  # Explicit optimizer step
        params_after = list(model.parameters())
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after, strict=True)
        )

    def test_raises_on_non_finite_loss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """train_step should fail fast if loss becomes non-finite."""
        model = _make_model()
        x, y = _make_batch()

        def _nan_loss(*args, **kwargs):
            return torch.tensor(float("nan"))

        monkeypatch.setattr(loop, "compute_loss_with_smoothing", _nan_loss)
        with pytest.raises(FloatingPointError, match="Non-finite loss"):
            train_step(model, x, y, _make_optimizer(model))


class TestTrain:
    """train() loop behaviour."""

    def test_returns_loss_list_of_correct_length(self):
        """train() returns a dict with losses and perplexities of length max_steps."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = train(model, loader, _make_optimizer(model), max_steps=5, log_interval=0)
        assert len(metrics["losses"]) == 5
        assert len(metrics["perplexities"]) == 5

    def test_all_losses_finite(self):
        """All returned losses are finite."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = train(model, loader, _make_optimizer(model), max_steps=10, log_interval=0)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in metrics["losses"])
        assert all(torch.isfinite(torch.tensor(ppl)) for ppl in metrics["perplexities"])

    def test_loss_decreases_on_repeated_batch(self):
        """Loss falls when training repeatedly on the same tiny batch."""
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        model = _make_model()
        # Single repeated batch — model must memorise it
        x = torch.randint(0, 128, (4, 8))
        y = torch.randint(0, 128, (4, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = train(
            model,
            loader,
            torch.optim.Adam(model.parameters(), lr=1e-2),
            max_steps=100,
            log_interval=0,
        )
        losses = metrics["losses"]
        assert (
            losses[-1] < losses[0]
        ), f"Expected loss to decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"

    def test_cycles_loader_when_steps_exceed_dataset(self):
        """train() cycles through the loader until max_steps is reached."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        # Only 2 batches in the loader, but we request 7 steps
        x = torch.randint(0, 128, (8, 8))
        y = torch.randint(0, 128, (8, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = train(model, loader, _make_optimizer(model), max_steps=7, log_interval=0)
        assert len(metrics["losses"]) == 7

    def test_train_sets_model_training_mode(self):
        """train() sets model.training = True even if model starts in eval mode."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        model.eval()  # deliberately put model in eval mode
        assert not model.training

        x = torch.randint(0, 128, (8, 8))
        y = torch.randint(0, 128, (8, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        train(model, loader, _make_optimizer(model), max_steps=2, log_interval=0)
        assert model.training

    def test_max_steps_one(self):
        """train() with max_steps=1 returns exactly one loss."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (8, 8))
        y = torch.randint(0, 128, (8, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = train(model, loader, _make_optimizer(model), max_steps=1, log_interval=0)
        assert len(metrics["losses"]) == 1
        assert isinstance(metrics["losses"][0], float)
        assert metrics["losses"][0] > 0.0


class TestCreateSimpleLoaders:
    """create_simple_loaders correctness."""

    def _tokens(self, n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.long)

    def test_split_sizes_sum_to_total_samples(self):
        """Train + val sample counts sum to total non-overlapping sequences."""
        seq_len = 8
        tokens = self._tokens(100)
        total_samples = len(tokens) // (seq_len + 1)  # 11
        train_loader, val_loader, _ = create_simple_loaders(tokens, seq_len, batch_size=4)
        assert len(train_loader.dataset) + len(val_loader.dataset) == total_samples  # type: ignore[arg-type]

    def test_validation_split_respected(self):
        """val_loader receives roughly validation_split fraction of samples."""
        seq_len = 4
        tokens = self._tokens(100)
        total_samples = len(tokens) // (seq_len + 1)  # 20
        _, val_loader, _ = create_simple_loaders(
            tokens, seq_len, batch_size=2, validation_split=0.2
        )
        expected_val = int(total_samples * 0.2)
        assert len(val_loader.dataset) == expected_val  # type: ignore[arg-type]

    def test_input_target_offset_by_one(self):
        """Each input token sequence is shifted by one to produce the target."""
        seq_len = 4
        tokens = torch.arange(20, dtype=torch.long)
        train_loader, _, _ = create_simple_loaders(
            tokens, seq_len, batch_size=20, validation_split=0.0
        )
        for x, y in train_loader:
            # x[i, t+1] should equal y[i, t] for all valid t
            assert torch.equal(x[:, 1:], y[:, :-1])
            break

    def test_sequences_are_non_overlapping(self):
        """Consecutive samples use non-overlapping windows of tokens."""
        seq_len = 3
        # 12 tokens → 3 samples of length 4 (seq_len+1), non-overlapping
        tokens = torch.arange(12, dtype=torch.long)
        train_loader, _, _ = create_simple_loaders(
            tokens, seq_len, batch_size=10, validation_split=0.0
        )
        x_batch, _ = next(iter(train_loader))
        # Sort by first token to get deterministic order despite shuffle=True
        x_batch = x_batch[x_batch[:, 0].argsort()]
        # Windows start at token 0, 4, 8 — confirming non-overlapping stride of seq_len+1
        assert x_batch[0, 0].item() == 0
        assert x_batch[1, 0].item() == 4
        assert x_batch[2, 0].item() == 8

    def test_raises_if_not_enough_tokens(self):
        """Raises ValueError when token count < seq_len + 1."""
        tokens = torch.arange(5, dtype=torch.long)
        with pytest.raises(ValueError, match="Not enough tokens"):
            create_simple_loaders(tokens, seq_len=10, batch_size=1)  # Returns 3 values now

    def test_batch_shape_is_correct(self):
        """Each batch has shape (batch_size, seq_len)."""
        seq_len = 8
        tokens = self._tokens(200)
        train_loader, _, _ = create_simple_loaders(tokens, seq_len, batch_size=4)
        x, y = next(iter(train_loader))
        assert x.shape == (4, seq_len)
        assert y.shape == (4, seq_len)

    def test_pin_memory_override_is_honored(self):
        """Explicit pin_memory should be used even on CPU device."""
        seq_len = 8
        tokens = self._tokens(200)
        train_loader, _, _ = create_simple_loaders(
            tokens,
            seq_len,
            batch_size=4,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        assert train_loader.pin_memory is True


class TestEnhancedTrainingFeatures:
    """Tests for Phase 3 training loop enhancements."""

    def test_train_step_with_accumulation(self):
        """Test train_step with gradient accumulation."""
        model = _make_model()
        x, y = _make_batch()
        optimizer = _make_optimizer(model)

        # First step: accumulate gradients
        train_step(
            model=model,
            x=x,
            y=y,
            optimizer=optimizer,
            accumulate_grad=True,
        )

        # Gradients should exist
        assert model.token_embedding.embedding.weight.grad is not None
        grad_sum_1 = model.token_embedding.embedding.weight.grad.sum().item()

        # Second step: accumulate more gradients
        train_step(
            model=model,
            x=x,
            y=y,
            optimizer=optimizer,
            accumulate_grad=True,
        )

        # Gradients should have accumulated
        grad_sum_2 = model.token_embedding.embedding.weight.grad.sum().item()
        assert abs(grad_sum_2) > abs(grad_sum_1)  # More gradients accumulated

    def test_train_step_amp_cpu(self):
        """Test that AMP on CPU falls back gracefully (no error)."""
        model = _make_model()
        x, y = _make_batch()
        optimizer = _make_optimizer(model)

        # AMP on CPU should not crash
        loss = train_step(
            model=model,
            x=x,
            y=y,
            optimizer=optimizer,
            use_amp=True,
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_optimizer_step_gradient_clipping(self):
        """Test gradient clipping in optimizer_step."""
        model = _make_model()
        optimizer = _make_optimizer(model)

        # Create large gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100.0

        # Apply gradient clipping (should not crash)
        optimizer_step(
            optimizer=optimizer,
            gradient_clip_norm=1.0,
            model=model,
        )

    def test_optimizer_step_returns_inf_on_non_finite_grad_norm(self):
        """optimizer_step should return inf grad_norm (not raise) and zero grads."""
        model = _make_model()
        optimizer = _make_optimizer(model)

        for param in model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        grad_norm = optimizer_step(
            optimizer=optimizer,
            gradient_clip_norm=1.0,
            model=model,
        )
        assert not math.isfinite(grad_norm)
        # Gradients should have been zeroed
        for param in model.parameters():
            assert param.grad is None or not param.grad.any()

    def test_train_with_gradient_accumulation(self):
        """Test training with gradient accumulation."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=_make_optimizer(model),
            max_steps=3,
            log_interval=0,
            gradient_accumulation_steps=2,
        )

        # Should have 3 steps worth of losses
        assert len(metrics["losses"]) == 3
        assert len(metrics["perplexities"]) == 3

    def test_train_with_gradient_clipping(self):
        """Test training with gradient clipping."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=_make_optimizer(model),
            max_steps=5,
            log_interval=0,
            gradient_clip_norm=1.0,
        )

        # Training should complete without error
        assert len(metrics["losses"]) == 5

    def test_train_with_scheduler(self):
        """Test training with LR scheduler."""
        from torch.utils.data import DataLoader, TensorDataset

        from src.training.scheduler import get_cosine_schedule_with_warmup

        model = _make_model()
        optimizer = _make_optimizer(model)
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2,
            num_training_steps=10,
        )

        # Record initial LR
        initial_lr = optimizer.param_groups[0]["lr"]

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=10,
            log_interval=0,
            lr_scheduler=scheduler,
        )

        # LR should have changed
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr != initial_lr

        # Training should complete
        assert len(metrics["losses"]) == 10

    def test_train_with_tokens_per_sec_logging(self):
        """Test training with tokens/sec logging."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=_make_optimizer(model),
            max_steps=5,
            log_interval=0,
            log_tokens_per_sec=True,
        )

        # Should have tokens_per_sec metrics
        assert "tokens_per_sec" in metrics
        assert len(metrics["tokens_per_sec"]) == 5

        # All values should be positive
        for tps in metrics["tokens_per_sec"]:
            assert tps > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_with_gpu_memory_logging(self):
        """Test training with GPU memory logging."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model().cuda()
        x = torch.randint(0, 128, (20, 8)).cuda()
        y = torch.randint(0, 128, (20, 8)).cuda()
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=5,
            log_interval=0,
            log_gpu_memory=True,
        )

        # Should have GPU memory metrics
        assert "gpu_memory_mb" in metrics
        assert len(metrics["gpu_memory_mb"]) == 5

        # All values should be positive
        for mem in metrics["gpu_memory_mb"]:
            assert mem > 0

    def test_train_combines_all_features(self):
        """Test training with all features enabled simultaneously."""
        from torch.utils.data import DataLoader, TensorDataset

        from src.training.scheduler import get_cosine_schedule_with_warmup

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        optimizer = _make_optimizer(model)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2,
            num_training_steps=10,
        )

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=10,
            log_interval=0,
            gradient_accumulation_steps=2,
            gradient_clip_norm=1.0,
            use_amp=False,  # Keep False for CPU testing
            lr_scheduler=scheduler,
            log_tokens_per_sec=True,
            log_gpu_memory=False,  # Keep False for CPU testing
        )

        # All features should work together
        assert len(metrics["losses"]) == 10
        assert len(metrics["perplexities"]) == 10
        assert len(metrics["tokens_per_sec"]) == 10
        assert "gpu_memory_mb" not in metrics  # Not requested

    def test_train_writes_component_profile_with_gradient_accumulation(self, tmp_path):
        """Component profiling writes one row per optimizer step, not per micro-batch."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (24, 8))
        y = torch.randint(0, 128, (24, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        profile_path = tmp_path / "component_profile.csv"

        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=_make_optimizer(model),
            max_steps=3,
            log_interval=0,
            gradient_accumulation_steps=2,
            component_profile_path=profile_path,
        )

        assert len(metrics["losses"]) == 3
        with profile_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert [row["step"] for row in rows] == ["1", "2", "3"]
        assert all(float(row["forward_backward_ms"]) > 0 for row in rows)
        assert all(float(row["optimizer_ms"]) > 0 for row in rows)
        assert all(float(row["step_ms"]) >= float(row["forward_backward_ms"]) for row in rows)

    def test_generalization_filter_damps_antialigned_gradients(self):
        """Anti-aligned train gradients can be damped without norm preservation."""
        model = torch.nn.Linear(2, 1, bias=False)
        train_grads = [torch.tensor([[2.0, -3.0]])]
        val_grads = [torch.tensor([[4.0, 5.0]])]

        keep_fraction = _restore_filtered_gradients(
            model,
            train_grads=train_grads,
            val_grads=val_grads,
            damping=0.25,
            preserve_norm=False,
        )

        assert keep_fraction == 0.5
        assert model.weight.grad is not None
        assert torch.allclose(model.weight.grad, torch.tensor([[2.0, -0.75]]))

    def test_generalization_filter_rejects_invalid_damping(self):
        model = torch.nn.Linear(2, 1, bias=False)

        with pytest.raises(ValueError, match="damping"):
            _restore_filtered_gradients(
                model,
                train_grads=[torch.ones(1, 2)],
                val_grads=[torch.ones(1, 2)],
                damping=1.1,
            )

    def test_generalization_filter_preserves_gradient_norm_by_default(self):
        """Default filtering should focus direction without shrinking step norm."""
        model = torch.nn.Linear(2, 1, bias=False)
        train_grad = torch.tensor([[2.0, -3.0]])
        train_grads = [train_grad]
        val_grads = [torch.tensor([[4.0, 5.0]])]

        _restore_filtered_gradients(
            model,
            train_grads=train_grads,
            val_grads=val_grads,
            damping=0.25,
        )

        assert model.weight.grad is not None
        assert torch.allclose(model.weight.grad.norm(), train_grad.norm(), atol=1e-6)
        assert model.weight.grad[0, 1].abs() < train_grad[0, 1].abs()

    def test_generalization_filter_restores_train_gradients_after_probe(self):
        """Validation probe gradients should only filter the train update, not replace it."""
        torch.manual_seed(123)
        model = _make_model()
        optimizer = _make_optimizer(model)
        x_train, y_train = _make_batch(batch_size=2, seq_len=4)
        x_val, y_val = _make_batch(batch_size=2, seq_len=4)

        train_step(model, x_train, y_train, optimizer)
        keep_fraction, probe_loss, gradients_unscaled = apply_generalization_gradient_filter(
            model=model,
            optimizer=optimizer,
            probe_batches=[(x_val, y_val)],
            damping=0.25,
        )

        assert 0.0 <= keep_fraction <= 1.0
        assert math.isfinite(probe_loss)
        assert gradients_unscaled is False
        assert any(
            param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters()
        )

    def test_train_with_generalization_filter_logs_alignment_metrics(self, tmp_path):
        """When enabled, the train loop records gradient-alignment filter diagnostics."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x_train = torch.randint(0, 128, (12, 8))
        y_train = torch.randint(0, 128, (12, 8))
        x_val = torch.randint(0, 128, (8, 8))
        y_val = torch.randint(0, 128, (8, 8))
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)
        csv_path = tmp_path / "loss_curve.csv"

        metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=_make_optimizer(model),
            max_steps=2,
            log_interval=0,
            csv_log_path=csv_path,
            val_loader=val_loader,
            generalization_filter_enabled=True,
            generalization_filter_val_batches=1,
            generalization_filter_damping=0.25,
        )

        assert len(metrics["losses"]) == 2
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert all(row["generalization_keep_fraction"] for row in rows)
        assert all(row["generalization_probe_loss"] for row in rows)
        assert all(0.0 <= float(row["generalization_keep_fraction"]) <= 1.0 for row in rows)

    def test_train_generalization_filter_requires_validation_loader(self):
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x_train = torch.randint(0, 128, (4, 8))
        y_train = torch.randint(0, 128, (4, 8))
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2)

        with pytest.raises(ValueError, match="requires val_loader"):
            train(
                model=model,
                train_loader=train_loader,
                optimizer=_make_optimizer(model),
                max_steps=1,
                log_interval=0,
                generalization_filter_enabled=True,
            )


class TestZLoss:
    """Z-loss regularizer correctness."""

    def test_z_loss_disabled_matches_baseline(self):
        """z_loss_weight=0 must return exactly the same value as not passing it."""
        torch.manual_seed(0)
        logits = torch.randn(2, 16, 64)
        targets = torch.randint(0, 64, (2, 16))
        baseline = compute_loss_with_smoothing(logits, targets)
        with_zero = compute_loss_with_smoothing(logits, targets, z_loss_weight=0.0)
        assert torch.equal(baseline, with_zero)

    def test_z_loss_nonzero_increases_loss(self):
        """With weight > 0, total loss must exceed CE-only loss."""
        torch.manual_seed(1)
        logits = torch.randn(2, 16, 64)
        targets = torch.randint(0, 64, (2, 16))
        ce_only = compute_loss_with_smoothing(logits, targets, z_loss_weight=0.0)
        with_z = compute_loss_with_smoothing(logits, targets, z_loss_weight=1e-4)
        assert with_z.item() > ce_only.item()

    def test_z_loss_scales_with_weight(self):
        """Doubling the weight doubles the z_loss contribution."""
        torch.manual_seed(2)
        logits = torch.randn(2, 8, 32)
        targets = torch.randint(0, 32, (2, 8))
        ce = compute_loss_with_smoothing(logits, targets, z_loss_weight=0.0)
        low = compute_loss_with_smoothing(logits, targets, z_loss_weight=1e-4)
        high = compute_loss_with_smoothing(logits, targets, z_loss_weight=2e-4)
        z_low = (low - ce).item()
        z_high = (high - ce).item()
        assert abs(z_high - 2 * z_low) < 1e-6

    def test_z_loss_formula(self):
        """z_loss = weight * mean(logsumexp(logits, dim=-1)^2) — verify by hand."""
        torch.manual_seed(3)
        logits = torch.randn(1, 4, 8)
        targets = torch.randint(0, 8, (1, 4))
        weight = 1e-3
        loss = compute_loss_with_smoothing(logits, targets, z_loss_weight=weight)
        ce = compute_loss_with_smoothing(logits, targets, z_loss_weight=0.0)
        lse = torch.logsumexp(logits.view(4, 8), dim=-1)
        expected_z = weight * (lse * lse).mean()
        assert torch.allclose(loss - ce, expected_z, atol=1e-6)

    def test_z_loss_chunk_consistency(self):
        """z_loss result is the same regardless of chunk_size."""
        torch.manual_seed(4)
        logits = torch.randn(3, 32, 128)
        targets = torch.randint(0, 128, (3, 32))
        full = compute_loss_with_smoothing(logits, targets, z_loss_weight=1e-4, chunk_size=3 * 32)
        chunked = compute_loss_with_smoothing(logits, targets, z_loss_weight=1e-4, chunk_size=16)
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_z_loss_gradients_flow(self):
        """Gradients propagate through z_loss back to logits."""
        torch.manual_seed(5)
        logits = torch.randn(1, 8, 32, requires_grad=True)
        targets = torch.randint(0, 32, (1, 8))
        loss = compute_loss_with_smoothing(logits, targets, z_loss_weight=1e-4)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_z_loss_gradient_formula(self):
        """Z-loss gradient matches analytic form: (2/N) * lse * softmax(logits)."""
        torch.manual_seed(6)
        logits = torch.randn(1, 4, 16, requires_grad=True)
        # Isolate the z_loss term (no CE) so the analytic gradient is easy to verify.
        # d/d(logits[i,k]) [ mean_i(lse_i^2) ] = (2/N) * lse_i * softmax(logits[i,k])
        lse = torch.logsumexp(logits.view(4, 16), dim=-1)  # (4,)
        z_term = (lse * lse).mean()
        z_term.backward()
        grad = logits.grad.detach().view(4, 16)
        expected = (
            (2.0 / 4.0) * lse.unsqueeze(1) * torch.softmax(logits.detach().view(4, 16), dim=-1)
        )
        assert torch.allclose(grad, expected, atol=1e-6)

    def test_train_step_with_z_loss_is_finite(self):
        """train_step with z_loss_weight returns a finite float."""
        model = _make_model()
        x, y = _make_batch()
        loss = train_step(model, x, y, _make_optimizer(model), z_loss_weight=1e-4)
        assert isinstance(loss, float)
        assert math.isfinite(loss)

    def test_z_loss_with_label_smoothing(self):
        """z_loss and label_smoothing compose correctly (both nonzero)."""
        torch.manual_seed(7)
        logits = torch.randn(2, 8, 32)
        targets = torch.randint(0, 32, (2, 8))
        ce_smooth = compute_loss_with_smoothing(
            logits, targets, label_smoothing=0.1, z_loss_weight=0.0
        )
        combined = compute_loss_with_smoothing(
            logits, targets, label_smoothing=0.1, z_loss_weight=1e-4
        )
        # combined must be larger and finite
        assert combined.item() > ce_smooth.item()
        assert torch.isfinite(combined)


class TestChunkedLoss:
    """compute_loss_with_smoothing chunked CE correctness."""

    def test_chunked_matches_full_no_smoothing(self):
        # Loss computed in one chunk vs many chunks should be numerically equal.
        torch.manual_seed(0)
        logits = torch.randn(4, 32, 256)
        targets = torch.randint(0, 256, (4, 32))
        full = compute_loss_with_smoothing(logits, targets, chunk_size=4 * 32)
        chunked = compute_loss_with_smoothing(logits, targets, chunk_size=16)
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_chunked_matches_full_with_smoothing(self):
        torch.manual_seed(1)
        logits = torch.randn(4, 32, 256)
        targets = torch.randint(0, 256, (4, 32))
        full = compute_loss_with_smoothing(logits, targets, label_smoothing=0.1, chunk_size=4 * 32)
        chunked = compute_loss_with_smoothing(logits, targets, label_smoothing=0.1, chunk_size=16)
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_gradients_flow_through_chunks(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 16, 64, requires_grad=True)
        targets = torch.randint(0, 64, (2, 16))
        loss = compute_loss_with_smoothing(logits, targets, chunk_size=8)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
        assert torch.isfinite(logits.grad).all()


class TestChunkedLMHeadLoss:
    """compute_chunked_lm_loss / chunked LM-head training path."""

    @staticmethod
    def _full_loss(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.0,
        z_loss_weight: float = 0.0,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        logits = F.linear(hidden, weight)
        return compute_loss_with_smoothing(
            logits,
            targets,
            label_smoothing=label_smoothing,
            z_loss_weight=z_loss_weight,
        )

    def test_matches_full_loss_no_smoothing(self):
        torch.manual_seed(0)
        B, T, D, V = 2, 32, 16, 64
        hidden = torch.randn(B, T, D)
        weight = torch.randn(V, D)
        targets = torch.randint(0, V, (B, T))
        full = self._full_loss(hidden, weight, targets)
        chunked = compute_chunked_lm_loss(hidden, weight, targets, chunk_size=8)
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_matches_full_loss_with_smoothing_and_zloss(self):
        torch.manual_seed(1)
        B, T, D, V = 2, 24, 16, 64
        hidden = torch.randn(B, T, D)
        weight = torch.randn(V, D)
        targets = torch.randint(0, V, (B, T))
        full = self._full_loss(hidden, weight, targets, label_smoothing=0.1, z_loss_weight=1e-3)
        chunked = compute_chunked_lm_loss(
            hidden,
            weight,
            targets,
            label_smoothing=0.1,
            z_loss_weight=1e-3,
            chunk_size=7,
        )
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_gradients_match_full_path(self):
        """Chunked path must produce the same hidden + weight gradients as the
        full ``lm_head(h) → CE`` reference within fp32 numerical tolerance.
        """
        torch.manual_seed(2)
        B, T, D, V = 2, 16, 8, 32
        hidden_a = torch.randn(B, T, D, requires_grad=True)
        weight_a = torch.randn(V, D, requires_grad=True)
        targets = torch.randint(0, V, (B, T))

        loss_full = self._full_loss(
            hidden_a, weight_a, targets, label_smoothing=0.05, z_loss_weight=1e-4
        )
        loss_full.backward()

        hidden_b = hidden_a.detach().clone().requires_grad_(True)
        weight_b = weight_a.detach().clone().requires_grad_(True)
        loss_chunked = compute_chunked_lm_loss(
            hidden_b,
            weight_b,
            targets,
            label_smoothing=0.05,
            z_loss_weight=1e-4,
            chunk_size=5,
        )
        loss_chunked.backward()

        assert torch.allclose(loss_full, loss_chunked, atol=1e-5)
        assert hidden_a.grad is not None and hidden_b.grad is not None
        assert weight_a.grad is not None and weight_b.grad is not None
        assert torch.allclose(hidden_a.grad, hidden_b.grad, atol=1e-5)
        assert torch.allclose(weight_a.grad, weight_b.grad, atol=1e-5)

    def test_chunk_size_invariance(self):
        torch.manual_seed(3)
        B, T, D, V = 2, 24, 8, 32
        hidden = torch.randn(B, T, D)
        weight = torch.randn(V, D)
        targets = torch.randint(0, V, (B, T))
        a = compute_chunked_lm_loss(hidden, weight, targets, chunk_size=T)
        b = compute_chunked_lm_loss(hidden, weight, targets, chunk_size=4)
        c = compute_chunked_lm_loss(hidden, weight, targets, chunk_size=1)
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)

    def test_invalid_chunk_size_raises(self):
        hidden = torch.randn(1, 4, 8)
        weight = torch.randn(16, 8)
        targets = torch.randint(0, 16, (1, 4))
        with pytest.raises(ValueError, match="chunk_size"):
            compute_chunked_lm_loss(hidden, weight, targets, chunk_size=0)

    def test_train_step_chunked_matches_default(self):
        """train_step with use_chunked_loss=True yields the same loss value
        and same parameter updates (within tolerance) as the default path.
        """
        torch.manual_seed(7)
        model_a = _make_model()
        model_b = _make_model()
        model_b.load_state_dict(model_a.state_dict())

        x, y = _make_batch()
        opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)

        loss_a = train_step(model_a, x, y, opt_a)
        optimizer_step(opt_a, model=model_a)

        loss_b = train_step(model_b, x, y, opt_b, use_chunked_loss=True, loss_chunk_size=3)
        optimizer_step(opt_b, model=model_b)

        assert abs(loss_a - loss_b) < 1e-4
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
            assert torch.allclose(p_a, p_b, atol=1e-4)

    def test_train_step_chunked_requires_forward_hidden(self):
        """A model without forward_hidden raises a clear error."""

        class Dummy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(4, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(x.float())

        model = Dummy()
        x = torch.randint(0, 4, (2, 4))
        y = torch.randint(0, 8, (2, 4))
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        with pytest.raises(AttributeError, match="forward_hidden"):
            train_step(model, x, y, opt, use_chunked_loss=True)

    def test_loss_mask_matches_selected_positions(self):
        torch.manual_seed(11)
        logits = torch.randn(2, 4, 16, requires_grad=True)
        targets = torch.randint(0, 16, (2, 4))
        loss_mask = torch.tensor(
            [[True, False, True, True], [False, True, False, True]], dtype=torch.bool
        )

        masked = compute_loss_with_smoothing(logits, targets, loss_mask=loss_mask)
        selected_logits = logits[loss_mask]
        selected_targets = targets[loss_mask]
        expected = torch.nn.functional.cross_entropy(selected_logits, selected_targets)

        assert torch.allclose(masked, expected, atol=1e-6)

    def test_chunked_lm_loss_mask_matches_full_masked_loss(self):
        torch.manual_seed(12)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        weight = torch.randn(16, 8, requires_grad=True)
        targets = torch.randint(0, 16, (2, 5))
        loss_mask = torch.tensor(
            [[True, True, False, True, False], [False, True, True, False, True]],
            dtype=torch.bool,
        )

        full_logits = torch.nn.functional.linear(hidden, weight)
        full = compute_loss_with_smoothing(full_logits, targets, loss_mask=loss_mask)
        chunked = compute_chunked_lm_loss(
            hidden, weight, targets, chunk_size=2, loss_mask=loss_mask
        )

        assert torch.allclose(full, chunked, atol=1e-6)


class TestPackedTokenDataset:
    def test_create_loader_uses_packed_boundaries_for_masks(self):
        tokens = np.asarray([10, 11, 12, 13, 14, 15, 20, 21, 22, 0, 0, 0], dtype=np.int32)
        sequence_offsets = np.asarray([0, 2, 3], dtype=np.int64)
        boundaries = np.asarray([2, 6, 3], dtype=np.int32)

        train_loader, val_loader, _ = create_simple_loaders(
            tokens,
            seq_len=5,
            batch_size=1,
            train_packing_metadata=(sequence_offsets, boundaries),
            validation_split=0.5,
            num_workers=0,
        )

        x, y, document_ids, loss_mask = next(iter(train_loader))
        assert x.shape == y.shape == document_ids.shape == loss_mask.shape == (1, 5)
        assert document_ids[0].tolist() == [0, 0, 1, 1, 1]
        assert loss_mask[0].tolist() == [True, False, True, True, True]

        x_val, y_val, doc_ids_val, mask_val = next(iter(val_loader))
        assert x_val[0].tolist() == [20, 21, 22, 0, 0]
        assert y_val[0].tolist() == [21, 22, 0, 0, 0]
        assert doc_ids_val[0].tolist() == [0, 0, 0, -1, -1]
        assert mask_val[0].tolist() == [True, True, False, False, False]
