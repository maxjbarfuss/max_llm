"""Unit tests for the Phase 2/3 training loop."""

import pytest
import torch

import src.training.loop as loop
from src.config.model import ModelConfig
from src.models.learning_model import LearningModel
from src.training.loop import compute_loss_with_smoothing, optimizer_step, train, train_step
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

    def test_optimizer_step_raises_on_non_finite_grad_norm(self):
        """optimizer_step should fail fast when grad norm is NaN/Inf."""
        model = _make_model()
        optimizer = _make_optimizer(model)

        for param in model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        with pytest.raises(FloatingPointError, match="Non-finite gradient norm"):
            optimizer_step(
                optimizer=optimizer,
                gradient_clip_norm=1.0,
                model=model,
            )

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
