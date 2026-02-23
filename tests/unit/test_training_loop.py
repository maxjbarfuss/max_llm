"""Unit tests for the Phase 2 training loop."""

import torch

from src.config.model import ModelConfig
from src.models.learning_model import SimpleLM
from src.training.loop import train, train_step


def _make_model() -> SimpleLM:
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
    return SimpleLM.from_config(config)


def _make_batch(batch_size: int = 4, seq_len: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 128, (batch_size, seq_len))
    y = torch.randint(0, 128, (batch_size, seq_len))
    return x, y


def _make_optimizer(model: SimpleLM) -> torch.optim.Optimizer:
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
        """Model parameters are updated after a training step."""
        model = _make_model()
        params_before = [p.clone() for p in model.parameters()]
        x, y = _make_batch()
        train_step(model, x, y, _make_optimizer(model))
        params_after = list(model.parameters())
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after, strict=True)
        )


class TestTrain:
    """train() loop behaviour."""

    def test_returns_loss_list_of_correct_length(self):
        """train() returns a list with exactly max_steps entries."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        losses = train(model, loader, _make_optimizer(model), max_steps=5, log_interval=0)
        assert len(losses) == 5

    def test_all_losses_finite(self):
        """All returned losses are finite."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        x = torch.randint(0, 128, (20, 8))
        y = torch.randint(0, 128, (20, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        losses = train(model, loader, _make_optimizer(model), max_steps=10, log_interval=0)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    def test_loss_decreases_on_repeated_batch(self):
        """Loss falls when training repeatedly on the same tiny batch."""
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        model = _make_model()
        # Single repeated batch — model must memorise it
        x = torch.randint(0, 128, (4, 8))
        y = torch.randint(0, 128, (4, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        losses = train(
            model, loader, torch.optim.Adam(model.parameters(), lr=1e-2),
            max_steps=100, log_interval=0,
        )
        assert losses[-1] < losses[0], (
            f"Expected loss to decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_cycles_loader_when_steps_exceed_dataset(self):
        """train() cycles through the loader until max_steps is reached."""
        from torch.utils.data import DataLoader, TensorDataset

        model = _make_model()
        # Only 2 batches in the loader, but we request 7 steps
        x = torch.randint(0, 128, (8, 8))
        y = torch.randint(0, 128, (8, 8))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        losses = train(model, loader, _make_optimizer(model), max_steps=7, log_interval=0)
        assert len(losses) == 7
