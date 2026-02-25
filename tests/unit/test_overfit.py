"""Overfit test for Phase 2 — verify model can memorize small dataset."""

import torch

from src.config.model import ModelConfig
from src.models.learning_model import SimpleLM
from src.training.loop import train
from src.training.train import create_simple_loaders


def _make_model() -> SimpleLM:
    """Create a small model for Phase 2 overfitting."""
    config = ModelConfig(
        hidden_size=128,
        num_layers=1,
        num_heads=4,
        vocab_size=256,
        max_seq_length=128,
        mla_latent_dim=64,
        rope_base=10000,
        intermediate_size=512,
        num_experts=1,
        experts_per_token=1,
        moe_frequency=0,
        gru_hidden_size=128,
        dropout=0.0,
    )
    return SimpleLM.from_config(config)


class TestOverfit:
    """Verify model can overfit a small dataset to < 0.1 loss within 500 steps."""

    def test_overfit_10k_tokens_to_loss_below_01(self):
        """
        Model overfits 10K-token subset: train loss < 0.1 within 500 steps.

        This is a core Phase 2 exit criterion.
        """
        # Create 10K-token dataset (repeating pattern for deterministic overfitting)
        tokens = torch.arange(256, dtype=torch.long).repeat(40)  # 10240 tokens
        assert len(tokens) >= 10000

        # Create loaders with no validation split (all train)
        train_loader, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=128,
            batch_size=4,
            validation_split=0.0,  # All data is training
            seed=42,
        )

        # Create model and optimizer with overfit settings
        model = _make_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,  # Higher LR for aggressive overfitting
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        # Train for up to 500 steps
        metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            max_steps=500,
            log_interval=0,  # Suppress logging
        )

        losses = metrics["losses"]
        assert len(losses) > 0

        # Find the minimum loss achieved
        min_loss = min(losses)
        final_loss = losses[-1]

        # Phase 2 exit criterion: loss < 0.1 within 500 steps
        assert (
            min_loss < 0.1
        ), f"Overfit test failed: min_loss={min_loss:.4f}, final_loss={final_loss:.4f}, steps={len(losses)}"

    def test_overfit_convergence_curve(self):
        """
        Loss decreases monotonically (approximately) during overfitting on small dataset.

        Verifies that training is making progress and not diverging.
        """
        tokens = torch.arange(256, dtype=torch.long).repeat(40)  # 10240 tokens
        train_loader, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=128,
            batch_size=4,
            validation_split=0.0,
            seed=42,
        )

        model = _make_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            max_steps=500,
            log_interval=0,
        )

        losses = metrics["losses"]

        # Verify loss generally decreases
        # Allow for some variance but check overall trend
        initial_loss = losses[0]
        final_loss = losses[-1]
        first_half_avg = sum(losses[: len(losses) // 2]) / (len(losses) // 2)
        second_half_avg = sum(losses[len(losses) // 2 :]) / (len(losses) - len(losses) // 2)

        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        assert (
            second_half_avg < first_half_avg
        ), f"Loss did not trend down: first_half_avg={first_half_avg:.4f}, second_half_avg={second_half_avg:.4f}"

    def test_overfit_within_500_steps(self):
        """Target loss < 0.1 is achieved within the 500-step limit."""
        tokens = torch.arange(256, dtype=torch.long).repeat(40)  # 10240 tokens
        train_loader, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=128,
            batch_size=4,
            validation_split=0.0,
            seed=42,
        )

        model = _make_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            max_steps=500,
            log_interval=0,
        )

        losses = metrics["losses"]

        # Find step at which loss first drops below 0.1
        step_threshold = None
        for step, loss in enumerate(losses):
            if loss < 0.1:
                step_threshold = step + 1  # 1-indexed
                break

        assert (
            step_threshold is not None and step_threshold <= 500
        ), f"Loss never dropped below 0.1 within 500 steps. Min loss: {min(losses):.4f}, achieved at step {losses.index(min(losses)) + 1}"

        print(f"✓ Loss < 0.1 achieved at step {step_threshold}")
