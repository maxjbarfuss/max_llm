"""Overfit tests — verify models can memorize small datasets.

Phase 2: SimpleLM overfits 10K-token char-level dataset.
Phase 3: DecoderLM overfits 1K-token char-level dataset (exit criterion).
"""

import torch

from src.config.model import ModelConfig
from src.models.learning_model import DecoderLM, SimpleLM
from src.training.loop import train
from src.training.train import create_simple_loaders


def _make_model() -> SimpleLM:
    """Create a small model for Phase 2 overfitting."""
    config = ModelConfig(
        model_type="simple_lm",
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
        train_loader, _, _ = create_simple_loaders(
            train_tokens=tokens,
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
        train_loader, _, _ = create_simple_loaders(
            train_tokens=tokens,
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
        train_loader, _, _ = create_simple_loaders(
            train_tokens=tokens,
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


def _make_decoder_model() -> DecoderLM:
    """Create a small DecoderLM for Phase 3 overfitting."""
    config = ModelConfig(
        model_type="decoder_lm",
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        vocab_size=256,
        max_seq_length=64,
        mla_latent_dim=64,
        rope_base=10000,
        intermediate_size=512,
        num_experts=1,
        experts_per_token=1,
        moe_frequency=0,
        gru_hidden_size=128,
        dropout=0.0,
    )
    return DecoderLM.from_config(config, attention_backend="standard")


class TestDecoderLMOverfit:
    """Phase 3 exit criterion: DecoderLM overfits 1K-token subset to loss < 0.5 within 1000 steps."""

    def test_decoder_lm_overfits_1k_tokens(self):
        """
        DecoderLM overfits 1K-token subset: train loss < 0.5 within 1000 steps.

        Phase 3 exit criterion: model_type=decoder_lm, perplexity < 2.0.
        """
        # 1K+ tokens with repeating pattern for deterministic overfitting
        tokens = torch.arange(256, dtype=torch.long).repeat(5)  # 1280 tokens
        assert len(tokens) >= 1000

        train_loader, _, _ = create_simple_loaders(
            train_tokens=tokens,
            seq_len=64,
            batch_size=4,
            validation_split=0.0,
            seed=42,
        )

        model = _make_decoder_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            max_steps=1000,
            log_interval=0,
        )

        losses = metrics["losses"]
        assert len(losses) > 0

        min_loss = min(losses)
        final_loss = losses[-1]
        min_ppl = 2.718281828**min_loss  # e^loss

        assert min_loss < 0.5, (
            f"DecoderLM overfit failed: min_loss={min_loss:.4f} (need < 0.5), "
            f"final_loss={final_loss:.4f}, steps={len(losses)}"
        )
        assert min_ppl < 2.0, f"DecoderLM overfit failed: min_ppl={min_ppl:.4f} (need < 2.0)"
        print(f"✓ DecoderLM loss < 0.5 achieved: min_loss={min_loss:.4f}, min_ppl={min_ppl:.4f}")
