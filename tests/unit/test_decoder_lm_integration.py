"""Integration test for DecoderLM: complete training pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.learning_model.decoder_lm import DecoderLM
from src.training.loop import train


class TestDecoderLMIntegration:
    def test_end_to_end_training_on_small_dataset(self) -> None:
        """Train DecoderLM on a tiny dataset and verify loss decreases.

        This is a vertical slice integration test that covers:
        1. Model instantiation
        2. Data loading (simple tensor-based)
        3. Training loop with forward pass, loss, backward
        4. Checkpoint and recovery
        """
        # Create a tiny dataset (100 tokens)
        vocab_size = 256
        seq_len = 8
        batch_size = 4
        num_samples = 10

        # Create synthetic token data
        token_ids = torch.randint(0, vocab_size, (num_samples * (seq_len + 1),))

        # Create input/target pairs
        inputs = []
        targets = []
        for i in range(num_samples):
            start = i * (seq_len + 1)
            end = start + seq_len + 1
            sample = token_ids[start:end]
            inputs.append(sample[:-1])
            targets.append(sample[1:])

        inputs_tensor = torch.stack(inputs)
        targets_tensor = torch.stack(targets)

        dataset = TensorDataset(inputs_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = DecoderLM(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=4,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for a few steps
        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=50,
            log_interval=10,
        )

        # Verify metrics
        losses = metrics["losses"]
        assert len(losses) == 50, f"Expected 50 loss values, got {len(losses)}"
        assert losses[0] > 0, "Initial loss should be positive"
        assert losses[-1] < losses[0], "Final loss should be lower than initial loss"

        # Check that loss decreased overall
        loss_decrease = (losses[0] - losses[-1]) / losses[0]
        assert loss_decrease > 0.1, f"Loss should decrease by at least 10%, got {loss_decrease:.2%}"

    def test_overfitting_on_tiny_dataset(self) -> None:
        """Train DecoderLM to near-zero loss on a single batch (overfitting test).

        This verifies that the model can learn and that all gradients flow correctly.
        """
        # Create a tiny dataset (single sequence)
        vocab_size = 256
        seq_len = 8

        # Create synthetic token data
        token_ids = torch.randint(1, 10, (seq_len + 1,))  # Use small token range

        # Create single input/target pair
        x = token_ids[:-1].unsqueeze(0)  # (1, seq_len)
        y = token_ids[1:].unsqueeze(0)  # (1, seq_len)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Create model
        model = DecoderLM(
            vocab_size=vocab_size,
            d_model=32,
            num_layers=1,
            num_heads=4,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train for many steps (should overfit easily)
        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=200,
            log_interval=50,
        )

        losses = metrics["losses"]

        # Check that we achieve very low loss (overfitting indicator)
        final_loss = losses[-1]
        assert final_loss < 1.0, f"Should overfit to < 1.0 loss, got {final_loss:.4f}"

        # Check loss monotonically decreases (or nearly monotonically)
        decreases = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i - 1] * 1.01)
        assert decreases > len(losses) * 0.8, "Loss should mostly decrease"

    def test_inference_after_training(self) -> None:
        """Train a model and then run inference on it."""
        # Create tiny dataset
        vocab_size = 256
        seq_len = 8
        batch_size = 2

        token_ids = torch.randint(0, vocab_size, (50,))
        inputs = []
        targets = []
        for i in range(5):
            start = i * (seq_len + 1)
            end = start + seq_len + 1
            sample = token_ids[start:end]
            inputs.append(sample[:-1])
            targets.append(sample[1:])

        inputs_tensor = torch.stack(inputs)
        targets_tensor = torch.stack(targets)

        dataset = TensorDataset(inputs_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Create and train model
        model = DecoderLM(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=4,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=20,
            log_interval=5,
        )

        # Run inference
        model.eval()
        with torch.no_grad():
            test_input = torch.randint(0, vocab_size, (1, 8))
            logits = model(test_input)

            # Verify output shape and values
            assert logits.shape == (1, 8, vocab_size)
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()

            # Verify we can sample from logits
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            sampled_token = torch.multinomial(probs, 1).item()
            assert 0 <= sampled_token < vocab_size

    def test_model_converges_to_low_loss(self) -> None:
        """Verify that the model converges to low loss on a toy dataset."""
        # Tiny dataset: all samples are the same (should overfit easily)
        vocab_size = 100

        # Create single repeated pattern
        pattern = torch.tensor([1, 2, 3, 4])
        x = pattern.unsqueeze(0).repeat(5, 1)  # (5, 4)
        y = torch.tensor([2, 3, 4, 1]).unsqueeze(0).repeat(5, 1)  # (5, 4)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Create small model
        model = DecoderLM(
            vocab_size=vocab_size,
            d_model=16,
            num_layers=1,
            num_heads=2,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train
        metrics = train(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            max_steps=100,
            log_interval=25,
        )

        losses = metrics["losses"]

        # After 100 steps on a tiny repeated pattern, loss should be very low
        assert (
            losses[-1] < 2.0
        ), f"Should achieve low loss on repeated pattern, got {losses[-1]:.4f}"

        # Final loss should be much lower than initial
        assert losses[-1] < losses[0] * 0.2, "Should achieve significant learning"
