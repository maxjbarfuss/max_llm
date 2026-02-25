"""Tests for seed hardening and deterministic training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.model import ModelConfig
from src.models.learning_model import SimpleLM
from src.training.loop import train
from src.training.train import create_simple_loaders
from src.utils import seed_everything


def _make_model_config() -> ModelConfig:
    """Create a minimal model config for testing."""
    return ModelConfig(
        hidden_size=64,  # Must be multiple of 64
        num_layers=1,
        num_heads=4,
        vocab_size=64,
        max_seq_length=32,
        mla_latent_dim=32,
        rope_base=10000,
        intermediate_size=None,
        num_experts=1,
        experts_per_token=1,
        moe_frequency=0,
        gru_hidden_size=None,
        dropout=0.0,
    )


class TestSeedEverything:
    """Tests for seed_everything utility."""

    def test_seed_everything_sets_torch_seed(self):
        """seed_everything sets PyTorch random seed."""
        seed_everything(42, deterministic=False)
        val1 = torch.rand(1).item()

        seed_everything(42, deterministic=False)
        val2 = torch.rand(1).item()

        assert val1 == val2, "Same seed should produce identical torch.rand values"

    def test_seed_everything_sets_python_random(self):
        """seed_everything sets Python random seed."""
        import random

        seed_everything(42, deterministic=False)
        val1 = random.random()

        seed_everything(42, deterministic=False)
        val2 = random.random()

        assert val1 == val2, "Same seed should produce identical random.random values"

    def test_seed_everything_sets_numpy_random(self):
        """seed_everything sets NumPy random seed."""
        import numpy as np

        seed_everything(42, deterministic=False)
        val1 = np.random.rand()

        seed_everything(42, deterministic=False)
        val2 = np.random.rand()

        assert val1 == val2, "Same seed should produce identical np.random.rand values"


class TestDeterministicReplay:
    """Tests for deterministic training replay."""

    def test_deterministic_model_initialization(self):
        """Same seed produces identical model initialization."""
        seed_everything(42, deterministic=True)
        model1 = SimpleLM.from_config(_make_model_config())

        seed_everything(42, deterministic=True)
        model2 = SimpleLM.from_config(_make_model_config())

        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.equal(
                p1, p2
            ), "Models initialized with same seed should have identical parameters"

    def test_deterministic_dataloader_shuffle(self):
        """Same seed produces identical DataLoader shuffle order."""
        tokens = torch.arange(0, 1000, dtype=torch.long)

        seed_everything(42, deterministic=True)
        loader1, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=16,
            batch_size=4,
            validation_split=0.1,
            seed=42,
        )

        seed_everything(42, deterministic=True)
        loader2, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=16,
            batch_size=4,
            validation_split=0.1,
            seed=42,
        )

        # Extract first batches from both loaders
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        assert torch.equal(
            batch1[0], batch2[0]
        ), "Same seed should produce identical first batch inputs"
        assert torch.equal(
            batch1[1], batch2[1]
        ), "Same seed should produce identical first batch targets"

    def test_deterministic_training_replay_single_step(self):
        """Single training step is deterministic with same seed."""
        # Prepare data
        tokens = torch.randint(0, 64, (500,), dtype=torch.long)
        x = tokens[:100].reshape(10, 10)
        y = tokens[1:101].reshape(10, 10)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # First run
        seed_everything(42, deterministic=True)
        model1 = SimpleLM.from_config(_make_model_config())
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        metrics1 = train(model1, loader, optimizer1, max_steps=1, log_interval=0)

        # Second run
        seed_everything(42, deterministic=True)
        model2 = SimpleLM.from_config(_make_model_config())
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        metrics2 = train(model2, loader, optimizer2, max_steps=1, log_interval=0)

        # Verify identical losses
        assert len(metrics1["losses"]) == len(metrics2["losses"]) == 1
        assert (
            metrics1["losses"][0] == metrics2["losses"][0]
        ), "Same seed should produce identical loss for step 1"

        # Verify identical final parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.equal(
                p1, p2
            ), "Same seed should produce identical parameters after one step"

    def test_deterministic_training_replay_multi_step(self):
        """Multi-step training is deterministic with same seed."""
        # Prepare data (enough for multiple batches)
        tokens = torch.randint(0, 64, (2000,), dtype=torch.long)

        # First run
        seed_everything(123, deterministic=True)
        loader1, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=16,
            batch_size=4,
            validation_split=0.1,
            seed=123,
        )
        model1 = SimpleLM.from_config(_make_model_config())
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        metrics1 = train(model1, loader1, optimizer1, max_steps=10, log_interval=0)

        # Second run
        seed_everything(123, deterministic=True)
        loader2, _ = create_simple_loaders(
            tokens=tokens,
            seq_len=16,
            batch_size=4,
            validation_split=0.1,
            seed=123,
        )
        model2 = SimpleLM.from_config(_make_model_config())
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        metrics2 = train(model2, loader2, optimizer2, max_steps=10, log_interval=0)

        # Verify identical loss trajectory
        assert len(metrics1["losses"]) == len(metrics2["losses"]) == 10
        for step, (loss1, loss2) in enumerate(
            zip(metrics1["losses"], metrics2["losses"], strict=True)
        ):
            assert loss1 == loss2, f"Losses at step {step + 1} differ: {loss1} != {loss2}"

        # Verify identical perplexity trajectory
        for step, (ppl1, ppl2) in enumerate(
            zip(metrics1["perplexities"], metrics2["perplexities"], strict=True)
        ):
            assert ppl1 == ppl2, f"Perplexities at step {step + 1} differ: {ppl1} != {ppl2}"

        # Verify identical final parameters
        for idx, (p1, p2) in enumerate(zip(model1.parameters(), model2.parameters(), strict=True)):
            assert torch.equal(p1, p2), f"Parameter {idx} differs after 10 steps with same seed"

    def test_different_seeds_produce_different_results(self):
        """Different seeds produce different training trajectories."""
        tokens = torch.randint(0, 64, (1000,), dtype=torch.long)

        # Run with seed 42
        seed_everything(42, deterministic=True)
        loader1, _ = create_simple_loaders(
            tokens=tokens, seq_len=16, batch_size=4, validation_split=0.1, seed=42
        )
        model1 = SimpleLM.from_config(_make_model_config())
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        metrics1 = train(model1, loader1, optimizer1, max_steps=5, log_interval=0)

        # Run with seed 99
        seed_everything(99, deterministic=True)
        loader2, _ = create_simple_loaders(
            tokens=tokens, seq_len=16, batch_size=4, validation_split=0.1, seed=99
        )
        model2 = SimpleLM.from_config(_make_model_config())
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        metrics2 = train(model2, loader2, optimizer2, max_steps=5, log_interval=0)

        # Verify losses are different
        assert (
            metrics1["losses"] != metrics2["losses"]
        ), "Different seeds should produce different losses"

        # Verify at least some parameters differ
        params_differ = False
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            if not torch.equal(p1, p2):
                params_differ = True
                break
        assert params_differ, "Different seeds should produce different final parameters"
