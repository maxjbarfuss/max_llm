"""Integration test: end-to-end Phase 2 pipeline with perplexity metrics."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.model import ModelConfig
from src.inference.sampler import sample_token
from src.inference.utils import load_checkpoint_into_model
from src.models.learning_model import SimpleLM
from src.training.loop import train
from src.training.train import create_simple_loaders, save_checkpoint


def _make_model_config() -> ModelConfig:
    """Create a model config for testing."""
    return ModelConfig(
        model_type="simple_lm",
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        vocab_size=128,
        max_seq_length=32,
        mla_latent_dim=64,
        rope_base=10000,
        intermediate_size=None,
        num_experts=1,
        experts_per_token=1,
        moe_frequency=0,
        gru_hidden_size=None,
        dropout=0.0,
    )


class TestPhase2Integration:
    """Full end-to-end pipeline: preprocess → train → checkpoint → infer."""

    def test_train_returns_metrics_dict_with_perplexities(self, tmp_path):
        """train() returns dict with losses and perplexities (not just list)."""
        config = _make_model_config()
        model = SimpleLM.from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Create synthetic data
        x = torch.randint(0, 128, (20, 32))
        y = torch.randint(0, 128, (20, 32))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        # Train and check metrics
        metrics = train(model, loader, optimizer, max_steps=10, log_interval=0)

        # Verify it's a dict with expected keys
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        assert "losses" in metrics
        assert "perplexities" in metrics
        assert len(metrics["losses"]) == 10
        assert len(metrics["perplexities"]) == 10

    def test_perplexities_computed_from_losses(self):
        """Perplexities are exp(loss) for each loss value."""
        config = _make_model_config()
        model = SimpleLM.from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        x = torch.randint(0, 128, (20, 32))
        y = torch.randint(0, 128, (20, 32))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        metrics = train(model, loader, optimizer, max_steps=10, log_interval=0)

        # Verify perplexity = exp(loss) for each step
        import math

        for loss, perplexity in zip(metrics["losses"], metrics["perplexities"], strict=True):
            expected_perplexity = math.exp(loss)
            assert (
                abs(perplexity - expected_perplexity) < 1e-5
            ), f"Perplexity mismatch: {perplexity} vs {expected_perplexity}"

    def test_loss_decreases_over_training(self):
        """Loss should generally decrease; verify first loss > final loss."""
        config = _make_model_config()
        model = SimpleLM.from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        x = torch.randint(0, 128, (20, 32))
        y = torch.randint(0, 128, (20, 32))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        metrics = train(model, loader, optimizer, max_steps=20, log_interval=0)

        # Loss should decrease over training
        assert (
            metrics["losses"][-1] < metrics["losses"][0]
        ), f"Loss did not decrease: {metrics['losses'][0]:.4f} → {metrics['losses'][-1]:.4f}"

    def test_checkpoint_and_inference_roundtrip(self, tmp_path):
        """Save checkpoint → load → generate: full roundtrip."""
        config = _make_model_config()
        model = SimpleLM.from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        x = torch.randint(0, 128, (20, 32))
        y = torch.randint(0, 128, (20, 32))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        train(model, loader, optimizer, max_steps=10, log_interval=0)

        # Save checkpoint
        ckpt_path = save_checkpoint(model, optimizer, step=10, output_dir=tmp_path)
        assert ckpt_path.exists()

        # Load fresh model
        model2 = SimpleLM.from_config(config)
        load_checkpoint_into_model(model2, str(ckpt_path), torch.device("cpu"))
        model2.eval()

        # Generate from checkpoint
        prompt_tokens = [10, 20, 30]
        generated = list(prompt_tokens)
        with torch.no_grad():
            for _ in range(10):
                input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
                logits = model2(input_ids)[0, -1]
                next_token = sample_token(
                    logits,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                )
                generated.append(next_token)

        # Verify correctness
        assert len(generated) == len(prompt_tokens) + 10
        assert all(0 <= t < 128 for t in generated)

    def test_full_workflow_with_data_loaders(self, tmp_path):
        """Full workflow: create loaders → train → checkpoint → infer."""
        config = _make_model_config()

        # Create synthetic token data
        tokens = torch.randint(0, 128, (1000,), dtype=torch.long)

        # Create loaders
        train_loader, val_loader = create_simple_loaders(
            tokens=tokens,
            seq_len=32,
            batch_size=4,
            validation_split=0.2,
            seed=42,
        )

        # Train
        model = SimpleLM.from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        metrics = train(model, train_loader, optimizer, max_steps=20, log_interval=0)

        # Verify metrics
        assert len(metrics["losses"]) == 20
        assert len(metrics["perplexities"]) == 20
        assert metrics["losses"][-1] < metrics["losses"][0]

        # Checkpoint
        ckpt_path = save_checkpoint(model, optimizer, step=20, output_dir=tmp_path)

        # Inference
        model2 = SimpleLM.from_config(config)
        load_checkpoint_into_model(model2, str(ckpt_path), torch.device("cpu"))
        model2.eval()

        prompt = [10, 20]
        generated = list(prompt)
        with torch.no_grad():
            for _ in range(5):
                logits = model2(torch.tensor(generated, dtype=torch.long).unsqueeze(0))[0, -1]
                generated.append(sample_token(logits, temperature=1.0, top_p=1.0, top_k=0))

        assert len(generated) == len(prompt) + 5
