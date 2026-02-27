"""Unit tests for checkpoint save/load (Phase 2)."""

import pytest
import torch

from src.config.model import ModelConfig
from src.inference.utils import load_checkpoint_into_model
from src.models.learning_model import SimpleLM
from src.training.train import save_checkpoint


def _make_model() -> SimpleLM:
    config = ModelConfig(
        model_type="simple_lm",
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


def _make_optimizer(model: SimpleLM) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


class TestSaveCheckpoint:
    def test_creates_file(self, tmp_path):
        """save_checkpoint writes a file to the output directory."""
        model = _make_model()
        path = save_checkpoint(model, _make_optimizer(model), step=10, output_dir=tmp_path)
        assert path.exists()

    def test_contains_expected_keys(self, tmp_path):
        """Checkpoint dict contains model_state, optimizer_state, and step."""
        model = _make_model()
        path = save_checkpoint(model, _make_optimizer(model), step=5, output_dir=tmp_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert "model_state" in ckpt
        assert "optimizer_state" in ckpt
        assert "step" in ckpt
        assert ckpt["step"] == 5

    def test_roundtrip_preserves_weights(self, tmp_path):
        """Weights loaded from checkpoint match the saved model exactly."""
        model = _make_model()
        weights_before = {k: v.clone() for k, v in model.state_dict().items()}
        path = save_checkpoint(model, _make_optimizer(model), step=1, output_dir=tmp_path)

        model2 = _make_model()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model2.load_state_dict(ckpt["model_state"])

        for key in weights_before:
            assert torch.equal(weights_before[key], model2.state_dict()[key]), key

    def test_creates_output_dir_if_missing(self, tmp_path):
        """save_checkpoint creates nested output directories as needed."""
        model = _make_model()
        nested = tmp_path / "a" / "b" / "c"
        path = save_checkpoint(model, _make_optimizer(model), step=1, output_dir=nested)
        assert path.exists()


class TestLoadCheckpoint:
    def test_loads_from_model_state_key(self, tmp_path):
        """load_checkpoint reads weights from the 'model_state' key."""
        model = _make_model()
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        path = tmp_path / "ckpt.pt"
        torch.save({"model_state": model.state_dict()}, path)

        fresh = _make_model()
        load_checkpoint_into_model(fresh, str(path), torch.device("cpu"))
        for key in weights:
            assert torch.equal(weights[key], fresh.state_dict()[key]), key

    def test_loads_from_state_dict_key(self, tmp_path):
        """load_checkpoint falls back to 'state_dict' key."""
        model = _make_model()
        path = tmp_path / "ckpt.pt"
        torch.save({"state_dict": model.state_dict()}, path)

        fresh = _make_model()
        load_checkpoint_into_model(fresh, str(path), torch.device("cpu"))
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], fresh.state_dict()[key])

    def test_loads_from_model_key(self, tmp_path):
        """load_checkpoint falls back to 'model' key."""
        model = _make_model()
        path = tmp_path / "ckpt.pt"
        torch.save({"model": model.state_dict()}, path)

        fresh = _make_model()
        load_checkpoint_into_model(fresh, str(path), torch.device("cpu"))
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], fresh.state_dict()[key])

    def test_loads_flat_state_dict(self, tmp_path):
        """load_checkpoint handles a flat state dict (no wrapper dict)."""
        model = _make_model()
        path = tmp_path / "ckpt.pt"
        torch.save(model.state_dict(), path)

        fresh = _make_model()
        load_checkpoint_into_model(fresh, str(path), torch.device("cpu"))
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], fresh.state_dict()[key])

    def test_raises_for_unsupported_format(self, tmp_path):
        """load_checkpoint raises ValueError for non-dict checkpoint."""
        path = tmp_path / "ckpt.pt"
        torch.save([1, 2, 3], path)  # list is not a supported format

        model = _make_model()
        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            load_checkpoint_into_model(model, str(path), torch.device("cpu"))
