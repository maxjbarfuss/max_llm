"""Tests for config versioning and checkpoint compatibility."""

import pytest
import torch

from src.config import (
    ConfigVersionMismatchError,
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    get_checkpoint_config_versions,
    validate_checkpoint_config_compatibility,
)
from src.models.learning_model import DecoderLM
from src.training.train import load_checkpoint, save_checkpoint
from tests.conftest import build_model_config


class TestConfigVersioning:
    """Test config version tracking and validation."""

    def test_all_configs_have_version(self):
        """All config classes should have __version__ attribute."""
        assert hasattr(ModelConfig, "__version__")
        assert hasattr(TrainingConfig, "__version__")
        assert hasattr(DataConfig, "__version__")
        assert hasattr(InferenceConfig, "__version__")

        assert isinstance(ModelConfig.__version__, int)
        assert isinstance(TrainingConfig.__version__, int)
        assert isinstance(DataConfig.__version__, int)
        assert isinstance(InferenceConfig.__version__, int)

    def test_checkpoint_includes_config_versions(self, tmp_path):
        """save_checkpoint should include config versions."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        checkpoint = torch.load(checkpoint_path)
        assert "config_versions" in checkpoint
        versions = checkpoint["config_versions"]
        assert "model" in versions
        assert "training" in versions
        assert "data" in versions
        assert "inference" in versions

        assert versions["model"] == ModelConfig.__version__
        assert versions["training"] == TrainingConfig.__version__
        assert versions["data"] == DataConfig.__version__
        assert versions["inference"] == InferenceConfig.__version__

    def test_load_checkpoint_validates_versions(self, tmp_path):
        """load_checkpoint should validate config versions."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        # Should load successfully with matching versions
        model2 = DecoderLM.from_config(config, attention_backend="standard")
        optimizer2 = torch.optim.Adam(model2.parameters())
        step = load_checkpoint(checkpoint_path, model2, optimizer2, strict_version_check=True)
        assert step == 100

    def test_load_checkpoint_version_mismatch_raises(self, tmp_path):
        """load_checkpoint should raise on version mismatch."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        # Manually modify checkpoint to simulate version mismatch
        checkpoint = torch.load(checkpoint_path)
        checkpoint["config_versions"]["model"] = 999  # Invalid version
        torch.save(checkpoint, checkpoint_path)

        # Should raise on version mismatch
        model2 = DecoderLM.from_config(config, attention_backend="standard")
        optimizer2 = torch.optim.Adam(model2.parameters())
        with pytest.raises(ValueError, match="Config version mismatch"):
            load_checkpoint(checkpoint_path, model2, optimizer2, strict_version_check=True)

    def test_load_checkpoint_version_mismatch_warns_if_not_strict(self, tmp_path):
        """load_checkpoint should warn but not raise if strict=False."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        # Manually modify checkpoint
        checkpoint = torch.load(checkpoint_path)
        checkpoint["config_versions"]["model"] = 999
        torch.save(checkpoint, checkpoint_path)

        # Should warn but not raise
        model2 = DecoderLM.from_config(config, attention_backend="standard")
        optimizer2 = torch.optim.Adam(model2.parameters())
        with pytest.warns(UserWarning, match="Config version mismatch"):
            step = load_checkpoint(checkpoint_path, model2, optimizer2, strict_version_check=False)
        assert step == 100

    def test_validate_checkpoint_config_compatibility(self, tmp_path):
        """validate_checkpoint_config_compatibility should validate versions."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        # Should pass with matching versions
        versions = validate_checkpoint_config_compatibility(checkpoint_path, strict=True)
        assert versions["model"] == ModelConfig.__version__

    def test_validate_checkpoint_config_compatibility_raises_on_mismatch(self, tmp_path):
        """validate_checkpoint_config_compatibility should raise on mismatch."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        # Manually modify checkpoint
        checkpoint = torch.load(checkpoint_path)
        checkpoint["config_versions"]["training"] = 999
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ConfigVersionMismatchError, match="Config version mismatch"):
            validate_checkpoint_config_compatibility(checkpoint_path, strict=True)

    def test_get_checkpoint_config_versions(self, tmp_path):
        """get_checkpoint_config_versions should return version dict."""
        config = build_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = save_checkpoint(model, optimizer, step=100, output_dir=tmp_path)

        versions = get_checkpoint_config_versions(checkpoint_path)
        assert versions is not None
        assert "model" in versions
        assert versions["model"] == ModelConfig.__version__

    def test_old_checkpoint_without_versions_warns(self, tmp_path):
        """Checkpoint without config_versions should warn."""
        # Create old-style checkpoint without versions
        checkpoint_path = tmp_path / "old_checkpoint.pt"
        torch.save(
            {
                "model_state": {},
                "optimizer_state": {},
                "step": 50,
            },
            checkpoint_path,
        )

        # Should warn about missing version info
        with pytest.warns(UserWarning, match="before config versioning"):
            validate_checkpoint_config_compatibility(checkpoint_path, strict=False)
