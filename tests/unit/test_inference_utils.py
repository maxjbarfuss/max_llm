"""Tests for inference utilities (device resolution, checkpoint loading, tokenizer creation)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.config.experiment import DataConfig, ExperimentConfig
from src.inference.utils import (
    create_tokenizer_from_data_config,
    load_checkpoint_into_model,
    resolve_device,
)
from src.models.learning_model import SimpleLM


class TestResolveDevice:
    """Test device resolution logic."""

    def test_cpu_device(self):
        """Test CPU device resolution."""
        device = resolve_device("cpu")
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_device_with_cuda(self, mock_cuda):
        """Test auto device resolution when CUDA is available."""
        device = resolve_device("auto")
        assert device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_device_without_cuda(self, mock_cuda):
        """Test auto device resolution when CUDA is not available."""
        device = resolve_device("auto")
        assert device.type == "cpu"

    def test_cuda_device(self):
        """Test explicit CUDA device resolution."""
        device = resolve_device("cuda")
        assert device.type == "cuda"


class TestLoadCheckpointIntoModel:
    """Test checkpoint loading with various formats."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        config = ExperimentConfig.from_toml("config/experiment.toml")
        return SimpleLM.from_config(config.model)

    def test_load_checkpoint_with_model_state_key(self, model):
        """Test loading checkpoint with 'model_state' key."""
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"model_state": state_dict}, checkpoint_path)

        try:
            load_checkpoint_into_model(model, checkpoint_path, torch.device("cpu"))
            # If no exception, loading succeeded
            assert True
        finally:
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_with_state_dict_key(self, model):
        """Test loading checkpoint with 'state_dict' key."""
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"state_dict": state_dict}, checkpoint_path)

        try:
            load_checkpoint_into_model(model, checkpoint_path, torch.device("cpu"))
            assert True
        finally:
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_with_model_key(self, model):
        """Test loading checkpoint with 'model' key."""
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"model": state_dict}, checkpoint_path)

        try:
            load_checkpoint_into_model(model, checkpoint_path, torch.device("cpu"))
            assert True
        finally:
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_direct_state_dict(self, model):
        """Test loading checkpoint that is directly a state_dict."""
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save(state_dict, checkpoint_path)

        try:
            load_checkpoint_into_model(model, checkpoint_path, torch.device("cpu"))
            assert True
        finally:
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_with_extra_metadata(self, model):
        """Test loading checkpoint with extra metadata (step, optimizer, etc)."""
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save(
                {
                    "model_state": state_dict,
                    "step": 1000,
                    "optimizer_state": {},
                    "config": {},
                },
                checkpoint_path,
            )

        try:
            load_checkpoint_into_model(model, checkpoint_path, torch.device("cpu"))
            assert True
        finally:
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_unsupported_format_raises(self, model):
        """Test that unsupported checkpoint format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            # Save a non-dict checkpoint (e.g., just a tensor)
            torch.save(torch.tensor([1, 2, 3]), checkpoint_path)

        try:
            with pytest.raises(ValueError, match="Unsupported checkpoint format"):
                load_checkpoint_into_model(model, checkpoint_path, torch.device("cpu"))
        finally:
            Path(checkpoint_path).unlink()


class TestCreateTokenizerFromDataConfig:
    """Test tokenizer creation from data config."""

    def test_create_utf8_tokenizer(self):
        """Test creating UTF-8 tokenizer."""
        config = ExperimentConfig.from_toml("config/experiment.toml")
        tokenizer = create_tokenizer_from_data_config(config.data)

        # Should be able to encode/decode
        tokens = tokenizer.encode("Hello")
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    def test_create_codepoint_tokenizer(self):
        """Test creating codepoint tokenizer with vocab_size."""
        # Create a mock data config with codepoint mode
        data_config = Mock(spec=DataConfig)
        data_config.tokenizer_name = "char"
        data_config.tokenizer_mode = "codepoint"
        data_config.tokenizer_vocab_size = 512

        tokenizer = create_tokenizer_from_data_config(data_config)

        # Should handle basic text
        tokens = tokenizer.encode("Test")
        assert len(tokens) > 0

    def test_create_utf16_tokenizer(self):
        """Test creating UTF-16 tokenizer."""
        data_config = Mock(spec=DataConfig)
        data_config.tokenizer_name = "char"
        data_config.tokenizer_mode = "utf16"

        tokenizer = create_tokenizer_from_data_config(data_config)

        tokens = tokenizer.encode("Hello")
        assert len(tokens) > 0

    def test_create_utf32_tokenizer(self):
        """Test creating UTF-32 tokenizer."""
        data_config = Mock(spec=DataConfig)
        data_config.tokenizer_name = "char"
        data_config.tokenizer_mode = "utf32"

        tokenizer = create_tokenizer_from_data_config(data_config)

        tokens = tokenizer.encode("Test")
        assert len(tokens) > 0
