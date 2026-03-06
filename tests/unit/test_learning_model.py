"""Unit tests for BaseLearningModel and DecoderLM."""

import io

import pytest
import torch

from src.config.model import ModelConfig
from src.models.learning_model import BaseLearningModel, DecoderLM


def make_model_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "model_type": "decoder_lm",
        "hidden_size": 128,
        "num_layers": 1,
        "num_heads": 4,
        "vocab_size": 128,
        "max_seq_length": 256,
        "mla_latent_dim": 64,
        "rope_base": 10000,
        "intermediate_size": None,
        "num_experts": 1,
        "experts_per_token": 1,
        "moe_frequency": 0,
        "gru_hidden_size": None,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


class TestBaseLearningModel:
    """Contract tests for the abstract base interface."""

    def test_cannot_instantiate_directly(self):
        """BaseLearningModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLearningModel()  # type: ignore[abstract]

    def test_decoder_lm_is_base_learning_model(self):
        """DecoderLM satisfies the BaseLearningModel contract."""
        model = DecoderLM.from_config(make_model_config(), attention_backend="standard")
        assert isinstance(model, BaseLearningModel)


class TestDecoderLMConstruction:
    """DecoderLM construction and configuration."""

    def test_from_config(self):
        """from_config constructs a DecoderLM with correct dimensions."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        assert isinstance(model, DecoderLM)

    def test_weight_tying(self):
        """LM head weight is the same tensor as the token embedding weight."""
        model = DecoderLM.from_config(make_model_config(), attention_backend="standard")
        assert model.lm_head.weight is model.token_embedding.embedding.weight

    def test_lm_head_no_bias(self):
        """LM head has no bias (standard for weight-tied heads)."""
        model = DecoderLM.from_config(make_model_config(), attention_backend="standard")
        assert model.lm_head.bias is None

    def test_weight_tying_survives_checkpoint_roundtrip(self):
        """Weight tying is re-established after save/load of state_dict."""
        model = DecoderLM.from_config(make_model_config(), attention_backend="standard")
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        model2 = DecoderLM.from_config(make_model_config(), attention_backend="standard")
        model2.load_state_dict(torch.load(buf, weights_only=True))
        assert model2.lm_head.weight is model2.token_embedding.embedding.weight

    def test_parameter_count_positive(self):
        """Model has a positive, finite parameter count."""
        model = DecoderLM.from_config(make_model_config(), attention_backend="standard")
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


class TestDecoderLMForward:
    """DecoderLM forward pass shape and dtype contracts."""

    def test_output_shape(self):
        """Forward output shape is (batch_size, seq_len, vocab_size)."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert model(x).shape == (2, 16, config.vocab_size)

    def test_output_dtype(self):
        """Forward output is float32."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 8))
        assert model(x).dtype == torch.float32

    def test_batch_size_one(self):
        """Forward works with batch size 1."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 32))
        assert model(x).shape == (1, 32, config.vocab_size)

    def test_various_batch_sizes(self):
        """Output shape scales correctly with batch size."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        for B in [1, 4, 16]:
            x = torch.randint(0, config.vocab_size, (B, 16))
            assert model(x).shape == (B, 16, config.vocab_size)

    def test_various_seq_lengths(self):
        """Output shape scales correctly with sequence length."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        for T in [1, 16, 64, 128]:
            x = torch.randint(0, config.vocab_size, (2, T))
            assert model(x).shape == (2, T, config.vocab_size)

    def test_seq_length_at_max(self):
        """Forward works at the maximum configured sequence length."""
        config = make_model_config(max_seq_length=64)
        model = DecoderLM.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 64))
        assert model(x).shape == (1, 64, config.vocab_size)

    def test_output_is_finite(self):
        """Forward output contains no NaN or Inf values."""
        config = make_model_config()
        model = DecoderLM.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert torch.isfinite(model(x)).all()

    def test_seq_len_exceeds_max_raises(self):
        """Forward raises AssertionError when seq_len > max_seq_length."""
        config = make_model_config(max_seq_length=16)
        model = DecoderLM.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 17))
        with pytest.raises(AssertionError, match="seq_len 17 exceeds max_seq_len 16"):
            model(x)
