"""Unit tests for BaseLearningModel and SimpleLM."""

import io

import pytest
import torch

from src.config.model import ModelConfig
from src.models.learning_model import BaseLearningModel, SimpleLM


def make_p2_model_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
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

    def test_simple_lm_is_base_learning_model(self):
        """SimpleLM satisfies the BaseLearningModel contract."""
        model = SimpleLM.from_config(make_p2_model_config())
        assert isinstance(model, BaseLearningModel)


class TestSimpleLMConstruction:
    """SimpleLM construction and configuration."""

    def test_from_config(self):
        """from_config constructs a SimpleLM with correct dimensions."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        assert isinstance(model, SimpleLM)

    def test_weight_tying(self):
        """LM head weight is the same tensor as the token embedding weight."""
        model = SimpleLM.from_config(make_p2_model_config())
        assert model.lm_head.weight is model.token_emb.weight

    def test_lm_head_no_bias(self):
        """LM head has no bias (standard for weight-tied heads)."""
        model = SimpleLM.from_config(make_p2_model_config())
        assert model.lm_head.bias is None

    def test_weight_tying_survives_checkpoint_roundtrip(self):
        """Weight tying is re-established after save/load of state_dict."""
        model = SimpleLM.from_config(make_p2_model_config())
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        model2 = SimpleLM.from_config(make_p2_model_config())
        model2.load_state_dict(torch.load(buf, weights_only=True))
        assert model2.lm_head.weight is model2.token_emb.weight

    def test_parameter_count(self):
        """Unique parameter count matches expected value (weight tying not double-counted)."""
        config = make_p2_model_config(vocab_size=128, hidden_size=128, max_seq_length=256)
        model = SimpleLM.from_config(config)
        # token_emb: V*H, pos_emb: S*H, ffn.weight: H*H, ffn.bias: H
        # lm_head.weight is tied to token_emb.weight — not counted again
        V, H, S = 128, 128, 256
        expected = V * H + S * H + H * H + H
        actual = sum(p.numel() for p in model.parameters())
        assert actual == expected


class TestSimpleLMForward:
    """SimpleLM forward pass shape and dtype contracts."""

    def test_output_shape(self):
        """Forward output shape is (batch_size, seq_len, vocab_size)."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert model(x).shape == (2, 16, config.vocab_size)

    def test_output_dtype(self):
        """Forward output is float32."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        x = torch.randint(0, config.vocab_size, (1, 8))
        assert model(x).dtype == torch.float32

    def test_batch_size_one(self):
        """Forward works with batch size 1."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        x = torch.randint(0, config.vocab_size, (1, 32))
        assert model(x).shape == (1, 32, config.vocab_size)

    def test_various_batch_sizes(self):
        """Output shape scales correctly with batch size."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        for B in [1, 4, 16]:
            x = torch.randint(0, config.vocab_size, (B, 16))
            assert model(x).shape == (B, 16, config.vocab_size)

    def test_various_seq_lengths(self):
        """Output shape scales correctly with sequence length."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        for T in [1, 16, 64, 128]:
            x = torch.randint(0, config.vocab_size, (2, T))
            assert model(x).shape == (2, T, config.vocab_size)

    def test_seq_length_at_max(self):
        """Forward works at the maximum configured sequence length."""
        config = make_p2_model_config(max_seq_length=64)
        model = SimpleLM.from_config(config)
        x = torch.randint(0, config.vocab_size, (1, 64))
        assert model(x).shape == (1, 64, config.vocab_size)

    def test_output_is_finite(self):
        """Forward output contains no NaN or Inf values."""
        config = make_p2_model_config()
        model = SimpleLM.from_config(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert torch.isfinite(model(x)).all()

    def test_seq_len_exceeds_max_raises(self):
        """Forward raises AssertionError when seq_len > max_seq_length."""
        config = make_p2_model_config(max_seq_length=16)
        model = SimpleLM.from_config(config)
        x = torch.randint(0, config.vocab_size, (1, 17))
        with pytest.raises(AssertionError, match="seq_len 17 exceeds max_seq_length 16"):
            model(x)
