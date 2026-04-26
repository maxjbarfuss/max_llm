"""Unit tests for LearningModel."""

import io
from copy import deepcopy

import pytest
import torch

from src.config.model import ModelConfig
from src.models.learning_model import LearningModel


def make_model_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "hidden_size": 128,
        "num_layers": 1,
        "num_heads": 4,
        "vocab_size": 128,
        "max_seq_length": 256,
        "mla_latent_dim": 64,
        "rope_base": None,
        "intermediate_size": None,
        "num_experts": 1,
        "experts_per_token": 1,
        "moe_frequency": 0,
        "gru_hidden_size": None,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


class TestLearningModelConstruction:
    """LearningModel construction and configuration."""

    def test_from_config(self):
        """from_config constructs a LearningModel with correct dimensions."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        assert isinstance(model, LearningModel)

    def test_weight_tying(self):
        """LM head weight is the same tensor as the token embedding weight."""
        model = LearningModel.from_config(make_model_config(), attention_backend="standard")
        assert model.lm_head.weight is model.token_embedding.embedding.weight

    def test_lm_head_no_bias(self):
        """LM head has no bias (standard for weight-tied heads)."""
        model = LearningModel.from_config(make_model_config(), attention_backend="standard")
        assert model.lm_head.bias is None

    def test_weight_tying_survives_checkpoint_roundtrip(self):
        """Weight tying is re-established after save/load of state_dict."""
        model = LearningModel.from_config(make_model_config(), attention_backend="standard")
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        model2 = LearningModel.from_config(make_model_config(), attention_backend="standard")
        model2.load_state_dict(torch.load(buf, weights_only=True))
        assert model2.lm_head.weight is model2.token_embedding.embedding.weight

    def test_parameter_count_positive(self):
        """Model has a positive, finite parameter count."""
        model = LearningModel.from_config(make_model_config(), attention_backend="standard")
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


class TestLearningModelForward:
    """LearningModel forward pass shape and dtype contracts."""

    def test_output_shape(self):
        """Forward output shape is (batch_size, seq_len, vocab_size)."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert model(x).shape == (2, 16, config.vocab_size)

    def test_output_dtype(self):
        """Forward output is float32."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 8))
        assert model(x).dtype == torch.float32

    def test_batch_size_one(self):
        """Forward works with batch size 1."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 32))
        assert model(x).shape == (1, 32, config.vocab_size)

    def test_various_batch_sizes(self):
        """Output shape scales correctly with batch size."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        for B in [1, 4, 16]:
            x = torch.randint(0, config.vocab_size, (B, 16))
            assert model(x).shape == (B, 16, config.vocab_size)

    def test_various_seq_lengths(self):
        """Output shape scales correctly with sequence length."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        for T in [1, 16, 64, 128]:
            x = torch.randint(0, config.vocab_size, (2, T))
            assert model(x).shape == (2, T, config.vocab_size)

    def test_seq_length_at_max(self):
        """Forward works at the maximum configured sequence length."""
        config = make_model_config(max_seq_length=64)
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 64))
        assert model(x).shape == (1, 64, config.vocab_size)

    def test_output_is_finite(self):
        """Forward output contains no NaN or Inf values."""
        config = make_model_config()
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert torch.isfinite(model(x)).all()

    def test_seq_len_exceeds_max_raises(self):
        """Forward raises AssertionError when seq_len > max_seq_length."""
        config = make_model_config(max_seq_length=16)
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (1, 17))
        with pytest.raises(AssertionError, match="seq_len 17 exceeds max_seq_len 16"):
            model(x)


class TestLearningModelGradientCheckpointing:
    """Activation checkpointing API behaviour."""

    @pytest.mark.parametrize("mode", ["full", "ffn"])
    def test_checkpointing_matches_uncheckpointed_forward_and_gradients(self, mode: str):
        """Checkpointing should preserve outputs and gradients for deterministic blocks."""
        torch.manual_seed(123)
        config = make_model_config(hidden_size=64, num_layers=2, max_seq_length=32)
        baseline = LearningModel.from_config(config, attention_backend="standard")
        checkpointed = deepcopy(baseline)
        checkpointed.gradient_checkpointing_enable(mode=mode)
        baseline.train()
        checkpointed.train()

        x = torch.randint(0, config.vocab_size, (2, 16))
        baseline_loss = baseline(x).sum()
        checkpointed_loss = checkpointed(x).sum()

        torch.testing.assert_close(checkpointed_loss, baseline_loss)
        baseline_loss.backward()
        checkpointed_loss.backward()

        baseline_grad = baseline.token_embedding.embedding.weight.grad
        checkpointed_grad = checkpointed.token_embedding.embedding.weight.grad
        assert baseline_grad is not None
        assert checkpointed_grad is not None
        torch.testing.assert_close(checkpointed_grad, baseline_grad)

    def test_checkpointing_enable_validates_mode_and_interval(self):
        """Invalid checkpointing options fail early."""
        model = LearningModel.from_config(make_model_config(), attention_backend="standard")
        with pytest.raises(ValueError, match="mode must be 'full', 'ffn', or 'attn_res_fused'"):
            model.gradient_checkpointing_enable(mode="attention")
        with pytest.raises(ValueError, match="interval must be >= 1"):
            model.gradient_checkpointing_enable(interval=0)

    def test_checkpointing_disable_clears_flag(self):
        """Checkpointing can be disabled after being enabled."""
        model = LearningModel.from_config(make_model_config(), attention_backend="standard")
        model.gradient_checkpointing_enable()
        assert model.gradient_checkpointing is True
        model.gradient_checkpointing_disable()
        assert model.gradient_checkpointing is False

    @pytest.mark.parametrize("mode", ["full", "ffn"])
    def test_interval_skips_odd_blocks(self, mode: str):
        """interval=2 should checkpoint only even-indexed blocks but produce identical output."""
        torch.manual_seed(7)
        config = make_model_config(hidden_size=64, num_layers=4, max_seq_length=32)
        baseline = LearningModel.from_config(config, attention_backend="standard")
        every2 = deepcopy(baseline)
        every2.gradient_checkpointing_enable(mode=mode, interval=2)
        baseline.train()
        every2.train()

        x = torch.randint(0, config.vocab_size, (2, 16))
        torch.testing.assert_close(every2(x), baseline(x))

    def test_attn_res_fused_matches_baseline_output_and_gradients(self):
        """attn_res_fused on block_attn should produce identical outputs and gradients to baseline."""
        torch.manual_seed(55)
        config = make_model_config(
            hidden_size=64,
            num_layers=8,
            num_heads=4,
            max_seq_length=32,
            res_type="block_attn",
            attn_res_num_blocks=4,
        )
        baseline = LearningModel.from_config(config, attention_backend="standard")
        fused = deepcopy(baseline)
        fused.gradient_checkpointing_enable(mode="attn_res_fused")
        baseline.train()
        fused.train()

        x = torch.randint(0, config.vocab_size, (2, 16))
        baseline_loss = baseline(x).sum()
        fused_loss = fused(x).sum()
        torch.testing.assert_close(fused_loss, baseline_loss)

        baseline_loss.backward()
        fused_loss.backward()
        torch.testing.assert_close(
            fused.token_embedding.embedding.weight.grad,
            baseline.token_embedding.embedding.weight.grad,
        )
        # Also verify attn_res query gradients flow correctly
        assert fused.attn_res is not None
        assert baseline.attn_res is not None
        torch.testing.assert_close(
            fused.attn_res.queries.grad,
            baseline.attn_res.queries.grad,
        )

    def test_attn_res_fused_with_interval(self):
        """attn_res_fused with interval=2 should produce the same output as baseline."""
        torch.manual_seed(77)
        config = make_model_config(
            hidden_size=64,
            num_layers=8,
            num_heads=4,
            max_seq_length=32,
            res_type="block_attn",
            attn_res_num_blocks=4,
        )
        baseline = LearningModel.from_config(config, attention_backend="standard")
        every2 = deepcopy(baseline)
        every2.gradient_checkpointing_enable(mode="attn_res_fused", interval=2)
        baseline.train()
        every2.train()

        x = torch.randint(0, config.vocab_size, (2, 16))
        torch.testing.assert_close(every2(x), baseline(x))

    @pytest.mark.parametrize("res_type", ["full_attn", "block_attn"])
    def test_ffn_mode_attn_residual_paths(self, res_type: str):
        """'ffn' mode on attn-residual paths should produce numerically identical output."""
        torch.manual_seed(42)
        config = make_model_config(
            hidden_size=64,
            num_layers=8,
            num_heads=4,
            max_seq_length=32,
            res_type=res_type,
            attn_res_num_blocks=4,
        )
        baseline = LearningModel.from_config(config, attention_backend="standard")
        ffn_ck = deepcopy(baseline)
        ffn_ck.gradient_checkpointing_enable(mode="ffn")
        baseline.train()
        ffn_ck.train()

        x = torch.randint(0, config.vocab_size, (2, 16))
        baseline_loss = baseline(x).sum()
        ffn_ck_loss = ffn_ck(x).sum()
        torch.testing.assert_close(ffn_ck_loss, baseline_loss)

        baseline_loss.backward()
        ffn_ck_loss.backward()
        torch.testing.assert_close(
            ffn_ck.token_embedding.embedding.weight.grad,
            baseline.token_embedding.embedding.weight.grad,
        )
