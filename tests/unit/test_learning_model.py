"""Unit tests for LearningModel."""

import io
from copy import deepcopy
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from src.config.model import ModelConfig
from src.models.depth_router import TokenDepthRouter
from src.models.learning_model import LearningModel


def make_model_config(**overrides: object) -> ModelConfig:
    values: dict[str, Any] = {
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
    return ModelConfig(**cast(Any, values))


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


class TestLoopedAttention:
    """Looped block execution (looped_num_blocks)."""

    def test_default_creates_one_block_per_layer(self):
        """Without looped_num_blocks, model has num_layers physical blocks."""
        config = make_model_config(num_layers=4)
        model = LearningModel.from_config(config, attention_backend="standard")
        assert len(model.blocks) == 4

    def test_looped_creates_k_physical_blocks(self):
        """looped_num_blocks=K creates exactly K physical blocks."""
        config = make_model_config(num_layers=6, looped_num_blocks=2)
        model = LearningModel.from_config(config, attention_backend="standard")
        assert len(model.blocks) == 2
        assert model.num_layers == 6

    def test_looped_num_blocks_stored(self):
        """looped_num_blocks is stored on the model."""
        config = make_model_config(num_layers=6, looped_num_blocks=3)
        model = LearningModel.from_config(config, attention_backend="standard")
        assert model.looped_num_blocks == 3

    def test_block_for_layer_cyclic_mapping(self):
        """_block_for_layer returns blocks cyclically for looped models."""
        config = make_model_config(num_layers=6, looped_num_blocks=2)
        model = LearningModel.from_config(config, attention_backend="standard")
        # Even logical layers → physical block 0; odd → physical block 1
        assert model._block_for_layer(0) is model.blocks[0]
        assert model._block_for_layer(1) is model.blocks[1]
        assert model._block_for_layer(2) is model.blocks[0]
        assert model._block_for_layer(3) is model.blocks[1]
        assert model._block_for_layer(4) is model.blocks[0]
        assert model._block_for_layer(5) is model.blocks[1]

    def test_share_layer_weights_is_looped_num_blocks_1(self):
        """share_layer_weights=True behaves identically to looped_num_blocks=1."""
        torch.manual_seed(0)
        config_shared = make_model_config(num_layers=4, share_layer_weights=True)
        config_looped = make_model_config(num_layers=4, looped_num_blocks=1)
        shared = LearningModel.from_config(config_shared, attention_backend="standard")
        looped = LearningModel.from_config(config_looped, attention_backend="standard")

        # Both have a single physical block
        assert len(shared.blocks) == 1
        assert len(looped.blocks) == 1

        # Copy shared weights into looped so we can compare forward passes
        looped.load_state_dict(shared.state_dict())
        x = torch.randint(0, config_shared.vocab_size, (2, 16))
        torch.testing.assert_close(shared(x), looped(x))

    def test_looped_forward_output_shape(self):
        """Looped model forward produces correct (B, T, vocab_size) shape."""
        config = make_model_config(num_layers=6, looped_num_blocks=2)
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        assert model(x).shape == (2, 16, config.vocab_size)

    def test_looped_kv_cache_has_num_layers_slots(self):
        """make_kv_cache returns num_layers logical slots, not K physical blocks."""
        config = make_model_config(
            num_layers=6, looped_num_blocks=2, rope_base=10000, pos_type="rope"
        )
        model = LearningModel.from_config(config, attention_backend="standard")
        cache = model.make_kv_cache(batch_size=1)
        assert len(cache) == 6

    @pytest.mark.parametrize(
        "res_type,attn_res_num_blocks",
        [("standard", 1), ("full_attn", 1), ("block_attn", 4)],
    )
    def test_looped_all_residual_paths(self, res_type: str, attn_res_num_blocks: int):
        """All residual variants produce valid output with looped blocks."""
        config = make_model_config(
            hidden_size=64,
            num_layers=8,
            num_heads=4,
            max_seq_length=32,
            res_type=res_type,
            attn_res_num_blocks=attn_res_num_blocks,
            looped_num_blocks=4,
        )
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, config.vocab_size)
        assert torch.isfinite(out).all()

    def test_looped_gradients_flow(self):
        """Looped model gradients flow to physical block parameters."""
        torch.manual_seed(7)
        config = make_model_config(num_layers=6, looped_num_blocks=2)
        model = LearningModel.from_config(config, attention_backend="standard")
        model.train()
        x = torch.randint(0, config.vocab_size, (2, 16))
        loss = model(x).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestMixtureOfDepthsRouter:
    """Mixture-of-Depths token router behaviour."""

    def test_router_selects_capacity_fraction_for_ffn_delta(self):
        """Router computes FFN deltas for exactly the selected token budget."""
        router = TokenDepthRouter(d_model=4, capacity_fraction=0.5, min_tokens=0)
        with torch.no_grad():
            router.score.weight.zero_()
            router.score.bias.zero_()

        x = torch.zeros(2, 4, 4)
        delta = router.apply_ffn_delta(x, nn.Identity(), lambda tensor: torch.ones_like(tensor))

        assert delta.shape == x.shape
        assert torch.count_nonzero(delta).item() == 2 * 2 * 4
        assert router.last_selected_fraction == 0.5

    def test_eval_threshold_can_skip_all_tokens(self):
        """Inference threshold permits true layer skipping when min_tokens=0."""
        router = TokenDepthRouter(
            d_model=4,
            capacity_fraction=1.0,
            min_tokens=0,
            inference_threshold=0.75,
        )
        with torch.no_grad():
            router.score.weight.zero_()
            router.score.bias.zero_()
        router.eval()

        x = torch.ones(1, 3, 4)
        out = router.apply_ffn_residual(x, nn.Identity(), lambda t: t * 2)

        torch.testing.assert_close(out, x)
        assert router.last_selected_fraction == 0.0

    def test_router_casts_float32_ffn_delta_to_input_dtype(self):
        """Routed scatter supports mixed precision FFNs that return fp32 deltas."""
        router = TokenDepthRouter(d_model=4, capacity_fraction=1.0)
        with torch.no_grad():
            router.score.weight.zero_()
            router.score.bias.zero_()

        x = torch.ones(2, 3, 4, dtype=torch.bfloat16)

        def fp32_feedforward(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.float() * 2.0

        delta = router.apply_ffn_delta(x, nn.Identity(), fp32_feedforward)

        assert delta.dtype == torch.bfloat16
        torch.testing.assert_close(delta, torch.full_like(x, 2.0))

    def test_router_selection_fraction_is_diagnostic_only(self):
        """Selection fraction updates in eager mode without affecting routed output."""
        router = TokenDepthRouter(d_model=4, capacity_fraction=0.25, min_tokens=0)
        with torch.no_grad():
            router.score.weight.zero_()
            router.score.bias.zero_()

        x = torch.zeros(2, 4, 4)
        delta = router.apply_ffn_delta(x, nn.Identity(), lambda tensor: torch.ones_like(tensor))

        assert router.last_selected_fraction == 0.25
        assert torch.count_nonzero(delta).item() == 2 * 1 * 4

    def test_model_builds_configured_router_layers(self):
        """LearningModel creates routers only for configured logical layers."""
        config = make_model_config(
            num_layers=6,
            mod_router_enabled=True,
            mod_router_start_layer=1,
            mod_router_frequency=2,
        )
        model = LearningModel.from_config(config, attention_backend="standard")
        assert set(model.depth_routers.keys()) == {"1", "3", "5"}

    def test_model_router_receives_gradients(self):
        """Soft-gated routing keeps router scores trainable on selected tokens."""
        torch.manual_seed(9)
        config = make_model_config(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_length=32,
            mod_router_enabled=True,
            mod_router_capacity_fraction=0.5,
        )
        model = LearningModel.from_config(config, attention_backend="standard")
        x = torch.randint(0, config.vocab_size, (2, 16))
        loss = model(x).sum()
        loss.backward()

        first_router = model.depth_routers["0"]
        assert isinstance(first_router, TokenDepthRouter)
        assert first_router.score.weight.grad is not None
        assert torch.count_nonzero(first_router.score.weight.grad).item() > 0

    def test_mod_router_all_residual_paths_produce_finite_outputs(self):
        """MoD FFN routing composes with the standard attn-residual variants."""
        for res_type, attn_res_num_blocks in [
            ("standard", 1),
            ("full_attn", 1),
            ("block_attn", 4),
        ]:
            config = make_model_config(
                hidden_size=64,
                num_layers=4,
                num_heads=4,
                max_seq_length=32,
                res_type=res_type,
                attn_res_num_blocks=attn_res_num_blocks,
                mod_router_enabled=True,
                mod_router_capacity_fraction=0.5,
            )
            model = LearningModel.from_config(config, attention_backend="standard")
            x = torch.randint(0, config.vocab_size, (2, 16))
            out = model(x)
            assert out.shape == (2, 16, config.vocab_size)
            assert torch.isfinite(out).all()
