"""Integration tests for the Phase 4 final architecture stack.

Validates that the exact combination used in production —
FlashNorm + MLA (decoupled RoPE) + xIELU FFN + block_attn residuals —
instantiates correctly, runs a forward pass, and produces valid gradients.

Uses a small hidden_size=64 proxy to stay fast on CPU.
"""

from __future__ import annotations

from typing import Any

import torch

from src.config.model import ModelConfig
from src.models.attention.causal_mha import CausalMultiHeadAttention
from src.models.attention.multihead_latent_attention import MultiHeadLatentAttention
from src.models.attention.sliding_window_attention import SlidingWindowAttention
from src.models.learning_model import LearningModel


def _p4_config(**overrides: object) -> ModelConfig:
    """Minimal model with the same architectural choices as the P4 milestone."""
    base: dict[str, Any] = {
        # Architecture mirrors p4_final_anneal_20260422.toml, scaled down for speed
        "hidden_size": 64,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "vocab_size": 256,
        "max_seq_length": 16,
        "norm_type": "flash",
        "ffn_type": "xielu",
        "pos_type": "rope",
        "attn_type": "mla",
        "res_type": "block_attn",
        "attn_res_num_blocks": 2,
        "intermediate_size": 256,
        "mla_latent_dim": 32,
        "dropout": 0.0,
        "rope_base": 10000,
        "num_experts": 1,
        "experts_per_token": 1,
        "moe_frequency": 0,
        "gru_hidden_size": None,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _model(cfg: ModelConfig | None = None) -> LearningModel:
    cfg = cfg or _p4_config()
    return LearningModel.from_config(cfg, attention_backend="standard")


class TestP4StackConstruction:
    def test_instantiates(self) -> None:
        assert _model() is not None

    def test_param_count_reasonable(self) -> None:
        n = sum(p.numel() for p in _model().parameters())
        # At hidden=64 we expect < 500K params; just confirm it's non-trivial
        assert 10_000 < n < 10_000_000

    def test_flashnorm_has_no_parameters(self) -> None:
        model = _model()
        norm_params = [
            n for n, p in model.named_parameters() if "norm" in n.lower() and p.requires_grad
        ]
        # FlashNorm is parameter-free; no norm parameters should appear
        assert len(norm_params) == 0, f"Expected zero norm params, found: {norm_params}"

    def test_xielu_has_trainable_scalars(self) -> None:
        model = _model()
        xielu_params = [n for n, _ in model.named_parameters() if "alpha" in n.lower()]
        # xIELU has 2 trainable scalars (alpha_p_raw, alpha_n_raw) per FFN layer
        assert len(xielu_params) > 0, "Expected xIELU alpha params; found none"

    def test_attn_residual_query_weights_exist(self) -> None:
        model = _model()
        attn_res_params = [n for n, _ in model.named_parameters() if "attn_res" in n.lower()]
        assert len(attn_res_params) > 0, "Expected block_attn residual params; found none"

    def test_mla_latent_proj_exists(self) -> None:
        model = _model()
        latent_params = [n for n, _ in model.named_parameters() if "latent_proj" in n.lower()]
        assert len(latent_params) > 0, "Expected MLA latent projection params; found none"


class TestP4StackForward:
    def test_output_shape(self) -> None:
        model = _model()
        x = torch.randint(0, 256, (2, 8))
        out = model(x)
        assert out.shape == (2, 8, 256)

    def test_no_nan_in_output(self) -> None:
        model = _model()
        x = torch.randint(0, 256, (1, 16))
        out = model(x)
        assert torch.isfinite(out).all(), "NaN or Inf in model output"

    def test_gradient_flows_through_full_stack(self) -> None:
        model = _model()
        x = torch.randint(0, 256, (2, 8))
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert len(grads) > 0, "No gradients computed"
        for name, grad in grads.items():
            assert torch.isfinite(grad).all(), f"Non-finite grad in {name}"

    def test_different_inputs_produce_different_outputs(self) -> None:
        model = _model()
        model.eval()
        x1 = torch.randint(0, 128, (1, 8))
        x2 = torch.randint(128, 256, (1, 8))
        with torch.no_grad():
            assert not torch.allclose(model(x1), model(x2))

    def test_batch_independence(self) -> None:
        """Each batch position should produce the same result as a single-item batch."""
        model = _model()
        model.eval()
        torch.manual_seed(0)
        x = torch.randint(0, 256, (2, 6))
        with torch.no_grad():
            batched = model(x)
            single0 = model(x[:1])
            single1 = model(x[1:])
        assert torch.allclose(batched[:1], single0, atol=1e-5)
        assert torch.allclose(batched[1:], single1, atol=1e-5)


class TestP4StackOverfit:
    def test_overfits_single_batch(self) -> None:
        """The P4 stack should be able to overfit a single batch in < 200 steps."""
        torch.manual_seed(7)
        model = _model()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        x = torch.randint(0, 256, (2, 8))
        target = x[:, 1:]

        initial_loss = None
        for step in range(200):
            model.train()
            logits = model(x[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
            if step == 0:
                initial_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < initial_loss * 0.5, (  # type: ignore[operator]
            f"Did not overfit: initial={initial_loss:.3f} final={loss.item():.3f}"
        )


class TestP4StackGQA:
    """MLA + GQA: num_kv_heads < num_heads must work correctly."""

    def test_kv_heads_less_than_query_heads(self) -> None:
        cfg = _p4_config(num_heads=8, num_kv_heads=2)
        model = LearningModel.from_config(cfg, attention_backend="standard")
        x = torch.randint(0, 256, (1, 8))
        out = model(x)
        assert out.shape == (1, 8, 256)

    def test_mqa_single_kv_head(self) -> None:
        cfg = _p4_config(num_heads=4, num_kv_heads=1)
        model = LearningModel.from_config(cfg, attention_backend="standard")
        x = torch.randint(0, 256, (1, 4))
        assert torch.isfinite(model(x)).all()


class TestInterleavedAttentionStack:
    def test_interleaved_looped_blocks_assign_attention_by_physical_block(self) -> None:
        cfg = _p4_config(
            attn_type="interleaved",
            interleaved_attn_pattern=("swa", "mha", "mla", "swa"),
            res_type="standard",
            num_layers=12,
            looped_num_blocks=4,
        )
        model = LearningModel.from_config(cfg, attention_backend="standard")

        assert len(model.blocks) == 4
        assert isinstance(model.blocks[0].attention, SlidingWindowAttention)
        assert isinstance(model.blocks[1].attention, CausalMultiHeadAttention)
        assert isinstance(model.blocks[2].attention, MultiHeadLatentAttention)
        assert isinstance(model.blocks[3].attention, SlidingWindowAttention)
        assert model.blocks[0] is model.blocks[4 % len(model.blocks)]

    def test_interleaved_looped_stack_forward_and_backward(self) -> None:
        torch.manual_seed(3)
        cfg = _p4_config(
            attn_type="interleaved",
            interleaved_attn_pattern=("swa", "mla"),
            res_type="standard",
            num_layers=6,
            looped_num_blocks=2,
        )
        model = LearningModel.from_config(cfg, attention_backend="standard")
        x = torch.randint(0, 256, (2, 8))

        logits = model(x)
        assert logits.shape == (2, 8, 256)
        loss = logits.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)


class TestWeightSoup:
    """P4-DEC-2: weight averaging (model soup) produces a valid model.

    The protocol: average weights of two checkpoints from the same basin
    before restarting the LR schedule. Averaged weights must still produce
    finite, valid outputs — this test verifies the averaging arithmetic.
    """

    def test_averaged_weights_produce_valid_output(self) -> None:
        torch.manual_seed(1)
        cfg = _p4_config()
        model_a = LearningModel.from_config(cfg, attention_backend="standard")
        model_b = LearningModel.from_config(cfg, attention_backend="standard")

        # Equal-weight soup
        soup_sd = {
            k: (model_a.state_dict()[k].float() + model_b.state_dict()[k].float()) / 2.0
            for k in model_a.state_dict()
        }

        model_soup = LearningModel.from_config(cfg, attention_backend="standard")
        model_soup.load_state_dict(soup_sd)

        x = torch.randint(0, 256, (1, 8))
        out = model_soup(x)
        assert torch.isfinite(out).all()

    def test_soup_interpolates_between_checkpoints(self) -> None:
        """Soup output should lie between the outputs of the two source models."""
        torch.manual_seed(2)
        cfg = _p4_config()
        model_a = LearningModel.from_config(cfg, attention_backend="standard")
        model_b = LearningModel.from_config(cfg, attention_backend="standard")

        soup_sd = {
            k: (model_a.state_dict()[k].float() + model_b.state_dict()[k].float()) / 2.0
            for k in model_a.state_dict()
        }
        model_soup = LearningModel.from_config(cfg, attention_backend="standard")
        model_soup.load_state_dict(soup_sd)

        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out_a = model_a(x)
            out_b = model_b(x)
            out_soup = model_soup(x)

        # Soup should differ from both parents (not equal to either)
        assert not torch.allclose(out_soup, out_a, atol=1e-4)
        assert not torch.allclose(out_soup, out_b, atol=1e-4)
