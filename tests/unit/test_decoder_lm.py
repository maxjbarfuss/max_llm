"""Tests for decoder-only language model."""

from __future__ import annotations

import pytest
import torch

from src.config.model import ModelConfig
from src.models.learning_model.decoder_lm import DecoderLM
from src.models.position.learned_position import LearnedPositionEmbedding


class TestDecoderLM:
    def test_output_shape_training(self) -> None:
        """Output should be (batch, seq_len, vocab_size) in training mode."""
        model = DecoderLM(
            vocab_size=256,
            d_model=64,
            num_layers=2,
            num_heads=4,
            attention_backend="standard",
        )
        x = torch.randint(0, 256, (2, 8))  # (batch, seq_len) of token IDs
        out = model(x)
        assert out.shape == (2, 8, 256)

    def test_output_dtype_float32(self) -> None:
        """Output should be float32."""
        model = DecoderLM(
            vocab_size=256,
            d_model=128,
            num_layers=2,
            num_heads=8,
            attention_backend="standard",
        )
        x = torch.randint(0, 256, (1, 16))
        out = model(x)
        assert out.dtype == torch.float32

    def test_has_embeddings(self) -> None:
        """Must have token and position embeddings."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        assert hasattr(model, "token_embedding")
        assert hasattr(model, "position_embedding")

    def test_has_transformer_blocks(self) -> None:
        """Must have transformer blocks."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=3, num_heads=4, attention_backend="standard"
        )
        assert hasattr(model, "blocks")
        assert len(model.blocks) == 3

    def test_has_lm_head(self) -> None:
        """Must have an LM head for output projection."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        assert hasattr(model, "lm_head")
        assert isinstance(model.lm_head, torch.nn.Linear)

    def test_weight_tied_embeddings(self) -> None:
        """LM head should share weights with token embedding (weight tying)."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        # Check that LM head weight is tied (same data pointer)
        assert torch.equal(model.lm_head.weight, model.token_embedding.embedding.weight)

    def test_single_token(self) -> None:
        """Should handle single token input."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        x = torch.tensor([[42]])  # Single token
        out = model(x)
        assert out.shape == (1, 1, 256)

    def test_long_sequence(self) -> None:
        """Should handle longer sequences."""
        model = DecoderLM(
            vocab_size=256,
            d_model=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=512,
            attention_backend="standard",
        )
        x = torch.randint(0, 256, (2, 256))  # Longer sequence
        out = model(x)
        assert out.shape == (2, 256, 256)

    def test_different_vocab_sizes(self) -> None:
        """Should work with different vocabulary sizes."""
        for vocab_size in [128, 256, 512, 1000]:
            model = DecoderLM(
                vocab_size=vocab_size,
                d_model=64,
                num_layers=2,
                num_heads=4,
                attention_backend="standard",
            )
            x = torch.randint(0, vocab_size, (1, 8))
            out = model(x)
            assert out.shape == (1, 8, vocab_size)

    def test_different_depths(self) -> None:
        """Should work with different numbers of layers."""
        for num_layers in [1, 2, 4, 8]:
            model = DecoderLM(
                vocab_size=256,
                d_model=64,
                num_layers=num_layers,
                num_heads=4,
                attention_backend="standard",
            )
            x = torch.randint(0, 256, (1, 8))
            out = model(x)
            assert out.shape == (1, 8, 256)
            assert len(model.blocks) == num_layers

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the entire model."""
        model = DecoderLM(
            vocab_size=256, d_model=32, num_layers=2, num_heads=4, attention_backend="standard"
        )
        x = torch.randint(0, 256, (1, 4))
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check that token embedding has gradients (input layer)
        assert model.token_embedding.embedding.weight.grad is not None

        # Check that LM head has gradients (output layer)
        assert model.lm_head.weight.grad is not None

    def test_output_changes_with_different_inputs(self) -> None:
        """Different inputs should produce different outputs."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        x1 = torch.randint(0, 256, (1, 8))
        x2 = torch.randint(0, 256, (1, 8))
        out1 = model(x1)
        out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_token_in_valid_range(self) -> None:
        """Should handle all valid token IDs."""
        vocab_size = 100
        model = DecoderLM(
            vocab_size=vocab_size,
            d_model=32,
            num_layers=2,
            num_heads=4,
            attention_backend="standard",
        )
        x = torch.randint(0, vocab_size, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, vocab_size)

    def test_batch_size_variation(self) -> None:
        """Should work with different batch sizes."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        for batch_size in [1, 2, 4, 8]:
            x = torch.randint(0, 256, (batch_size, 8))
            out = model(x)
            assert out.shape == (batch_size, 8, 256)

    def test_final_layer_norm(self) -> None:
        """Should have final layer norm before LM head."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        assert hasattr(model, "final_norm")
        assert isinstance(model.final_norm, torch.nn.LayerNorm)

    def test_d_model_divisible_by_num_heads(self) -> None:
        """d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            DecoderLM(
                vocab_size=256,
                d_model=63,
                num_layers=2,
                num_heads=4,
                attention_backend="standard",
            )

    def test_logits_can_be_sampled(self) -> None:
        """Output logits should be suitable for sampling (finite values)."""
        model = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        model.eval()
        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out = model(x)

        # Check for NaN or Inf
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic_with_seed(self) -> None:
        """With same seed, should produce same output."""
        torch.manual_seed(42)
        x = torch.randint(0, 256, (1, 4))

        torch.manual_seed(42)
        model1 = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        model1.eval()
        out1 = model1(x)

        torch.manual_seed(42)
        model2 = DecoderLM(
            vocab_size=256, d_model=64, num_layers=2, num_heads=4, attention_backend="standard"
        )
        model2.eval()
        model2.load_state_dict(model1.state_dict())
        out2 = model2(x)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_phase3_pre_layernorm_architecture(self) -> None:
        """Phase 3 minimal transformer should use learned positions + pre-LN blocks."""
        model = DecoderLM(
            vocab_size=256,
            d_model=64,
            num_layers=2,
            num_heads=4,
            attention_backend="standard",
        )
        assert isinstance(model.position_embedding, LearnedPositionEmbedding)
        block = model.blocks[0]
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)

    def test_cross_layer_parameter_sharing_reuses_one_block(self) -> None:
        """When enabled, one block should be reused across all logical layers."""
        model = DecoderLM(
            vocab_size=256,
            d_model=64,
            num_layers=4,
            num_heads=4,
            attention_backend="standard",
            share_layer_weights=True,
        )
        assert len(model.blocks) == 1

        x = torch.randint(0, 256, (2, 8))
        out = model(x)
        assert out.shape == (2, 8, 256)

        # State dict should only contain one physical block when sharing is enabled
        block_keys = [k for k in model.state_dict() if k.startswith("blocks.")]
        assert any(k.startswith("blocks.0.") for k in block_keys)
        assert not any(k.startswith("blocks.1.") for k in block_keys)

    def test_cross_layer_parameter_sharing_reduces_parameters(self) -> None:
        """Shared-layer model should have fewer parameters than non-shared model."""
        non_shared = DecoderLM(
            vocab_size=256,
            d_model=128,
            num_layers=4,
            num_heads=8,
            attention_backend="standard",
            share_layer_weights=False,
        )
        shared = DecoderLM(
            vocab_size=256,
            d_model=128,
            num_layers=4,
            num_heads=8,
            attention_backend="standard",
            share_layer_weights=True,
        )

        non_shared_params = sum(p.numel() for p in non_shared.parameters())
        shared_params = sum(p.numel() for p in shared.parameters())
        assert shared_params < non_shared_params

    def test_factorized_embeddings_create_projection_and_disable_weight_tying(self) -> None:
        """Factorized embeddings should project to d_model and not tie lm_head weights."""
        model = DecoderLM(
            vocab_size=256,
            d_model=128,
            num_layers=2,
            num_heads=8,
            attention_backend="standard",
            embedding_dim=64,
        )
        assert model.use_factorized
        assert model.embedding_projection is not None
        assert model.token_embedding.embedding.embedding_dim == 64
        assert model.embedding_projection.in_features == 64
        assert model.embedding_projection.out_features == 128
        assert model.lm_head.weight.data_ptr() != model.token_embedding.embedding.weight.data_ptr()

        x = torch.randint(0, 256, (2, 8))
        out = model(x)
        assert out.shape == (2, 8, 256)

    def test_non_factorized_embeddings_keep_weight_tying(self) -> None:
        """Without factorization, lm_head should be tied to token embedding."""
        model = DecoderLM(
            vocab_size=256,
            d_model=128,
            num_layers=2,
            num_heads=8,
            attention_backend="standard",
            embedding_dim=None,
        )
        assert not model.use_factorized
        assert model.embedding_projection is None
        assert model.lm_head.weight.data_ptr() == model.token_embedding.embedding.weight.data_ptr()

    def test_from_config_wires_sharing_and_factorized_embedding(self) -> None:
        """from_config should propagate share_layer_weights and embedding_dim."""
        config = ModelConfig(
            model_type="decoder_lm",
            hidden_size=128,
            num_layers=4,
            num_heads=8,
            vocab_size=256,
            max_seq_length=128,
            mla_latent_dim=64,
            rope_base=10000,
            intermediate_size=512,
            num_experts=1,
            experts_per_token=1,
            moe_frequency=0,
            gru_hidden_size=128,
            dropout=0.1,
            embedding_dim=64,
            share_layer_weights=True,
        )
        model = DecoderLM.from_config(config, attention_backend="standard")
        assert model.share_layer_weights
        assert len(model.blocks) == 1
        assert model.use_factorized
        assert model.embedding_dim == 64
