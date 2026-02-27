"""Unit tests for Phase 2 inference: sampling and generate loop."""

import torch

from src.config.model import ModelConfig
from src.inference.run import sample_token
from src.models.learning_model import SimpleLM


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


class TestSampleToken:
    def test_greedy_returns_argmax(self):
        """temperature=0 returns the argmax token."""
        logits = torch.tensor([0.1, 0.9, 0.2, 0.05])
        token = sample_token(logits, temperature=0.0, top_p=1.0, top_k=0)
        assert token == 1

    def test_sampled_token_in_vocab_range(self):
        """Sampled token is a valid vocab index."""
        torch.manual_seed(42)
        logits = torch.zeros(128)
        token = sample_token(logits, temperature=1.0, top_p=1.0, top_k=0)
        assert 0 <= token < 128

    def test_top_k_constrains_to_top_candidates(self):
        """top_k=5 only samples from the 5 highest-logit tokens."""
        torch.manual_seed(0)
        logits = torch.arange(128, dtype=torch.float)
        for _ in range(30):
            token = sample_token(logits, temperature=1.0, top_p=1.0, top_k=5)
            assert token >= 123, f"Expected token in top-5 (>=123), got {token}"


class TestGenerateLoop:
    def test_generate_extends_prompt_by_max_new_tokens(self, tmp_path):
        """Full loop: train → save checkpoint → load → generate → correct length."""
        from torch.utils.data import DataLoader, TensorDataset

        from src.training.loop import train
        from src.training.train import save_checkpoint

        torch.manual_seed(42)
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randint(0, 128, (8, 16))
        y = torch.randint(0, 128, (8, 16))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        _ = train(model, loader, optimizer, max_steps=5, log_interval=0)
        ckpt_path = save_checkpoint(model, optimizer, step=5, output_dir=tmp_path)

        # Load fresh model from checkpoint and generate
        model2 = _make_model()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        prompt = [10, 20, 30]
        tokens = list(prompt)
        max_new = 10
        with torch.no_grad():
            for _ in range(max_new):
                input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                logits = model2(input_ids)[0, -1]
                tokens.append(sample_token(logits, temperature=1.0, top_p=1.0, top_k=0))

        assert len(tokens) == len(prompt) + max_new
        assert all(0 <= t < 128 for t in tokens)

    def test_checkpoint_weights_differ_from_random_init(self, tmp_path):
        """Loaded checkpoint weights are not equal to a fresh random init."""
        from torch.utils.data import DataLoader, TensorDataset

        from src.training.loop import train
        from src.training.train import save_checkpoint

        torch.manual_seed(0)
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randint(0, 128, (8, 16))
        y = torch.randint(0, 128, (8, 16))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        _ = train(model, loader, optimizer, max_steps=10, log_interval=0)
        ckpt_path = save_checkpoint(model, optimizer, step=10, output_dir=tmp_path)

        torch.manual_seed(99)  # different seed → different random init
        model_random = _make_model()
        model_trained = _make_model()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_trained.load_state_dict(ckpt["model_state"])

        any_differ = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(model_random.parameters(), model_trained.parameters(), strict=True)
        )
        assert any_differ, "Trained checkpoint should differ from a random init"
