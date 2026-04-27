"""Unit tests for configuration dataclasses.

Tests validation logic and default values for all configuration classes.
"""

from pathlib import Path

import pytest

from src.config import DataConfig, ExperimentConfig, InferenceConfig, ModelConfig, TrainingConfig
from tests.conftest import (
    build_data_config,
    build_inference_config,
    build_model_config,
    build_training_config,
)


# Legacy fixtures with original test defaults for backward compatibility
def make_model_config(**overrides):
    """Legacy test fixture with original defaults."""
    defaults = {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "vocab_size": 50304,
        "max_seq_length": 2048,
        "mla_latent_dim": 768,
        "moe_frequency": 2,
        "num_experts": 16,
        "experts_per_token": 2,
        "dropout": 0.1,
    }
    defaults.update(overrides)
    return build_model_config(**defaults)


def make_training_config(**overrides):
    """Legacy test fixture with original defaults."""
    defaults = {
        "batch_size": 32,
        "max_steps": 100000,
        "warmup_steps": 2000,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "precision_schedule": [(0, 30000, "fp4"), (30000, 80000, "fp8"), (80000, -1, "mixed")],
        "checkpoint_interval": 5000,
        "eval_interval": 1000,
        "log_interval": 100,
        "keep_last_n_checkpoints": 3,
        "use_torch_compile": True,
        "attention_backend": "flash",
        "selective_checkpointing": True,
    }
    defaults.update(overrides)
    return build_training_config(**defaults)


def make_inference_config(**overrides):
    """Legacy test fixture with original defaults."""
    defaults = {
        "device": "auto",
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "use_kv_cache": True,
        "kv_cache_dtype": "fp8",
    }
    defaults.update(overrides)
    return build_inference_config(**defaults)


def make_data_config(**overrides):
    """Legacy test fixture with original defaults."""
    defaults = {
        "dataset_path": "openwebtext",
        "tokenizer_name": "gpt2",
        "max_length": 2048,
        "num_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "streaming": True,
        "cache_dir": "./data/cache",
        "num_shards": 64,
        "validation_split": 0.01,
    }
    defaults.update(overrides)
    return build_data_config(**defaults)


class TestModelConfig:
    """Tests for ModelConfig validation and properties."""

    def test_default_config_valid(self):
        """Baseline model configuration should be valid."""
        config = make_model_config()
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.head_dim == 64

    def test_head_dim_computation(self):
        """Head dimension should be computed correctly."""
        config = make_model_config(hidden_size=768, num_heads=12)
        assert config.head_dim == 64

        config2 = make_model_config(hidden_size=1024, num_heads=16)
        assert config2.head_dim == 64

    def test_latent_head_dim_computation(self):
        """Latent head dimension should be computed correctly."""
        config = make_model_config(mla_latent_dim=512, num_heads=8)
        assert config.latent_head_dim == 64

    def test_hidden_size_divisible_by_heads(self):
        """Hidden size must be divisible by number of heads."""
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            make_model_config(hidden_size=768, num_heads=11)

    def test_hidden_size_multiple_of_64(self):
        """Hidden size must be multiple of 64."""
        with pytest.raises(ValueError, match="must be multiple of 64"):
            make_model_config(hidden_size=770)

    def test_vocab_size_multiple_of_64(self):
        """Vocab size must be multiple of 64."""
        with pytest.raises(ValueError, match="must be multiple of 64"):
            make_model_config(vocab_size=50000)

    def test_mla_latent_dim_divisible_by_heads(self):
        """MLA latent dim must be divisible by number of heads."""
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            make_model_config(mla_latent_dim=500, num_heads=12)

    def test_mla_requires_rope_pos_type(self):
        """MLA must run with decoupled RoPE positional branch."""
        with pytest.raises(ValueError, match="requires decoupled RoPE"):
            make_model_config(attn_type="mla", pos_type="learned", rope_base=None)

    def test_interleaved_attn_default_pattern_valid(self):
        """Interleaved attention defaults to an SWA/MLA pattern."""
        config = make_model_config(attn_type="interleaved")
        assert config.interleaved_attn_pattern == ("swa", "mla")

    def test_interleaved_attn_accepts_toml_list_pattern(self):
        """TOML arrays are normalized to tuples for immutable configs."""
        config = make_model_config(
            attn_type="interleaved",
            interleaved_attn_pattern=["swa", "mha", "mla"],
        )
        assert config.interleaved_attn_pattern == ("swa", "mha", "mla")

    def test_interleaved_attn_rejects_empty_pattern(self):
        with pytest.raises(ValueError, match="interleaved_attn_pattern"):
            make_model_config(attn_type="interleaved", interleaved_attn_pattern=[])

    def test_interleaved_attn_rejects_invalid_pattern_entry(self):
        with pytest.raises(ValueError, match="interleaved_attn_pattern entries"):
            make_model_config(
                attn_type="interleaved",
                interleaved_attn_pattern=["swa", "rla"],
            )

    def test_interleaved_mla_requires_rope_pos_type(self):
        with pytest.raises(ValueError, match="MLA attention requires"):
            make_model_config(
                attn_type="interleaved",
                interleaved_attn_pattern=["swa", "mla"],
                pos_type="learned",
                rope_base=None,
            )

    def test_interleaved_swa_validates_window_size(self):
        with pytest.raises(ValueError, match="swa_window_size"):
            make_model_config(
                attn_type="interleaved",
                interleaved_attn_pattern=["swa", "mla"],
                swa_window_size=0,
            )

    def test_moe_validation(self):
        """MoE configuration should be validated."""
        # Valid MoE config
        config = make_model_config(num_experts=16, experts_per_token=2, moe_frequency=2)
        assert config.num_experts == 16

        # Invalid: experts_per_token > num_experts
        with pytest.raises(ValueError, match="must be between 1 and num_experts"):
            make_model_config(num_experts=4, experts_per_token=5)

    def test_intermediate_size_default(self):
        """Intermediate size should default to 4x hidden size."""
        config = make_model_config(hidden_size=768)
        assert config.intermediate_size == 4 * 768

    def test_ffn_chunk_size_default_and_validation(self):
        """FFN chunking should be disabled by default and positive when set."""
        assert make_model_config().ffn_chunk_size is None
        assert make_model_config(ffn_chunk_size=256).ffn_chunk_size == 256
        with pytest.raises(ValueError, match="ffn_chunk_size"):
            make_model_config(ffn_chunk_size=0)

    def test_gru_hidden_size_default(self):
        """GRU hidden size should default to hidden size."""
        config = make_model_config(hidden_size=768)
        assert config.gru_hidden_size == 768

    def test_frozen_config(self):
        """Config should be immutable after creation."""
        from dataclasses import FrozenInstanceError

        config = make_model_config()
        with pytest.raises(FrozenInstanceError):
            config.hidden_size = 1024

    def test_looped_num_blocks_default_is_none(self):
        """looped_num_blocks defaults to None (one block per logical layer)."""
        config = make_model_config()
        assert config.looped_num_blocks is None

    def test_looped_num_blocks_valid(self):
        """looped_num_blocks accepts any positive value <= num_layers."""
        config = make_model_config(num_layers=12, looped_num_blocks=4)
        assert config.looped_num_blocks == 4

    def test_looped_num_blocks_equals_num_layers_valid(self):
        """looped_num_blocks == num_layers is valid (identity mapping)."""
        config = make_model_config(num_layers=12, looped_num_blocks=12)
        assert config.looped_num_blocks == 12

    def test_looped_num_blocks_one_valid(self):
        """looped_num_blocks=1 is valid (equivalent to share_layer_weights)."""
        config = make_model_config(num_layers=12, looped_num_blocks=1)
        assert config.looped_num_blocks == 1

    def test_looped_num_blocks_zero_invalid(self):
        """looped_num_blocks=0 must raise ValueError."""
        with pytest.raises(ValueError, match="looped_num_blocks must be positive"):
            make_model_config(num_layers=12, looped_num_blocks=0)

    def test_looped_num_blocks_negative_invalid(self):
        """looped_num_blocks < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="looped_num_blocks must be positive"):
            make_model_config(num_layers=12, looped_num_blocks=-1)

    def test_looped_num_blocks_exceeds_num_layers_invalid(self):
        """looped_num_blocks > num_layers must raise ValueError."""
        with pytest.raises(ValueError, match="looped_num_blocks.*must be <=.*num_layers"):
            make_model_config(num_layers=6, looped_num_blocks=7)

    def test_mod_router_defaults(self):
        """MoD router defaults are disabled and conservative."""
        config = make_model_config()
        assert config.mod_router_enabled is False
        assert config.mod_router_capacity_fraction == 0.5
        assert config.mod_router_min_tokens == 1
        assert config.mod_router_start_layer == 0
        assert config.mod_router_frequency == 1
        assert config.mod_router_use_soft_gate is True
        assert config.mod_router_inference_threshold is None

    def test_mod_router_valid(self):
        """MoD router accepts a sparse every-other-layer configuration."""
        config = make_model_config(
            mod_router_enabled=True,
            mod_router_capacity_fraction=0.25,
            mod_router_min_tokens=0,
            mod_router_start_layer=2,
            mod_router_frequency=2,
            mod_router_inference_threshold=0.6,
        )
        assert config.mod_router_enabled is True
        assert config.mod_router_frequency == 2

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
    def test_mod_router_capacity_fraction_invalid(self, fraction: float):
        with pytest.raises(ValueError, match="mod_router_capacity_fraction"):
            make_model_config(mod_router_capacity_fraction=fraction)

    def test_mod_router_min_tokens_invalid(self):
        with pytest.raises(ValueError, match="mod_router_min_tokens"):
            make_model_config(mod_router_min_tokens=-1)

    def test_mod_router_start_layer_invalid_when_enabled(self):
        with pytest.raises(ValueError, match="mod_router_start_layer"):
            make_model_config(num_layers=4, mod_router_enabled=True, mod_router_start_layer=4)

    def test_mod_router_frequency_invalid(self):
        with pytest.raises(ValueError, match="mod_router_frequency"):
            make_model_config(mod_router_frequency=0)

    def test_mod_router_inference_threshold_invalid(self):
        with pytest.raises(ValueError, match="mod_router_inference_threshold"):
            make_model_config(mod_router_inference_threshold=1.1)


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_default_config_valid(self):
        """Baseline training configuration should be valid."""
        config = make_training_config()
        assert config.batch_size == 32
        assert config.max_steps == 100000

    def test_effective_batch_size(self):
        """Effective batch size should account for gradient accumulation."""
        config = make_training_config(batch_size=32, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 128

    def test_learning_rate_bounds(self):
        """Learning rate must be in valid range."""
        with pytest.raises(ValueError, match="learning_rate must be in"):
            make_training_config(learning_rate=1.5)

        with pytest.raises(ValueError):
            make_training_config(learning_rate=0.0)

    def test_warmup_steps_validation(self):
        """Warmup steps must be <= max steps."""
        with pytest.raises(ValueError, match="warmup_steps must be"):
            make_training_config(max_steps=1000, warmup_steps=2000)

    def test_precision_schedule_validation(self):
        """Precision schedule should be validated."""
        valid_schedule = [
            (0, 1000, "fp4"),
            (1000, 5000, "fp8"),
            (5000, -1, "mixed"),
        ]
        config = make_training_config(precision_schedule=valid_schedule)
        assert len(config.precision_schedule) == 3

        # Invalid precision
        invalid_schedule = [(0, 1000, "invalid")]
        with pytest.raises(ValueError, match="Invalid precision"):
            make_training_config(precision_schedule=invalid_schedule)

    def test_wsd_fraction_validation(self):
        """WSD stable + decay fractions must not exceed 1."""
        with pytest.raises(ValueError, match=r"wsd_stable_fraction \+ wsd_decay_fraction"):
            make_training_config(wsd_stable_fraction=0.8, wsd_decay_fraction=0.3)

    def test_wsd_alpha_validation(self):
        """Lowered-linear alpha must be in (0, 1]."""
        with pytest.raises(ValueError, match="wsd_lowered_linear_alpha"):
            make_training_config(wsd_lowered_linear_alpha=0.0)

    def test_benchmark_defaults(self):
        config = make_training_config()
        assert config.benchmark_tasks == []
        assert config.benchmark_eval_interval == 0
        assert config.benchmark_max_examples == 128
        assert config.benchmark_split == "validation"
        assert config.benchmark_length_normalize is True

    def test_benchmark_interval_negative_raises(self):
        with pytest.raises(ValueError, match="benchmark_eval_interval"):
            make_training_config(benchmark_eval_interval=-1)

    def test_benchmark_max_examples_zero_raises(self):
        with pytest.raises(ValueError, match="benchmark_max_examples"):
            make_training_config(benchmark_max_examples=0)

    def test_benchmark_split_empty_raises(self):
        with pytest.raises(ValueError, match="benchmark_split"):
            make_training_config(benchmark_split="")

    def test_benchmark_tasks_list_accepted(self):
        config = make_training_config(
            benchmark_tasks=["hellaswag", "piqa"],
            benchmark_eval_interval=500,
            benchmark_max_examples=64,
        )
        assert config.benchmark_tasks == ["hellaswag", "piqa"]
        assert config.benchmark_eval_interval == 500

    def test_z_loss_weight_default(self):
        config = make_training_config()
        assert config.z_loss_weight == 0.0

    def test_z_loss_weight_negative_raises(self):
        with pytest.raises(ValueError, match="z_loss_weight"):
            make_training_config(z_loss_weight=-1e-4)

    def test_muon_optimizer_defaults(self):
        config = make_training_config()
        assert config.optimizer_type == "adamw"
        assert config.muon_lr is None
        assert config.muon_momentum == 0.95
        assert config.muon_ns_steps == 5

    def test_muon_lr_out_of_range_raises(self):
        with pytest.raises(ValueError, match="muon_lr"):
            make_training_config(muon_lr=0.0)

        with pytest.raises(ValueError, match="muon_lr"):
            make_training_config(muon_lr=1.0)

    def test_muon_momentum_out_of_range_raises(self):
        with pytest.raises(ValueError, match="muon_momentum"):
            make_training_config(muon_momentum=-0.1)

        with pytest.raises(ValueError, match="muon_momentum"):
            make_training_config(muon_momentum=1.0)

    def test_muon_ns_steps_must_be_positive(self):
        with pytest.raises(ValueError, match="muon_ns_steps"):
            make_training_config(muon_ns_steps=0)

    def test_selective_checkpointing_defaults(self):
        config = make_training_config()
        assert config.selective_checkpointing_mode == "full"
        assert config.selective_checkpointing_interval == 1

    def test_selective_checkpointing_mode_valid(self):
        for mode in ("full", "ffn"):
            cfg = make_training_config(selective_checkpointing_mode=mode)
            assert cfg.selective_checkpointing_mode == mode

    def test_selective_checkpointing_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="selective_checkpointing_mode"):
            make_training_config(selective_checkpointing_mode="attention")

    def test_selective_checkpointing_interval_valid(self):
        cfg = make_training_config(selective_checkpointing_interval=2)
        assert cfg.selective_checkpointing_interval == 2

    def test_selective_checkpointing_interval_zero_raises(self):
        with pytest.raises(ValueError, match="selective_checkpointing_interval"):
            make_training_config(selective_checkpointing_interval=0)

    def test_selective_checkpointing_interval_negative_raises(self):
        with pytest.raises(ValueError, match="selective_checkpointing_interval"):
            make_training_config(selective_checkpointing_interval=-1)


class TestInferenceConfig:
    """Tests for InferenceConfig validation."""

    def test_default_config_valid(self):
        """Baseline inference configuration should be valid."""
        config = make_inference_config()
        assert config.device == "auto"
        assert config.kv_cache_dtype == "fp8"

    def test_sampling_validation(self):
        """Sampling parameters should be validated."""
        with pytest.raises(ValueError, match="top_p must be in"):
            make_inference_config(top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be in"):
            make_inference_config(top_p=1.5)

        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            make_inference_config(max_new_tokens=0)


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_default_config_valid(self):
        """Baseline data configuration should be valid."""
        config = make_data_config()
        assert config.num_workers == 6
        assert config.streaming is True
        assert config.tokenizer_backend == "gpt2_bpe"
        assert config.tokenizer_name == "gpt2"

    def test_unigram_backend_config(self):
        """Unigram tokenizer backend should be accepted."""
        config = make_data_config(
            tokenizer_backend="unigram",
            tokenizer_name="unigram-local",
            unigram_model_path="./tokenizers/unigram.model",
        )
        assert config.tokenizer_backend == "unigram"
        assert config.unigram_model_path == "./tokenizers/unigram.model"

    def test_packing_metadata_paths_default_and_empty_string(self):
        config = make_data_config()
        assert config.packing_metadata_path is None
        assert config.validation_packing_metadata_path is None
        assert config.test_packing_metadata_path is None

        config = make_data_config(
            packing_metadata_path="",
            validation_packing_metadata_path="   ",
            test_packing_metadata_path="",
        )
        assert config.packing_metadata_path is None
        assert config.validation_packing_metadata_path is None
        assert config.test_packing_metadata_path is None

    def test_unigram_model_path_validation(self):
        """Unigram model path cannot be empty when provided."""
        with pytest.raises(ValueError, match="unigram_model_path cannot be empty"):
            make_data_config(
                tokenizer_backend="unigram",
                tokenizer_name="unigram-local",
                unigram_model_path="",
            )

    def test_tokenizer_name_required(self):
        """Tokenizer name cannot be empty."""
        with pytest.raises(ValueError, match="tokenizer_name cannot be empty"):
            make_data_config(tokenizer_name="")

    def test_validation_split_fraction(self):
        """Validation split fraction should be in (0, 1)."""
        config = make_data_config(validation_split=0.1)
        assert config.validation_split == 0.1

        with pytest.raises(ValueError, match="validation_split must be"):
            make_data_config(validation_split=1.5)

    def test_validation_split_absolute(self):
        """Validation split must be a float ratio, not an absolute count."""
        with pytest.raises(ValueError, match="validation_split must be a float ratio"):
            make_data_config(validation_split=1000)


class TestExperimentConfig:
    """Tests for ExperimentConfig composition."""

    def test_default_config_valid(self):
        """Baseline experiment configuration should be valid."""
        config = ExperimentConfig(
            name="test_experiment",
            output_dir="./outputs",
            model=make_model_config(),
            training=make_training_config(),
            inference=make_inference_config(),
            data=make_data_config(),
        )
        assert config.name == "test_experiment"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.inference, InferenceConfig)
        assert isinstance(config.data, DataConfig)

    def test_data_max_length_validation(self):
        """Data max length must be <= model max sequence length."""
        model_config = make_model_config(max_seq_length=1024)
        data_config = make_data_config(max_length=512)

        # Valid: data max_length <= model max_seq_length
        config = ExperimentConfig(
            name="test",
            output_dir="./outputs",
            model=model_config,
            training=make_training_config(),
            inference=make_inference_config(),
            data=data_config,
        )
        assert config.data.max_length <= config.model.max_seq_length

        # Invalid: data max_length > model max_seq_length
        with pytest.raises(ValueError, match="must be <="):
            ExperimentConfig(
                name="test",
                output_dir="./outputs",
                model=make_model_config(max_seq_length=512),
                training=make_training_config(),
                inference=make_inference_config(),
                data=make_data_config(max_length=1024),
            )

    def test_custom_configs(self):
        """Custom sub-configurations should be used."""
        custom_model = make_model_config(hidden_size=1024, num_layers=24, num_heads=16)
        custom_training = make_training_config(batch_size=64)
        custom_inference = make_inference_config(device="cpu")
        custom_data = make_data_config(num_workers=8)

        config = ExperimentConfig(
            name="custom",
            output_dir="./outputs",
            model=custom_model,
            training=custom_training,
            inference=custom_inference,
            data=custom_data,
        )

        assert config.model.hidden_size == 1024
        assert config.training.batch_size == 64
        assert config.inference.device == "cpu"
        assert config.data.num_workers == 8

    def test_from_toml(self, tmp_path: Path):
        """TOML config should map into logical sections."""
        config_file = tmp_path / "experiment.toml"
        config_file.write_text(
            """
[experiment]
name = "toml-test"
output_dir = "./tmp-outputs"

[model]
hidden_size = 768
num_layers = 12
num_heads = 12
vocab_size = 50304
max_seq_length = 2048
mla_latent_dim = 768
rope_base = 10000
intermediate_size = 3072
ffn_chunk_size = 256
num_experts = 16
experts_per_token = 2
moe_frequency = 2
gru_hidden_size = 768
dropout = 0.1

[training]
batch_size = 8
gradient_accumulation_steps = 1
max_steps = 500
warmup_steps = 20
learning_rate = 0.0003
weight_decay = 0.1
betas = [0.9, 0.95]
epsilon = 1e-8
gradient_clip_norm = 1.0
precision_schedule = [[0, 100, "fp4"], [100, 300, "fp8"], [300, -1, "mixed"]]
moe_balance_loss_weight = 0.01
distributed_backend = "ddp"
checkpoint_interval = 100
eval_interval = 50
log_interval = 10
keep_last_n_checkpoints = 3
use_torch_compile = true
attention_backend = "flash"
selective_checkpointing = true

[inference]
device = "cpu"
max_new_tokens = 32
temperature = 0.8
top_p = 0.9
top_k = 50
use_kv_cache = true
kv_cache_dtype = "fp8"

[data]
dataset_path = "openwebtext"
tokenizer_backend = "unigram"
tokenizer_name = "unigram-local"
tokenizer_mode = "codepoint"
tokenizer_vocab_size = 128
unigram_model_path = "./tokenizers/unigram.model"
max_length = 2048
num_workers = 6
prefetch_factor = 2
pin_memory = true
persistent_workers = true
streaming = true
cache_dir = "./data/cache"
num_shards = 64
validation_split = 0.01
seed = 42
""".strip(),
            encoding="utf-8",
        )

        config = ExperimentConfig.from_toml(config_file)
        assert config.name == "toml-test"
        assert config.output_dir == "./tmp-outputs"
        assert config.training.batch_size == 8
        assert config.model.ffn_chunk_size == 256
        assert config.inference.device == "cpu"
        assert config.data.tokenizer_backend == "unigram"

    def test_from_split_toml_files(self, tmp_path: Path):
        """Split TOML files should map to config sections."""
        experiment_file = tmp_path / "experiment.toml"
        model_file = tmp_path / "model.toml"
        training_file = tmp_path / "training.toml"
        inference_file = tmp_path / "inference.toml"
        data_file = tmp_path / "data.toml"

        experiment_file.write_text(
            '[experiment]\nname = "split-test"\noutput_dir = "./split-outputs"\n',
            encoding="utf-8",
        )
        model_file.write_text(
            "\n".join(
                [
                    "hidden_size = 768",
                    "num_layers = 12",
                    "num_heads = 12",
                    "vocab_size = 50304",
                    "max_seq_length = 2048",
                    "mla_latent_dim = 768",
                    "rope_base = 10000",
                    "intermediate_size = 3072",
                    "num_experts = 16",
                    "experts_per_token = 2",
                    "moe_frequency = 2",
                    "gru_hidden_size = 768",
                    "dropout = 0.1",
                ]
            ),
            encoding="utf-8",
        )
        training_file.write_text(
            "\n".join(
                [
                    "batch_size = 16",
                    "gradient_accumulation_steps = 1",
                    "max_steps = 20000",
                    "warmup_steps = 500",
                    "learning_rate = 0.0003",
                    "weight_decay = 0.1",
                    "betas = [0.9, 0.95]",
                    "epsilon = 1e-8",
                    "gradient_clip_norm = 1.0",
                    'precision_schedule = [[0, 1000, "fp4"], [1000, 5000, "fp8"], [5000, -1, "mixed"]]',
                    "moe_balance_loss_weight = 0.01",
                    'distributed_backend = "ddp"',
                    "checkpoint_interval = 500",
                    "eval_interval = 100",
                    "log_interval = 10",
                    "keep_last_n_checkpoints = 3",
                    "use_torch_compile = true",
                    'attention_backend = "flash"',
                    "selective_checkpointing = true",
                ]
            ),
            encoding="utf-8",
        )
        inference_file.write_text(
            "\n".join(
                [
                    'device = "auto"',
                    "max_new_tokens = 128",
                    "temperature = 0.8",
                    "top_p = 0.9",
                    "top_k = 50",
                    "use_kv_cache = true",
                    'kv_cache_dtype = "fp8"',
                ]
            ),
            encoding="utf-8",
        )
        data_file.write_text(
            "\n".join(
                [
                    'dataset_path = "openwebtext"',
                    'tokenizer_name = "gpt2"',
                    'tokenizer_mode = "codepoint"',
                    "tokenizer_vocab_size = 128",
                    'tokenizer_backend = "gpt2_bpe"',
                    'unigram_model_path = ""',
                    "max_length = 2048",
                    "num_workers = 6",
                    "prefetch_factor = 2",
                    "pin_memory = true",
                    "persistent_workers = true",
                    "streaming = true",
                    'cache_dir = "./data/cache"',
                    "num_shards = 64",
                    "validation_split = 0.01",
                    "seed = 42",
                ]
            ),
            encoding="utf-8",
        )

        config = ExperimentConfig.from_toml_files(
            experiment_path=experiment_file,
            model_path=model_file,
            training_path=training_file,
            inference_path=inference_file,
            data_path=data_file,
        )
        assert config.name == "split-test"
        assert config.output_dir == "./split-outputs"
        assert config.model.hidden_size == 768
        assert config.training.batch_size == 16
        assert config.inference.max_new_tokens == 128
        assert config.data.tokenizer_name == "gpt2"


class TestMilestoneToml:
    """Smoke tests for on-disk milestone TOML configs."""
