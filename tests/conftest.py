"""Shared pytest fixtures and utilities for all tests.

Provides schema-aware config builders that automatically adapt to new fields.
Tests only need to override fields they care about, reducing fixture brittleness.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from src.config import DataConfig, InferenceConfig, ModelConfig, TrainingConfig


def build_model_config(**overrides: Any) -> ModelConfig:
    """Build a ModelConfig with sensible test defaults.

    Tests can override any field. This factory auto-adapts when new fields
    are added to ModelConfig, eliminating fixture brittleness.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        ModelConfig with test defaults + overrides.

    Example:
        >>> cfg = build_model_config(hidden_size=128, num_layers=2)
    """
    # Minimal valid defaults that satisfy all validation constraints
    defaults = {
        "hidden_size": 64,  # Multiple of 64, smallest valid value
        "num_layers": 1,
        "num_heads": 4,  # 64 % 4 = 0 ✓
        "vocab_size": 128,  # Multiple of 64
        "max_seq_length": 64,
        "mla_latent_dim": 64,  # Multiple of num_heads
        "rope_base": 10000,
        "intermediate_size": None,  # Auto-computed: 4 * hidden_size
        "num_experts": 1,
        "experts_per_token": 1,
        "moe_frequency": 0,  # MoE disabled by default
        "gru_hidden_size": None,  # Auto-computed: hidden_size
        "dropout": 0.0,
    }

    # Merge overrides
    defaults.update(overrides)
    return ModelConfig(**defaults)


def build_training_config(**overrides: Any) -> TrainingConfig:
    """Build a TrainingConfig with sensible test defaults.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        TrainingConfig with test defaults + overrides.
    """
    defaults = {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_steps": 100,
        "warmup_steps": 10,
        "learning_rate": 1e-3,
        "optimizer_type": "adamw",
        "muon_lr": None,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "epsilon": 1e-8,
        "gradient_clip_norm": 1.0,
        "precision_schedule": [(0, -1, "mixed")],
        "moe_balance_loss_weight": 0.01,
        "distributed_backend": "ddp",
        "checkpoint_interval": 50,
        "eval_interval": 25,
        "log_interval": 10,
        "keep_last_n_checkpoints": 2,
        "use_torch_compile": False,  # Disabled for faster tests
        "attention_backend": "torch",  # Most compatible
        "selective_checkpointing": False,
        "generalization_filter_enabled": False,
        "generalization_filter_interval": 1,
        "generalization_filter_val_batches": 1,
        "generalization_filter_damping": 0.25,
        "generalization_filter_preserve_norm": True,
    }

    defaults.update(overrides)
    return TrainingConfig(**defaults)


def build_data_config(**overrides: Any) -> DataConfig:
    """Build a DataConfig with sensible test defaults.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        DataConfig with test defaults + overrides.
    """
    defaults = {
        "dataset_path": "test_dataset",
        "tokenizer_name": "codepoint",
        "tokenizer_mode": "codepoint",
        "tokenizer_vocab_size": 128,
        "tokenizer_backend": "gpt2_bpe",
        "unigram_model_path": None,
        "max_length": 64,
        "num_workers": 0,
        "prefetch_factor": 2,
        "pin_memory": False,
        "persistent_workers": False,
        "streaming": False,
        "cache_dir": "/tmp/test_cache",
        "num_shards": 1,
        "validation_split": 0.1,
        "validation_dataset_path": None,
        "test_dataset_path": None,
        "seed": 42,
    }

    defaults.update(overrides)
    return DataConfig(**defaults)


def build_inference_config(**overrides: Any) -> InferenceConfig:
    """Build an InferenceConfig with sensible test defaults.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        InferenceConfig with test defaults + overrides.
    """
    defaults = {
        "device": "cpu",
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "use_kv_cache": True,
        "kv_cache_dtype": "bf16",
    }

    defaults.update(overrides)
    return InferenceConfig(**defaults)


def validate_config_schema_coverage() -> None:
    """Assert that all config fields have defaults in builders.

    This test helper catches missing fields when new parameters are added.
    Run during test discovery to ensure fixture coverage is complete.
    """
    # Check ModelConfig
    model_fields = {f.name for f in fields(ModelConfig) if not f.name.startswith("_")}
    model_builder_keys = set(build_model_config().__dict__.keys())
    missing = model_fields - model_builder_keys
    assert not missing, f"build_model_config missing fields: {missing}"

    # Check TrainingConfig
    training_fields = {f.name for f in fields(TrainingConfig) if not f.name.startswith("_")}
    training_builder_keys = set(build_training_config().__dict__.keys())
    missing = training_fields - training_builder_keys
    assert not missing, f"build_training_config missing fields: {missing}"

    # Check DataConfig
    data_fields = {f.name for f in fields(DataConfig) if not f.name.startswith("_")}
    data_builder_keys = set(build_data_config().__dict__.keys())
    missing = data_fields - data_builder_keys
    assert not missing, f"build_data_config missing fields: {missing}"

    # Check InferenceConfig
    inference_fields = {f.name for f in fields(InferenceConfig) if not f.name.startswith("_")}
    inference_builder_keys = set(build_inference_config().__dict__.keys())
    missing = inference_fields - inference_builder_keys
    assert not missing, f"build_inference_config missing fields: {missing}"
