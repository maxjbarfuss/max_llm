"""Unit tests for configuration dataclasses.

Tests validation logic and default values for all configuration classes.
"""

import pytest
import torch

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig


class TestModelConfig:
    """Tests for ModelConfig validation and properties."""
    
    def test_default_config_valid(self):
        """Default configuration should be valid."""
        config = ModelConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.head_dim == 64
    
    def test_head_dim_computation(self):
        """Head dimension should be computed correctly."""
        config = ModelConfig(hidden_size=768, num_heads=12)
        assert config.head_dim == 64
        
        config2 = ModelConfig(hidden_size=1024, num_heads=16)
        assert config2.head_dim == 64
    
    def test_latent_head_dim_computation(self):
        """Latent head dimension should be computed correctly."""
        config = ModelConfig(mla_latent_dim=512, num_heads=8)
        assert config.latent_head_dim == 64
    
    def test_hidden_size_divisible_by_heads(self):
        """Hidden size must be divisible by number of heads."""
        with pytest.raises(AssertionError, match="must be divisible by num_heads"):
            ModelConfig(hidden_size=768, num_heads=11)
    
    def test_hidden_size_multiple_of_64(self):
        """Hidden size must be multiple of 64."""
        with pytest.raises(AssertionError, match="must be multiple of 64"):
            ModelConfig(hidden_size=770)
    
    def test_vocab_size_multiple_of_64(self):
        """Vocab size must be multiple of 64."""
        with pytest.raises(AssertionError, match="must be multiple of 64"):
            ModelConfig(vocab_size=50000)
    
    def test_mla_latent_dim_divisible_by_heads(self):
        """MLA latent dim must be divisible by number of heads."""
        with pytest.raises(AssertionError, match="must be divisible by num_heads"):
            ModelConfig(mla_latent_dim=500, num_heads=12)
    
    def test_moe_validation(self):
        """MoE configuration should be validated."""
        # Valid MoE config
        config = ModelConfig(num_experts=16, experts_per_token=2, moe_frequency=2)
        assert config.num_experts == 16
        
        # Invalid: experts_per_token > num_experts
        with pytest.raises(AssertionError, match="must be between 1 and num_experts"):
            ModelConfig(num_experts=4, experts_per_token=5)
    
    def test_intermediate_size_default(self):
        """Intermediate size should default to 4x hidden size."""
        config = ModelConfig(hidden_size=768)
        assert config.intermediate_size == 4 * 768
    
    def test_gru_hidden_size_default(self):
        """GRU hidden size should default to hidden size."""
        config = ModelConfig(hidden_size=768)
        assert config.gru_hidden_size == 768
    
    def test_frozen_config(self):
        """Config should be immutable after creation."""
        config = ModelConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.hidden_size = 1024


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""
    
    def test_default_config_valid(self):
        """Default training configuration should be valid."""
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.max_steps == 100000
    
    def test_effective_batch_size(self):
        """Effective batch size should account for gradient accumulation."""
        config = TrainingConfig(batch_size=32, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 128
    
    def test_learning_rate_bounds(self):
        """Learning rate must be in valid range."""
        with pytest.raises(AssertionError, match="learning_rate must be in"):
            TrainingConfig(learning_rate=1.5)
        
        with pytest.raises(AssertionError):
            TrainingConfig(learning_rate=0.0)
    
    def test_warmup_steps_validation(self):
        """Warmup steps must be <= max steps."""
        with pytest.raises(AssertionError, match="warmup_steps must be"):
            TrainingConfig(max_steps=1000, warmup_steps=2000)
    
    def test_precision_schedule_validation(self):
        """Precision schedule should be validated."""
        valid_schedule = [
            (0, 1000, "fp4"),
            (1000, 5000, "fp8"),
            (5000, -1, "mixed"),
        ]
        config = TrainingConfig(precision_schedule=valid_schedule)
        assert len(config.precision_schedule) == 3
        
        # Invalid precision
        invalid_schedule = [(0, 1000, "invalid")]
        with pytest.raises(AssertionError, match="Invalid precision"):
            TrainingConfig(precision_schedule=invalid_schedule)


class TestDataConfig:
    """Tests for DataConfig validation."""
    
    def test_default_config_valid(self):
        """Default data configuration should be valid."""
        config = DataConfig()
        assert config.num_workers == 6
        assert config.streaming is True
    
    def test_validation_split_fraction(self):
        """Validation split fraction should be in (0, 1)."""
        config = DataConfig(validation_split=0.1)
        assert config.validation_split == 0.1
        
        with pytest.raises(AssertionError, match="validation_split fraction must be"):
            DataConfig(validation_split=1.5)
    
    def test_validation_split_absolute(self):
        """Validation split can be absolute number."""
        config = DataConfig(validation_split=1000)
        assert config.validation_split == 1000
        
        with pytest.raises(AssertionError, match="validation_split must be positive"):
            DataConfig(validation_split=-100)


class TestExperimentConfig:
    """Tests for ExperimentConfig composition."""
    
    def test_default_config_valid(self):
        """Default experiment configuration should be valid."""
        config = ExperimentConfig(name="test_experiment")
        assert config.name == "test_experiment"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_data_max_length_validation(self):
        """Data max length must be <= model max sequence length."""
        model_config = ModelConfig(max_seq_length=1024)
        data_config = DataConfig(max_length=512)
        
        # Valid: data max_length <= model max_seq_length
        config = ExperimentConfig(
            name="test",
            model=model_config,
            data=data_config
        )
        assert config.data.max_length <= config.model.max_seq_length
        
        # Invalid: data max_length > model max_seq_length
        with pytest.raises(AssertionError, match="must be <="):
            ExperimentConfig(
                name="test",
                model=ModelConfig(max_seq_length=512),
                data=DataConfig(max_length=1024)
            )
    
    def test_custom_configs(self):
        """Custom sub-configurations should be used."""
        custom_model = ModelConfig(hidden_size=1024, num_layers=24)
        custom_training = TrainingConfig(batch_size=64)
        custom_data = DataConfig(num_workers=8)
        
        config = ExperimentConfig(
            name="custom",
            model=custom_model,
            training=custom_training,
            data=custom_data
        )
        
        assert config.model.hidden_size == 1024
        assert config.training.batch_size == 64
        assert config.data.num_workers == 8
