"""Dataset preparation framework.

First-class data preparation pipeline with safe defaults, registry-based
strategies, and a CLI entrypoint.
"""

from src.data.preparation.config import DataPreparationConfig, load_config
from src.data.preparation.pipeline import PreparationPipeline

__all__ = ["DataPreparationConfig", "PreparationPipeline", "load_config"]
