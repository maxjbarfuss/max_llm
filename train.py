"""Compatibility wrapper for training entrypoint.

Prefer:
    python -m src.training.train --config config/experiment.toml
"""

from src.training.train import main

if __name__ == "__main__":
    main()
