"""CLI entrypoint for dataset preparation."""

from __future__ import annotations

import argparse
import logging

from src.data.preparation.pipeline import PreparationPipeline


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset")
    parser.add_argument("--config", required=True, help="Path to JSON or TOML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    PreparationPipeline().run_from_config_path(args.config)


if __name__ == "__main__":
    main()
