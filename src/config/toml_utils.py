"""TOML loading helpers for configuration modules."""

import importlib
from pathlib import Path
import sys
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:  # pragma: no cover
    _toml_loader = None


def load_toml(file_path: str | Path) -> dict[str, Any]:
    """Load TOML file into dictionary."""
    path = Path(file_path)
    with path.open("rb") as toml_file:
        if _toml_loader is not None:
            return _toml_loader.load(toml_file)
        tomli = importlib.import_module("tomli")
        return cast(dict[str, Any], tomli.load(toml_file))


def section_or_root(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return [name] table if present, otherwise return root keys."""
    section = raw.get(name)
    if section is None:
        return raw
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return cast(dict[str, Any], section)


def require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return required TOML table by name."""
    if name not in raw:
        raise ValueError(f"Missing required TOML section: [{name}]")
    section = raw[name]
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return cast(dict[str, Any], section)
