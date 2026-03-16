"""TOML loading helpers for configuration modules."""

from pathlib import Path
from typing import Any, cast

try:
    import tomllib as _toml_loader  # type: ignore[import-not-found]  # Python 3.11+ stdlib
except ImportError:  # pragma: no cover
    import tomli as _toml_loader  # type: ignore[import-not-found]


def load_toml(file_path: str | Path) -> dict[str, Any]:
    """Load TOML file into dictionary."""
    with Path(file_path).open("rb") as f:
        return cast(dict[str, Any], _toml_loader.load(f))


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
