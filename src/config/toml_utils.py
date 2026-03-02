"""TOML loading helpers for configuration modules."""

import importlib
import sys
import warnings
from dataclasses import MISSING, fields
from pathlib import Path
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


def apply_field_defaults(
    config_class: type,
    raw_data: dict[str, Any],
    warn_on_defaults: bool = True,
) -> dict[str, Any]:
    """Apply default values for missing optional fields in config.

    Enables backward compatibility when loading old TOML files that don't
    have newly-added optional fields.

    Args:
        config_class: Dataclass type to inspect for fields.
        raw_data: Raw TOML data dictionary.
        warn_on_defaults: If True, warns when defaults are applied.

    Returns:
        Updated dictionary with defaults filled in for missing fields.
    """
    result = raw_data.copy()
    applied_defaults = []

    for field in fields(config_class):
        # Skip if field is already present
        if field.name in result:
            continue

        # Only apply defaults for fields with default values or default_factory
        if field.default is MISSING and field.default_factory is MISSING:
            # Has neither default nor default_factory - field is required
            continue

        if field.default is not MISSING:
            # Has a default value
            result[field.name] = field.default
            applied_defaults.append(field.name)
        elif field.default_factory is not MISSING:
            # Has a default_factory
            result[field.name] = field.default_factory()
            applied_defaults.append(field.name)

    if applied_defaults and warn_on_defaults:
        warnings.warn(
            f"Applied defaults for missing fields in {config_class.__name__}: "
            f"{', '.join(applied_defaults)}. "
            f"Consider updating your TOML config file.",
            UserWarning,
            stacklevel=2,
        )

    return result
