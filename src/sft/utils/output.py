"""Smart output path generation."""

from __future__ import annotations

from pathlib import Path


def default_output(input_path: Path, suffix: str) -> Path:
    """Generate a default output path: {stem}.{suffix}.safetensors"""
    return input_path.parent / f"{input_path.stem}.{suffix}.safetensors"


def resolve_output(
    explicit: Path | None,
    input_path: Path,
    suffix: str,
) -> Path:
    """Resolve the output path: use explicit -o if given, otherwise smart default."""
    if explicit is not None:
        return explicit
    return default_output(input_path, suffix)
