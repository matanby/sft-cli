"""Tests for the cast command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_cast_to_fp32(mini_model: Path) -> None:
    result = runner.invoke(app, ["cast", str(mini_model), "--dtype", "fp32"])
    assert result.exit_code == 0, result.output

    output = mini_model.parent / "mini_model.fp32.safetensors"
    assert output.exists()

    tensors = load_file(str(output))
    assert len(tensors) == 14
    for tensor in tensors.values():
        assert tensor.dtype == np.float32


def test_cast_with_include(mini_model: Path) -> None:
    """Only tensors matching include pattern are cast; others keep original dtype."""
    result = runner.invoke(
        app,
        ["cast", str(mini_model), "--dtype", "fp32", "--include", "model.layers.**"],
    )
    assert result.exit_code == 0, result.output

    output = mini_model.parent / "mini_model.fp32.safetensors"
    tensors = load_file(str(output))

    for name, tensor in tensors.items():
        if name.startswith("model.layers."):
            assert tensor.dtype == np.float32, f"{name} should be fp32"
        else:
            assert tensor.dtype == np.float16, f"{name} should stay fp16"


def test_cast_with_exclude(mini_model: Path) -> None:
    """Excluded tensors keep their original dtype."""
    result = runner.invoke(
        app,
        [
            "cast",
            str(mini_model),
            "--dtype",
            "fp32",
            "--exclude",
            "model.embed_tokens.*",
        ],
    )
    assert result.exit_code == 0, result.output

    output = mini_model.parent / "mini_model.fp32.safetensors"
    tensors = load_file(str(output))

    assert tensors["model.embed_tokens.weight"].dtype == np.float16
    for name, tensor in tensors.items():
        if name != "model.embed_tokens.weight":
            assert tensor.dtype == np.float32, f"{name} should be fp32"


def test_cast_smart_output(mini_model: Path) -> None:
    """Without -o, output uses {stem}.{dtype}.safetensors naming."""
    result = runner.invoke(app, ["cast", str(mini_model), "--dtype", "fp32"])
    assert result.exit_code == 0, result.output

    expected = mini_model.parent / "mini_model.fp32.safetensors"
    assert expected.exists()
    assert "mini_model.fp32.safetensors" in result.output


def test_cast_explicit_output(mini_model: Path) -> None:
    """Explicit -o flag writes to the specified path."""
    custom = mini_model.parent / "custom.safetensors"
    result = runner.invoke(
        app, ["cast", str(mini_model), "--dtype", "fp32", "-o", str(custom)]
    )
    assert result.exit_code == 0, result.output
    assert custom.exists()

    tensors = load_file(str(custom))
    assert len(tensors) == 14


def test_cast_dry_run(mini_model: Path) -> None:
    """--dry-run prints what would happen without writing a file."""
    result = runner.invoke(
        app, ["cast", str(mini_model), "--dtype", "fp32", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Would cast" in result.output

    output = mini_model.parent / "mini_model.fp32.safetensors"
    assert not output.exists()


def test_cast_invalid_dtype(mini_model: Path) -> None:
    """Invalid dtype gives an error exit."""
    result = runner.invoke(app, ["cast", str(mini_model), "--dtype", "bogus"])
    assert result.exit_code != 0
