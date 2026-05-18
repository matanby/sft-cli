"""Tests for the rename command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_rename_simple_prefix(mini_model: Path, tmp_path: Path) -> None:
    """--sub 'model\\.' 'transformer.' renames keys."""
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        [
            "rename",
            str(mini_model),
            "--sub",
            r"model\.",
            "--sub",
            "transformer.",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    tensors = load_file(str(out))
    for name in tensors:
        assert not name.startswith("model."), f"key {name} should have been renamed"
    assert any(name.startswith("transformer.") for name in tensors)


def test_rename_regex_groups(tmp_path: Path) -> None:
    """--sub with backreferences (capture groups)."""
    tensors = {
        "layers.0.weight": np.zeros((2, 2), dtype=np.float32),
        "layers.1.weight": np.zeros((2, 2), dtype=np.float32),
        "layers.10.weight": np.zeros((2, 2), dtype=np.float32),
    }
    src = tmp_path / "src.safetensors"
    save_file(tensors, str(src))

    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        [
            "rename",
            str(src),
            "--sub",
            r"layers\.(\d+)",
            "--sub",
            r"blocks.\1",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    renamed = load_file(str(out))
    assert "blocks.0.weight" in renamed
    assert "blocks.1.weight" in renamed
    assert "blocks.10.weight" in renamed
    assert "layers.0.weight" not in renamed


def test_rename_multiple_subs(mini_model: Path, tmp_path: Path) -> None:
    """Two --sub pairs applied in order."""
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        [
            "rename",
            str(mini_model),
            "--sub",
            r"model\.",
            "--sub",
            "transformer.",
            "--sub",
            r"layers",
            "--sub",
            "blocks",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    tensors = load_file(str(out))
    layer_keys = [k for k in tensors if "blocks" in k]
    assert len(layer_keys) > 0, (
        "second substitution should have renamed 'layers' to 'blocks'"
    )
    for name in tensors:
        assert "model." not in name, f"{name} should not contain 'model.'"
        assert "layers" not in name, f"{name} should not contain 'layers'"


def test_rename_smart_output(mini_model: Path) -> None:
    """No -o generates {stem}.renamed.safetensors."""
    result = runner.invoke(
        app,
        [
            "rename",
            str(mini_model),
            "--sub",
            r"model\.",
            "--sub",
            "transformer.",
        ],
    )
    assert result.exit_code == 0, result.output

    expected = mini_model.parent / "mini_model.renamed.safetensors"
    assert expected.exists()
    assert "mini_model.renamed.safetensors" in result.output


def test_rename_dry_run(mini_model: Path) -> None:
    """--dry-run shows mappings, no file written."""
    result = runner.invoke(
        app,
        [
            "rename",
            str(mini_model),
            "--sub",
            r"model\.",
            "--sub",
            "transformer.",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Would rename" in result.output
    assert "→" in result.output

    output = mini_model.parent / "mini_model.renamed.safetensors"
    assert not output.exists()


def test_rename_no_matches(mini_model: Path, tmp_path: Path) -> None:
    """Pattern that matches nothing still copies file, prints warning."""
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        [
            "rename",
            str(mini_model),
            "--sub",
            r"nonexistent_prefix\.",
            "--sub",
            "x.",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert out.exists(), "file should still be written even with no renames"

    original = load_file(str(mini_model))
    copied = load_file(str(out))
    assert set(original.keys()) == set(copied.keys())
