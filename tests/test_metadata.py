"""Tests for the metadata command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_metadata_view(mini_model: Path) -> None:
    result = runner.invoke(app, ["metadata", str(mini_model)])
    assert result.exit_code == 0, result.output
    assert "format: pt" in result.output
    assert "model_type: llama" in result.output


def test_metadata_view_json(mini_model: Path) -> None:
    result = runner.invoke(app, ["metadata", str(mini_model), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, dict)
    assert data["format"] == "pt"
    assert data["model_type"] == "llama"


def test_metadata_set(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        ["metadata", str(mini_model), "--set", "new_key=new_val", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    from sft.ops.metadata import get_metadata

    md = get_metadata(out)
    assert md["new_key"] == "new_val"
    assert md["format"] == "pt"
    assert md["model_type"] == "llama"


def test_metadata_unset(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        ["metadata", str(mini_model), "--unset", "model_type", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    from sft.ops.metadata import get_metadata

    md = get_metadata(out)
    assert "model_type" not in md
    assert md["format"] == "pt"


def test_metadata_set_and_unset(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        [
            "metadata",
            str(mini_model),
            "--set",
            "new_key=hello",
            "--unset",
            "model_type",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    from sft.ops.metadata import get_metadata

    md = get_metadata(out)
    assert md["new_key"] == "hello"
    assert "model_type" not in md
    assert md["format"] == "pt"


def test_metadata_smart_output(mini_model: Path) -> None:
    result = runner.invoke(
        app,
        ["metadata", str(mini_model), "--set", "foo=bar"],
    )
    assert result.exit_code == 0, result.output

    expected = mini_model.parent / "mini_model.metadata.safetensors"
    assert expected.exists()
    assert "mini_model.metadata.safetensors" in result.output


def test_metadata_no_metadata(tmp_path: Path) -> None:
    tensors = {"weight": np.ones((4, 4), dtype=np.float32)}
    path = tmp_path / "bare.safetensors"
    save_file(tensors, str(path))

    result = runner.invoke(app, ["metadata", str(path)])
    assert result.exit_code == 0, result.output
    assert "No metadata" in result.output


def test_metadata_preserves_tensors(mini_model: Path, tmp_path: Path) -> None:
    original = load_file(str(mini_model))

    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app,
        ["metadata", str(mini_model), "--set", "extra=val", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output

    written = load_file(str(out))
    assert set(written.keys()) == set(original.keys())
    for name in original:
        np.testing.assert_array_equal(original[name], written[name])
