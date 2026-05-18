"""Tests for the ls command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def _make_file(
    path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_ls_multiple_files(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "alpha.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "beta.safetensors",
        {"w": np.zeros((8, 8), dtype=np.float16)},
    )
    result = runner.invoke(app, ["ls", str(f1), str(f2)])
    assert result.exit_code == 0
    assert "alpha.safetensors" in result.output
    assert "beta.safetensors" in result.output


def test_ls_json(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "one.safetensors",
        {
            "a": np.zeros((2, 2), dtype=np.float32),
            "b": np.zeros((3,), dtype=np.float16),
        },
    )
    result = runner.invoke(app, ["ls", str(f1), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    row = data[0]
    assert row["file"] == "one.safetensors"
    assert row["tensors"] == 2
    assert "params" in row
    assert "bytes" in row
    assert "dtypes" in row


def test_ls_sort_by_size(tmp_path: Path) -> None:
    small = _make_file(
        tmp_path / "small.safetensors",
        {"w": np.zeros((2, 2), dtype=np.float16)},
    )
    big = _make_file(
        tmp_path / "big.safetensors",
        {"w": np.zeros((64, 64), dtype=np.float32)},
    )
    result = runner.invoke(app, ["ls", str(big), str(small), "--sort", "size"])
    assert result.exit_code == 0
    lines = [line for line in result.output.strip().splitlines() if line.strip()]
    # First data line (after header) should be the smaller file
    assert "small.safetensors" in lines[1]
    assert "big.safetensors" in lines[2]


def test_ls_single_file(mini_model: Path) -> None:
    result = runner.invoke(app, ["ls", str(mini_model)])
    assert result.exit_code == 0
    assert "mini_model.safetensors" in result.output
