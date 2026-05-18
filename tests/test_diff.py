"""Tests for the diff command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.diff import diff_files

runner = CliRunner()


def _make_file(
    path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_diff_identical(mini_model: Path) -> None:
    result = diff_files(mini_model, mini_model)
    assert result.added == []
    assert result.removed == []
    assert result.shape_changed == {}
    assert result.dtype_changed == {}
    assert len(result.unchanged) > 0
    assert result.value_diffs is None


def test_diff_added_tensors(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w1": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {
            "w1": np.zeros((4, 4), dtype=np.float32),
            "w2": np.ones((8,), dtype=np.float16),
        },
    )
    result = diff_files(base, target)
    assert result.added == ["w2"]
    assert result.removed == []
    assert "w1" in result.unchanged


def test_diff_removed_tensors(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {
            "w1": np.zeros((4, 4), dtype=np.float32),
            "w2": np.ones((8,), dtype=np.float16),
        },
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w1": np.zeros((4, 4), dtype=np.float32)},
    )
    result = diff_files(base, target)
    assert result.removed == ["w2"]
    assert result.added == []
    assert "w1" in result.unchanged


def test_diff_shape_changed(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": np.zeros((8, 8), dtype=np.float32)},
    )
    result = diff_files(base, target)
    assert "w" in result.shape_changed
    assert result.shape_changed["w"] == ((4, 4), (8, 8))
    assert result.added == []
    assert result.removed == []


def test_diff_dtype_changed(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float16)},
    )
    result = diff_files(base, target)
    assert "w" in result.dtype_changed
    assert result.dtype_changed["w"][0] == "F32"
    assert result.dtype_changed["w"][1] == "F16"


def test_diff_delta(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    w_base = rng.randn(8, 8).astype(np.float32)
    w_target = w_base + 0.01 * rng.randn(8, 8).astype(np.float32)

    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": w_base, "b": np.ones(4, dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": w_target, "b": np.ones(4, dtype=np.float32)},
    )
    result = diff_files(base, target, compute_delta=True)
    assert result.value_diffs is not None
    assert "w" in result.value_diffs
    assert result.value_diffs["w"].l2_norm > 0
    assert result.value_diffs["w"].cosine_sim > 0.99
    # Identical tensor should have l2_norm == 0 and cosine_sim == 1.0
    assert result.value_diffs["b"].l2_norm == pytest.approx(0.0)
    assert result.value_diffs["b"].cosine_sim == pytest.approx(1.0)


def test_diff_json(mini_model: Path) -> None:
    result = runner.invoke(app, ["diff", str(mini_model), str(mini_model), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, dict)
    assert data["added"] == []
    assert data["removed"] == []
    assert len(data["unchanged"]) > 0


def test_diff_cli_structural(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w1": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {
            "w1": np.zeros((4, 4), dtype=np.float32),
            "extra": np.ones((2,), dtype=np.float16),
        },
    )
    result = runner.invoke(app, ["diff", str(base), str(target)])
    assert result.exit_code == 0
    assert "Added (1):" in result.output
    assert "+ extra" in result.output
    assert "Unchanged: 1 tensors" in result.output


def test_diff_cli_delta(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": rng.randn(4, 4).astype(np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": rng.randn(4, 4).astype(np.float32)},
    )
    result = runner.invoke(app, ["diff", str(base), str(target), "--delta"])
    assert result.exit_code == 0
    assert "L2 norm" in result.output
    assert "cosine sim" in result.output
    assert "Summary:" in result.output
