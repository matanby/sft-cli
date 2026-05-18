"""Tests for the stat command — per-tensor statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_stat_basic(mini_model: Path) -> None:
    result = runner.invoke(app, ["stat", str(mini_model)])
    assert result.exit_code == 0
    assert "model.embed_tokens.weight" in result.output
    assert "mean" in result.output
    assert "std" in result.output


def test_stat_reports_dtypes(mini_model: Path) -> None:
    result = runner.invoke(app, ["stat", str(mini_model)])
    assert result.exit_code == 0
    assert "fp16" in result.output
    assert "fp32" in result.output


def test_stat_include(mini_model: Path) -> None:
    result = runner.invoke(app, ["stat", str(mini_model), "--include", "**.norm.**"])
    assert result.exit_code == 0
    assert "norm" in result.output
    assert "q_proj" not in result.output


def test_stat_json(mini_model: Path) -> None:
    result = runner.invoke(app, ["stat", str(mini_model), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    assert "mean" in data[0]


def test_stat_check_healthy(mini_model: Path) -> None:
    result = runner.invoke(app, ["stat", str(mini_model), "--check"])
    assert result.exit_code == 0


def test_stat_check_nans(model_with_nans: Path) -> None:
    result = runner.invoke(app, ["stat", str(model_with_nans), "--check"])
    assert result.exit_code == 1
    assert "NaN" in (result.output + (result.stderr or ""))


def test_stat_sparsity(tmp_path: Path) -> None:
    arr = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    save_file({"test.weight": arr}, str(tmp_path / "sparse.safetensors"))

    result = runner.invoke(
        app, ["stat", str(tmp_path / "sparse.safetensors"), "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert abs(data[0]["sparsity"] - 0.5) < 1e-6
