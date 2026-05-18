"""Tests for the info command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_info_shows_file_summary(mini_model: Path) -> None:
    result = runner.invoke(app, ["info", str(mini_model)])
    assert result.exit_code == 0
    assert mini_model.name in result.output
    assert "14" in result.output
    assert "fp16" in result.output


def test_info_shows_metadata(mini_model: Path) -> None:
    result = runner.invoke(app, ["info", str(mini_model)])
    assert result.exit_code == 0
    assert "llama" in result.output


def test_info_json_output(mini_model: Path) -> None:
    result = runner.invoke(app, ["info", str(mini_model), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["file"] == mini_model.name
    assert data["tensors"] == 14
    assert "total_parameters" in data
    assert "file_size" in data
    assert "total_tensor_bytes" in data
    assert "dtypes" in data
    assert "metadata" in data
    assert data["metadata"]["format"] == "pt"


def test_info_rejects_non_safetensors(tmp_path: Path) -> None:
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello")
    result = runner.invoke(app, ["info", str(txt_file)])
    assert result.exit_code == 1


def test_info_rejects_missing_file() -> None:
    result = runner.invoke(app, ["info", "/nonexistent/model.safetensors"])
    assert result.exit_code != 0


def test_info_alias(mini_model: Path) -> None:
    result = runner.invoke(app, ["i", str(mini_model)])
    assert result.exit_code == 0
    assert mini_model.name in result.output
    assert "14" in result.output
