"""Tests for sft lora info command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_lora_info_shows_rank(lora_adapter: Path):
    result = runner.invoke(app, ["lora", "info", str(lora_adapter)])
    assert result.exit_code == 0
    assert "4" in result.output  # rank
    assert "q_proj" in result.output
    assert "v_proj" in result.output


def test_lora_info_shows_alpha(lora_adapter: Path):
    result = runner.invoke(app, ["lora", "info", str(lora_adapter)])
    assert result.exit_code == 0
    assert "8" in result.output  # alpha
    assert "2.0" in result.output  # effective scale


def test_lora_info_json(lora_adapter: Path):
    result = runner.invoke(app, ["lora", "info", str(lora_adapter), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["rank"] == 4
    assert data["alpha"] == 8.0
    assert "q_proj" in data["target_modules"]


def test_lora_info_non_lora(mini_model: Path):
    result = runner.invoke(app, ["lora", "info", str(mini_model)])
    assert result.exit_code == 1
    assert "not a lora" in result.output.lower() or "error" in result.output.lower()
