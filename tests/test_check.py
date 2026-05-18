"""Tests for the check command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_check_healthy_file(mini_model: Path) -> None:
    result = runner.invoke(app, ["check", str(mini_model)])
    assert result.exit_code == 0
    assert "✓" in result.output
    assert "healthy" in result.output.lower()


def test_check_detects_nans(model_with_nans: Path) -> None:
    result = runner.invoke(app, ["check", str(model_with_nans)])
    assert result.exit_code == 1
    assert "NaN" in result.output or "nan" in result.output


def test_check_detects_inf(model_with_nans: Path) -> None:
    result = runner.invoke(app, ["check", str(model_with_nans)])
    assert result.exit_code == 1
    assert "Inf" in result.output or "inf" in result.output


def test_check_skip_values(model_with_nans: Path) -> None:
    result = runner.invoke(app, ["check", str(model_with_nans), "--skip-values"])
    assert result.exit_code == 0
    assert "skipped" in result.output.lower()


def test_check_corrupted_file(tmp_path: Path) -> None:
    bad = tmp_path / "corrupted.safetensors"
    bad.write_bytes(b"\x00" * 16)
    result = runner.invoke(app, ["check", str(bad)])
    assert result.exit_code == 1
    assert "✗" in result.output
