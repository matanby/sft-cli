"""Tests for CLI entry point and subcommand routing."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "sft" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer's no_args_is_help uses exit code 0 on some versions, 2 on others
    assert result.exit_code in (0, 2)
    assert "safetensors" in result.output.lower() or "usage" in result.output.lower()


def test_browse_subcommand_validates_extension(tmp_path: Path):
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("not a safetensors file")
    result = runner.invoke(app, ["browse", str(bad_file)])
    assert result.exit_code == 1
    assert "safetensors" in result.output.lower()


def test_browse_alias_validates_extension(tmp_path: Path):
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("not a safetensors file")
    result = runner.invoke(app, ["b", str(bad_file)])
    assert result.exit_code == 1
    assert "safetensors" in result.output.lower()


def test_browse_subcommand_rejects_missing_file():
    result = runner.invoke(app, ["browse", "/nonexistent/model.safetensors"])
    assert result.exit_code != 0
