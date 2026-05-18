"""Tests for the tree command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_tree_shows_hierarchy(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model)])
    assert result.exit_code == 0
    for keyword in ("model", "layers", "self_attn", "q_proj"):
        assert keyword in result.output


def test_tree_shows_tensor_info(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model)])
    assert result.exit_code == 0
    assert "fp16" in result.output


def test_tree_depth_limit(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model), "--depth", "1"])
    assert result.exit_code == 0
    assert "layers" in result.output
    assert "q_proj" not in result.output


def test_tree_uses_box_drawing(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model)])
    assert result.exit_code == 0
    assert "├" in result.output
    assert "└" in result.output


def test_tree_alias(mini_model: Path):
    result = runner.invoke(app, ["t", str(mini_model)])
    assert result.exit_code == 0
    assert "model" in result.output
