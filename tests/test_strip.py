"""Tests for the strip command — remove tensors matching a pattern."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.numpy import safe_open
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_strip_removes_matching_tensors(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "stripped.safetensors"
    result = runner.invoke(
        app,
        ["strip", str(mini_model), "--exclude", "**.q_proj.weight", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    with safe_open(str(out), framework="numpy") as f:
        keys = list(f.keys())
    assert not any("q_proj" in k for k in keys)
    assert any("k_proj" in k for k in keys)


def test_strip_default_output_path(mini_model: Path) -> None:
    result = runner.invoke(
        app, ["strip", str(mini_model), "--exclude", "**.q_proj.weight"]
    )
    assert result.exit_code == 0, result.output
    default = mini_model.with_name(mini_model.stem + ".stripped.safetensors")
    assert default.exists()
    assert default != mini_model


def test_strip_dry_run_does_not_write(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "would_be.safetensors"
    result = runner.invoke(
        app,
        [
            "strip",
            str(mini_model),
            "--exclude",
            "**.q_proj.weight",
            "-o",
            str(out),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert "Would" in result.output or "would" in result.output


def test_strip_json_happy_path(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "stripped.safetensors"
    result = runner.invoke(
        app,
        [
            "strip",
            str(mini_model),
            "--exclude",
            "**.q_proj.weight",
            "-o",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "included" in data
    assert "excluded" in data
    assert all("q_proj" in n for n in data["excluded"])
    assert data["dry_run"] is False
    assert data["output_path"] == str(out)


def test_strip_json_dry_run_omits_output_path(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "stripped.safetensors"
    result = runner.invoke(
        app,
        [
            "strip",
            str(mini_model),
            "--exclude",
            "**.q_proj.weight",
            "-o",
            str(out),
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["dry_run"] is True
    assert data["output_path"] is None
    assert not out.exists()


def test_strip_no_match_leaves_all(mini_model: Path, tmp_path: Path) -> None:
    out = tmp_path / "stripped.safetensors"
    result = runner.invoke(
        app,
        [
            "strip",
            str(mini_model),
            "--exclude",
            "**.does_not_exist",
            "-o",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["excluded"] == []
    assert len(data["included"]) == 14


def test_strip_rejects_non_safetensors(tmp_path: Path) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    result = runner.invoke(app, ["strip", str(txt), "--exclude", "**"])
    assert result.exit_code != 0


def test_strip_no_args_shows_help() -> None:
    result = runner.invoke(app, ["strip"])
    assert result.exit_code != 0
    assert "Usage" in result.output or "usage" in result.output
