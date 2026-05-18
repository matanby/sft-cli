"""Tests for sft lora svd command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app
from sft.ops.lora.svd import analyze_svd

runner = CliRunner()


def test_svd_basic(lora_adapter: Path):
    analysis = analyze_svd(lora_adapter)
    module_names = [m.module for m in analysis.modules]
    assert "q_proj" in module_names
    assert "v_proj" in module_names
    assert len(analysis.modules) == 2


def test_svd_suggested_rank(lora_adapter: Path):
    analysis = analyze_svd(lora_adapter)
    for m in analysis.modules:
        assert 1 <= m.suggested_rank <= m.rank


def test_svd_threshold(lora_adapter: Path):
    low = analyze_svd(lora_adapter, threshold=0.90)
    high = analyze_svd(lora_adapter, threshold=0.99)
    for lo, hi in zip(low.modules, high.modules):
        assert lo.suggested_rank <= hi.suggested_rank


def test_svd_json(lora_adapter: Path):
    result = runner.invoke(app, ["lora", "svd", str(lora_adapter), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "threshold" in data
    assert "modules" in data
    assert len(data["modules"]) == 2
    for mod in data["modules"]:
        assert "module" in mod
        assert "rank" in mod
        assert "sv_90" in mod
        assert "sv_95" in mod
        assert "sv_99" in mod
        assert "suggested_rank" in mod
        assert "singular_values" in mod


def test_svd_non_lora(mini_model: Path):
    result = runner.invoke(app, ["lora", "svd", str(mini_model)])
    assert result.exit_code == 1
    assert "not a lora" in result.output.lower() or "error" in result.output.lower()
