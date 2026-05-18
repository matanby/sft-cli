"""Tests for sft lora compat."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.lora.compat import check_compat

runner = CliRunner()


def test_compat_compatible(lora_base_model: Path, lora_adapter: Path):
    """Compatible base model and adapter produces a passing result."""
    result = check_compat(lora_base_model, lora_adapter)

    assert result.compatible is True
    assert result.matched_modules == 2
    assert result.missing_modules == []
    assert result.shape_mismatches == []


def test_compat_missing_module(tmp_path: Path, lora_adapter: Path):
    """Base model missing a target module is reported as incompatible."""
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((8, 8), dtype=np.float32),
    }
    base = tmp_path / "partial_base.safetensors"
    save_file(tensors, str(base))

    result = check_compat(base, lora_adapter)

    assert result.compatible is False
    assert "v_proj" in result.missing_modules
    assert result.matched_modules == 1


def test_compat_shape_mismatch(tmp_path: Path, lora_adapter: Path):
    """Base model with wrong shape is reported as incompatible."""
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones(
            (2048, 2048), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.ones(
            (2048, 2048), dtype=np.float32
        ),
    }
    base = tmp_path / "wrong_shape_base.safetensors"
    save_file(tensors, str(base))

    result = check_compat(base, lora_adapter)

    assert result.compatible is False
    assert len(result.shape_mismatches) == 2
    assert result.shape_mismatches[0].module in ("q_proj", "v_proj")


def test_compat_non_lora(lora_base_model: Path, mini_model: Path):
    """Passing a non-LoRA file as adapter raises ValueError."""
    with pytest.raises(ValueError, match="No LoRA pairs found"):
        check_compat(lora_base_model, mini_model)


def test_compat_cli_compatible(lora_base_model: Path, lora_adapter: Path):
    """CLI exits 0 and prints 'Compatible: yes' for a matching pair."""
    result = runner.invoke(
        app, ["lora", "compat", str(lora_base_model), str(lora_adapter)]
    )

    assert result.exit_code == 0
    assert "Compatible: yes" in result.output


def test_compat_cli_incompatible(tmp_path: Path, lora_adapter: Path):
    """CLI exits 1 and prints 'Compatible: no' when modules are missing."""
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((8, 8), dtype=np.float32),
    }
    base = tmp_path / "partial_base.safetensors"
    save_file(tensors, str(base))

    result = runner.invoke(app, ["lora", "compat", str(base), str(lora_adapter)])

    assert result.exit_code == 1
    assert "Compatible: no" in result.output
