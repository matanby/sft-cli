"""Tests for sft lora merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from sft.ops.lora.merge import merge_lora


def test_merge_basic(lora_base_model: Path, lora_adapter: Path, tmp_path: Path):
    """Merged file has same tensor names as base, and merged weights differ."""
    out = tmp_path / "merged.safetensors"
    result = merge_lora(lora_base_model, lora_adapter, dst=out)

    assert out.exists()
    assert len(result.merged_modules) == 2

    base = load_file(str(lora_base_model))
    merged = load_file(str(out))

    assert set(merged.keys()) == set(base.keys())

    for name in result.merged_modules:
        assert not np.array_equal(merged[name], base[name])


def test_merge_scale(lora_base_model: Path, lora_adapter: Path, tmp_path: Path):
    """Scale 0.0 produces weights identical to the base model."""
    out = tmp_path / "merged.safetensors"
    result = merge_lora(lora_base_model, lora_adapter, dst=out, scale=0.0)

    assert result.scale == 0.0

    base = load_file(str(lora_base_model))
    merged = load_file(str(out))

    for name in base:
        np.testing.assert_array_equal(merged[name], base[name])


def test_merge_non_target_unchanged(
    lora_base_model: Path, lora_adapter: Path, tmp_path: Path
):
    """Non-target tensors (embed, layernorm, gate_proj) are byte-identical."""
    out = tmp_path / "merged.safetensors"
    result = merge_lora(lora_base_model, lora_adapter, dst=out)

    base = load_file(str(lora_base_model))
    merged = load_file(str(out))

    non_target = set(base.keys()) - set(result.merged_modules)
    assert len(non_target) > 0

    for name in non_target:
        np.testing.assert_array_equal(merged[name], base[name])


def test_merge_smart_output(lora_base_model: Path, lora_adapter: Path):
    """Default output path is {base_stem}.merged.safetensors."""
    result = merge_lora(lora_base_model, lora_adapter)

    expected = lora_base_model.parent / "base_model.merged.safetensors"
    assert result.output_path == expected
    assert expected.exists()


def test_merge_dry_run(lora_base_model: Path, lora_adapter: Path):
    """Dry run reports modules but writes no file."""
    result = merge_lora(lora_base_model, lora_adapter, dry_run=True)

    assert len(result.merged_modules) == 2
    assert result.output_path is not None

    default_out = lora_base_model.parent / "base_model.merged.safetensors"
    assert not default_out.exists()
