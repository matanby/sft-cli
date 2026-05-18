"""Tests for sft lora resize command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from sft.ops.lora.resize import resize_lora


def test_resize_basic(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)

    assert result.original_rank == 4
    assert result.new_rank == 2
    assert result.modules_resized == 2
    assert dst.exists()


def test_resize_shapes(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    resize_lora(lora_adapter, dst, target_rank=2)

    tensors = load_file(str(dst))
    for name, t in tensors.items():
        if "lora_A" in name:
            assert t.shape[0] == 2, f"{name} rank dim should be 2, got {t.shape[0]}"
            assert t.shape[1] == 8, f"{name} in_features should be 8, got {t.shape[1]}"
        elif "lora_B" in name:
            assert t.shape[0] == 8, f"{name} out_features should be 8, got {t.shape[0]}"
            assert t.shape[1] == 2, f"{name} rank dim should be 2, got {t.shape[1]}"


def test_resize_reconstruction(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)

    old = load_file(str(lora_adapter))
    new = load_file(str(dst))

    for pair_key in [
        "base_model.model.model.layers.0.self_attn.q_proj",
        "base_model.model.model.layers.0.self_attn.v_proj",
    ]:
        a_name = f"{pair_key}.lora_A.weight"
        b_name = f"{pair_key}.lora_B.weight"

        old_delta = old[b_name] @ old[a_name]
        new_delta = new[b_name] @ new[a_name]

        error = np.linalg.norm(old_delta - new_delta, "fro") / np.linalg.norm(
            old_delta, "fro"
        )
        assert error < 1.0, f"Reconstruction error too large: {error}"

    for err in result.errors.values():
        assert err < 1.0


def test_resize_smart_output(lora_adapter: Path):
    from sft.utils.output import resolve_output

    dst = resolve_output(None, lora_adapter, "r2")
    assert dst.name == "adapter.r2.safetensors"


def test_resize_error_if_rank_too_large(lora_adapter: Path):
    import pytest

    with pytest.raises(ValueError, match="must be less than current rank"):
        resize_lora(lora_adapter, None, target_rank=8)

    with pytest.raises(ValueError, match="must be less than current rank"):
        resize_lora(lora_adapter, None, target_rank=4)


def test_resize_metadata_updated(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    resize_lora(lora_adapter, dst, target_rank=2)

    from sft.index import TensorIndex

    index = TensorIndex.from_file(dst)
    assert index.metadata["rank"] == "2"
    assert index.metadata["alpha"] == "8"


def test_resize_error_matches_energy(lora_adapter: Path, tmp_path: Path):
    """Reported error is consistent with the SVD energy not retained."""
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)

    old = load_file(str(lora_adapter))
    new = load_file(str(dst))

    for pair_key in [
        "base_model.model.model.layers.0.self_attn.q_proj",
        "base_model.model.model.layers.0.self_attn.v_proj",
    ]:
        a_name = f"{pair_key}.lora_A.weight"
        b_name = f"{pair_key}.lora_B.weight"
        target = pair_key.split(".")[-1]

        old_delta = old[b_name].astype(np.float64) @ old[a_name].astype(np.float64)
        new_delta = new[b_name].astype(np.float64) @ new[a_name].astype(np.float64)

        actual_error = np.linalg.norm(old_delta - new_delta, "fro") / np.linalg.norm(
            old_delta, "fro"
        )
        reported_error = result.errors[target]

        np.testing.assert_allclose(actual_error, reported_error, rtol=0.05)
