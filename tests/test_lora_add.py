"""Tests for LoRA add (task arithmetic) operation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from sft.ops.lora.add import add_loras
from sft.ops.lora.detect import detect_lora
from sft.utils.tensor_io import read_tensors


@pytest.fixture
def lora_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two compatible LoRA adapters with known values.

    Both use eye(8,4) for B so the combined delta's column space stays
    within rank 4, making the SVD re-decomposition lossless at rank 4.
    """
    t1 = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.eye(
            4, 8, dtype=np.float32
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.eye(
            8, 4, dtype=np.float32
        ),
    }
    t2 = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
            np.ones((4, 8), dtype=np.float32) * 0.5
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.eye(
            8, 4, dtype=np.float32
        ),
    }
    p1 = tmp_path / "lora1.safetensors"
    p2 = tmp_path / "lora2.safetensors"
    save_file(t1, str(p1), metadata={"rank": "4", "alpha": "4"})
    save_file(t2, str(p2), metadata={"rank": "4", "alpha": "4"})
    return p1, p2


def test_add_equal_weights(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    out = tmp_path / "out.safetensors"

    result = add_loras([p1, p2], weights=None, dst=out)

    assert result.combined_modules == 1
    assert result.output_rank == 4
    assert result.output_path == out
    assert out.exists()

    tensors = read_tensors(out)
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in tensors
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" in tensors

    a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert a.shape == (4, 8)
    assert b.shape == (8, 4)

    reconstructed = b @ a
    delta1 = np.eye(8, 4, dtype=np.float32) @ np.eye(4, 8, dtype=np.float32)
    delta2 = np.eye(8, 4, dtype=np.float32) @ (np.ones((4, 8), dtype=np.float32) * 0.5)
    expected = 0.5 * delta1 + 0.5 * delta2
    np.testing.assert_allclose(reconstructed, expected, atol=1e-5)


def test_add_custom_weights(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    out_equal = tmp_path / "equal.safetensors"
    out_custom = tmp_path / "custom.safetensors"

    add_loras([p1, p2], weights=None, dst=out_equal)
    add_loras([p1, p2], weights=[0.7, 0.3], dst=out_custom)

    t_equal = read_tensors(out_equal)
    t_custom = read_tensors(out_custom)

    a_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    b_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"

    recon_equal = t_equal[b_key] @ t_equal[a_key]
    recon_custom = t_custom[b_key] @ t_custom[a_key]

    assert not np.allclose(recon_equal, recon_custom, atol=1e-6)

    delta1 = np.eye(8, 4, dtype=np.float32) @ np.eye(4, 8, dtype=np.float32)
    delta2 = np.eye(8, 4, dtype=np.float32) @ (np.ones((4, 8), dtype=np.float32) * 0.5)
    expected_custom = 0.7 * delta1 + 0.3 * delta2
    np.testing.assert_allclose(recon_custom, expected_custom, atol=1e-5)


def test_add_output_rank(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    out = tmp_path / "rank2.safetensors"

    result = add_loras([p1, p2], weights=None, dst=out, output_rank=2)

    assert result.output_rank == 2
    tensors = read_tensors(out)

    a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert a.shape == (2, 8)
    assert b.shape == (8, 2)

    info = detect_lora(out)
    assert info is not None
    assert info.rank == 2


def test_add_default_output(
    lora_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p1, p2 = lora_pair
    monkeypatch.chdir(tmp_path)

    result = add_loras([p1, p2], weights=None)

    assert result.output_path == Path("combined.safetensors")
    assert (tmp_path / "combined.safetensors").exists()


def test_add_incompatible_error(
    lora_adapter: Path, mini_model: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="No LoRA pairs found"):
        add_loras(
            [lora_adapter, mini_model], weights=None, dst=tmp_path / "out.safetensors"
        )
