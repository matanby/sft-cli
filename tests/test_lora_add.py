"""Tests for LoRA add (task arithmetic) operation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.lora.add import add_loras
from sft.ops.lora.conflict import (
    frob_inner_factored,
    gram_schmidt_orthogonalize,
    norm_scale_factor,
    validate_mode,
)
from sft.ops.lora.detect import detect_lora
from sft.utils.tensor_io import read_tensors

runner = CliRunner()


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


# --- Conflict-resolution helpers (pure functions) --------------------------

A_KEY = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
B_KEY = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"


def test_norm_scale_factor_matches_reference_norm() -> None:
    rng = np.random.RandomState(0)
    dw1 = rng.randn(8, 8)
    dw2 = 3.7 * rng.randn(8, 8)
    s = norm_scale_factor(dw1, dw2)
    assert np.isclose(np.linalg.norm(s * dw2), np.linalg.norm(dw1))


def test_norm_scale_factor_zero_other_is_noop() -> None:
    dw1 = np.ones((4, 4))
    dw2 = np.zeros((4, 4))
    assert norm_scale_factor(dw1, dw2) == 1.0


def test_gram_schmidt_orthogonalize_is_orthogonal() -> None:
    rng = np.random.RandomState(1)
    dw1 = rng.randn(8, 8)
    dw2 = rng.randn(8, 8)
    dw2_ortho = gram_schmidt_orthogonalize(dw1, dw2)
    overlap = abs(np.sum(dw1 * dw2_ortho))
    scale = np.linalg.norm(dw1) * np.linalg.norm(dw2)
    assert overlap <= 1e-9 * scale


def test_gram_schmidt_orthogonalize_zero_ref_is_noop() -> None:
    dw1 = np.zeros((4, 4))
    dw2 = np.arange(16, dtype=np.float64).reshape(4, 4)
    np.testing.assert_array_equal(gram_schmidt_orthogonalize(dw1, dw2), dw2)


def test_frob_inner_factored_matches_dense() -> None:
    rng = np.random.RandomState(2)
    a1 = rng.randn(4, 8).astype(np.float32)
    b1 = rng.randn(8, 4).astype(np.float32)
    a2 = rng.randn(4, 8).astype(np.float32)
    b2 = rng.randn(8, 4).astype(np.float32)
    dense = float(np.sum((b1 @ a1) * (b2 @ a2)))
    assert np.isclose(frob_inner_factored(a1, b1, a2, b2), dense, rtol=1e-5)


def test_validate_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        validate_mode("bogus")


# --- add --mode integration -------------------------------------------------


def test_add_norm_scaler_exact(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """On the shared-column-space fixture the combined delta is recoverable at
    rank 4, so it equals w0·dw1 + w1·(s·dw2) exactly."""
    p1, p2 = lora_pair
    out = tmp_path / "ns.safetensors"
    add_loras([p1, p2], weights=None, dst=out, mode="norm-scaler")

    t1, t2, out_t = read_tensors(p1), read_tensors(p2), read_tensors(out)
    dw1 = (t1[B_KEY] @ t1[A_KEY]).astype(np.float64)
    dw2 = (t2[B_KEY] @ t2[A_KEY]).astype(np.float64)
    s = norm_scale_factor(dw1, dw2)
    expected = 0.5 * dw1 + 0.5 * (s * dw2)
    actual = out_t[B_KEY] @ out_t[A_KEY]
    np.testing.assert_allclose(actual, expected, atol=1e-5)

    assert detect_lora(out).metadata.get("conflict_mode") == "norm-scaler"


def test_add_gram_schmidt_exact(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    out = tmp_path / "gs.safetensors"
    add_loras([p1, p2], weights=None, dst=out, mode="gram-schmidt")

    t1, t2, out_t = read_tensors(p1), read_tensors(p2), read_tensors(out)
    dw1 = (t1[B_KEY] @ t1[A_KEY]).astype(np.float64)
    dw2 = (t2[B_KEY] @ t2[A_KEY]).astype(np.float64)
    dw2_ortho = gram_schmidt_orthogonalize(dw1, dw2)
    expected = 0.5 * dw1 + 0.5 * dw2_ortho
    actual = out_t[B_KEY] @ out_t[A_KEY]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_add_mode_changes_output(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    none_out = tmp_path / "none.safetensors"
    gs_out = tmp_path / "gs.safetensors"
    add_loras([p1, p2], weights=None, dst=none_out, mode="none")
    add_loras([p1, p2], weights=None, dst=gs_out, mode="gram-schmidt")

    t_none, t_gs = read_tensors(none_out), read_tensors(gs_out)
    recon_none = t_none[B_KEY] @ t_none[A_KEY]
    recon_gs = t_gs[B_KEY] @ t_gs[A_KEY]
    assert not np.allclose(recon_none, recon_gs, atol=1e-6)


def test_add_zero_second_delta_leaves_reference(tmp_path: Path) -> None:
    """A zero-delta second adapter makes both modes no-ops: output == reference."""
    t1 = {A_KEY: np.eye(4, 8, dtype=np.float32), B_KEY: np.eye(8, 4, dtype=np.float32)}
    t2 = {
        A_KEY: np.zeros((4, 8), dtype=np.float32),
        B_KEY: np.zeros((8, 4), dtype=np.float32),
    }
    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file(t1, str(p1), metadata={"rank": "4"})
    save_file(t2, str(p2), metadata={"rank": "4"})

    for mode in ("norm-scaler", "gram-schmidt"):
        out = tmp_path / f"{mode}.safetensors"
        # weights [1, 1] so the reference is not down-scaled.
        add_loras([p1, p2], weights=[1.0, 1.0], dst=out, mode=mode)
        out_t = read_tensors(out)
        dw1 = np.eye(8, 4, dtype=np.float32) @ np.eye(4, 8, dtype=np.float32)
        np.testing.assert_allclose(out_t[B_KEY] @ out_t[A_KEY], dw1, atol=1e-5)


def test_add_invalid_mode_raises(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    with pytest.raises(ValueError, match="Invalid mode"):
        add_loras([p1, p2], weights=None, dst=tmp_path / "o.safetensors", mode="bogus")


def test_cli_add_mode(lora_pair: tuple[Path, Path], tmp_path: Path) -> None:
    p1, p2 = lora_pair
    out = tmp_path / "out.safetensors"
    result = runner.invoke(
        app, ["lora", "add", str(p1), str(p2), "--mode", "norm-scaler", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Mode: norm-scaler" in result.output
    assert out.exists()


def test_cli_add_invalid_mode(lora_pair: tuple[Path, Path]) -> None:
    p1, p2 = lora_pair
    result = runner.invoke(app, ["lora", "add", str(p1), str(p2), "--mode", "bogus"])
    assert result.exit_code == 2
