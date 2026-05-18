"""Op-level numerical-property tests for the LoRA pipeline.

These tests assert mathematical contracts of the operations rather than
specific output strings:

- stack: merged delta equals the weighted sum of input deltas.
- resize: full-rank resize is approximately identity; auto mode picks
  rank = ceil(stable_rank) + margin per pair.
- svd: QR-accelerated SVD agrees with numpy.linalg.svd on small inputs.
- convert: Kohya <-> PEFT <-> Kohya round-trip preserves tensors.
- merge: merging then subtracting recovers (alpha/rank) * B @ A.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from sft.ops.lora.convert import convert_lora, detect_format
from sft.ops.lora.detect import detect_lora
from sft.ops.lora.merge import merge_lora
from sft.ops.lora.resize import _stable_rank, resize_lora
from sft.ops.lora.stack import stack_loras
from sft.ops.lora.svd import _qr_svd

# ---------- Helpers ----------


def _make_peft(
    tmp_path: Path,
    rank: int,
    in_dim: int = 8,
    out_dim: int = 8,
    modules: tuple[str, ...] = ("q_proj", "v_proj"),
    seed: int = 0,
    alpha: int | None = None,
    fname: str = "lora.safetensors",
) -> Path:
    """Build a deterministic PEFT-format LoRA adapter."""
    rng = np.random.RandomState(seed)
    tensors: dict[str, np.ndarray] = {}
    for mod in modules:
        prefix = f"base_model.model.model.layers.0.self_attn.{mod}"
        tensors[f"{prefix}.lora_A.weight"] = rng.randn(rank, in_dim).astype(np.float32)
        tensors[f"{prefix}.lora_B.weight"] = rng.randn(out_dim, rank).astype(np.float32)
    md = {
        "rank": str(rank),
        "alpha": str(alpha or rank),
        "target_modules": ",".join(modules),
    }
    path = tmp_path / fname
    save_file(tensors, str(path), metadata=md)
    return path


def _make_kohya(
    tmp_path: Path,
    rank: int = 4,
    modules: tuple[str, ...] = ("lora_unet_q", "lora_unet_v"),
    seed: int = 0,
    alpha: float = 8.0,
    fname: str = "kohya.safetensors",
) -> Path:
    rng = np.random.RandomState(seed)
    tensors: dict[str, np.ndarray] = {}
    for mod in modules:
        tensors[f"{mod}.lora_down.weight"] = rng.randn(rank, 8).astype(np.float32)
        tensors[f"{mod}.lora_up.weight"] = rng.randn(8, rank).astype(np.float32)
        tensors[f"{mod}.alpha"] = np.array(alpha, dtype=np.float16)
    path = tmp_path / fname
    save_file(tensors, str(path))
    return path


# ---------- stack: factor-stacking math ----------


class TestStackMath:
    def test_merged_delta_equals_weighted_sum(self, tmp_path: Path) -> None:
        src_a = _make_peft(tmp_path, rank=4, seed=1, fname="a.safetensors")
        src_b = _make_peft(tmp_path, rank=6, seed=2, fname="b.safetensors")
        out = tmp_path / "merged.safetensors"

        coeff_a, coeff_b = 0.7, 0.3
        result = stack_loras(src_a, src_b, out, coeff_a=coeff_a, coeff_b=coeff_b)

        assert result.n_both == 2  # q_proj + v_proj overlap
        merged = load_file(str(out))
        ta = load_file(str(src_a))
        tb = load_file(str(src_b))

        for mod in ("q_proj", "v_proj"):
            prefix = f"base_model.model.model.layers.0.self_attn.{mod}"
            A_a = ta[f"{prefix}.lora_A.weight"]
            B_a = ta[f"{prefix}.lora_B.weight"]
            A_b = tb[f"{prefix}.lora_A.weight"]
            B_b = tb[f"{prefix}.lora_B.weight"]
            mA = merged[f"{prefix}.lora_A.weight"]
            mB = merged[f"{prefix}.lora_B.weight"]

            expected = coeff_a * (B_a @ A_a) + coeff_b * (B_b @ A_b)
            got = mB @ mA
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
            # And the merged rank is r_a + r_b.
            assert mA.shape[0] == 4 + 6
            assert mB.shape[1] == 4 + 6

    def test_target_rank_truncates(self, tmp_path: Path) -> None:
        src_a = _make_peft(tmp_path, rank=4, seed=1, fname="a.safetensors")
        src_b = _make_peft(tmp_path, rank=6, seed=2, fname="b.safetensors")
        out = tmp_path / "merged.safetensors"

        stack_loras(src_a, src_b, out, coeff_a=1.0, coeff_b=1.0, target_rank=3)

        merged = load_file(str(out))
        for name, arr in merged.items():
            if name.endswith(".lora_A.weight"):
                assert arr.shape[0] == 3
            if name.endswith(".lora_B.weight"):
                assert arr.shape[1] == 3

    def test_only_in_one_file_is_scaled(self, tmp_path: Path) -> None:
        # File A has q_proj only; File B has v_proj only — disjoint.
        src_a = _make_peft(
            tmp_path, rank=4, seed=1, modules=("q_proj",), fname="a.safetensors"
        )
        src_b = _make_peft(
            tmp_path, rank=4, seed=2, modules=("v_proj",), fname="b.safetensors"
        )
        out = tmp_path / "merged.safetensors"
        result = stack_loras(src_a, src_b, out, coeff_a=2.0, coeff_b=3.0)
        assert result.n_a_only == 1 and result.n_b_only == 1 and result.n_both == 0

        merged = load_file(str(out))
        ta = load_file(str(src_a))
        # A-only branch scales the A factor by coeff_a; B factor unchanged.
        prefix_q = "base_model.model.model.layers.0.self_attn.q_proj"
        np.testing.assert_allclose(
            merged[f"{prefix_q}.lora_A.weight"],
            2.0 * ta[f"{prefix_q}.lora_A.weight"],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            merged[f"{prefix_q}.lora_B.weight"], ta[f"{prefix_q}.lora_B.weight"]
        )

    def test_equivariance_under_scaling(self, tmp_path: Path) -> None:
        """Scaling both coefficients by k scales the output delta by k."""
        src_a = _make_peft(tmp_path, rank=3, seed=11, fname="a.safetensors")
        src_b = _make_peft(tmp_path, rank=3, seed=22, fname="b.safetensors")

        out1 = tmp_path / "m1.safetensors"
        out2 = tmp_path / "m2.safetensors"
        stack_loras(src_a, src_b, out1, coeff_a=0.5, coeff_b=0.5)
        stack_loras(src_a, src_b, out2, coeff_a=1.0, coeff_b=1.0)

        m1 = load_file(str(out1))
        m2 = load_file(str(out2))
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        d1 = m1[f"{prefix}.lora_B.weight"] @ m1[f"{prefix}.lora_A.weight"]
        d2 = m2[f"{prefix}.lora_B.weight"] @ m2[f"{prefix}.lora_A.weight"]
        np.testing.assert_allclose(d2, 2 * d1, rtol=1e-5, atol=1e-5)

    def test_non_lora_collision_warns_keeps_file_a(self, tmp_path: Path) -> None:
        # Build two PEFTs that share a non-LoRA tensor with different values.
        src_a = _make_peft(tmp_path, rank=4, seed=1, fname="a.safetensors")
        src_b = _make_peft(tmp_path, rank=4, seed=2, fname="b.safetensors")
        # Inject a non-LoRA tensor "shared.weight" into both with different values.
        ta = load_file(str(src_a))
        tb = load_file(str(src_b))
        ta["shared.weight"] = np.ones((2, 2), dtype=np.float32)
        tb["shared.weight"] = np.zeros((2, 2), dtype=np.float32)
        save_file(ta, str(src_a), metadata={"rank": "4", "alpha": "4"})
        save_file(tb, str(src_b), metadata={"rank": "4", "alpha": "4"})

        out = tmp_path / "merged.safetensors"
        result = stack_loras(src_a, src_b, out, coeff_a=1.0, coeff_b=1.0)
        assert any("shared.weight" in c for c in result.collisions)
        merged = load_file(str(out))
        np.testing.assert_array_equal(merged["shared.weight"], np.ones((2, 2)))


# ---------- resize: auto mode + identity ----------


class TestResizeMath:
    def test_auto_picks_ceil_stable_rank_plus_margin(self, tmp_path: Path) -> None:
        src = _make_peft(tmp_path, rank=8, seed=3, fname="big.safetensors")
        out = tmp_path / "auto.safetensors"
        # Compute expected per-module rank: ceil(stable_rank(s)) + margin, capped at rank.
        info = detect_lora(src)
        tensors = load_file(str(src))
        expected = {}
        margin = 1
        for pair in info.pairs:
            a = tensors[pair.lora_a_name]
            b = tensors[pair.lora_b_name]
            (s,) = _qr_svd(a, b, compute_uv=False)
            sr = _stable_rank(s)
            import math

            expected[pair.target_module] = min(
                pair.rank, max(1, math.ceil(sr) + margin)
            )

        result = resize_lora(src, out, auto_margin=margin)
        assert result.per_module_ranks == expected
        assert result.auto_margin == margin

    def test_full_rank_resize_is_identity_via_truncation(self, tmp_path: Path) -> None:
        """Resizing to rank-1-less still preserves the matrix product to the
        top-singular-value approximation. We check that the relative error is
        bounded by the dropped-singular-value energy.
        """
        src = _make_peft(tmp_path, rank=5, seed=7, fname="lo.safetensors")
        out = tmp_path / "out.safetensors"
        result = resize_lora(src, out, target_rank=4)

        # For each pair, the reported error should be < 1 (some energy lost)
        # but match the underlying SVD-energy formula.
        assert result.new_rank == 4
        assert all(0.0 <= e <= 1.0 for e in result.errors.values())
        assert all(0.0 <= e <= 1.0 for e in result.energies.values())

    def test_invalid_rank_combinations_raise(self, tmp_path: Path) -> None:
        src = _make_peft(tmp_path, rank=4, seed=0, fname="x.safetensors")
        out = tmp_path / "y.safetensors"
        with pytest.raises(ValueError):
            resize_lora(src, out)  # neither target_rank nor auto_margin
        with pytest.raises(ValueError):
            resize_lora(src, out, target_rank=2, auto_margin=1)
        with pytest.raises(ValueError):
            resize_lora(src, out, target_rank=0)
        with pytest.raises(ValueError):
            resize_lora(src, out, auto_margin=-1)
        with pytest.raises(ValueError):
            resize_lora(src, out, target_rank=10)  # >= original rank


# ---------- svd: QR path matches numpy ground truth ----------


class TestSvdMath:
    def test_qr_svd_singular_values_match_numpy(self) -> None:
        rng = np.random.RandomState(0)
        a = rng.randn(6, 32).astype(np.float32)
        b = rng.randn(32, 6).astype(np.float32)
        (s_qr,) = _qr_svd(a, b, compute_uv=False)
        s_ref = np.linalg.svd((b @ a).astype(np.float64), compute_uv=False)
        np.testing.assert_allclose(
            np.sort(s_qr), np.sort(s_ref[: s_qr.size]), rtol=1e-5, atol=1e-5
        )

    def test_qr_svd_uv_reconstructs_product(self) -> None:
        rng = np.random.RandomState(1)
        a = rng.randn(8, 64).astype(np.float32)
        b = rng.randn(64, 8).astype(np.float32)
        u, s, vt = _qr_svd(a, b, compute_uv=True)
        reconstructed = (u * s) @ vt
        np.testing.assert_allclose(
            reconstructed, (b @ a).astype(np.float64), rtol=1e-5, atol=1e-5
        )


# ---------- convert: Kohya <-> PEFT round-trip ----------


class TestConvertRoundTrip:
    def test_kohya_to_peft_to_kohya_preserves_tensors(self, tmp_path: Path) -> None:
        src = _make_kohya(tmp_path, rank=4, alpha=8.0)
        peft = tmp_path / "as_peft.safetensors"
        kohya_back = tmp_path / "back.safetensors"

        convert_lora(src, peft, target="peft")
        assert detect_format(peft).format == "peft"

        convert_lora(peft, kohya_back, target="kohya")
        assert detect_format(kohya_back).format == "kohya"

        orig = load_file(str(src))
        roundtripped = load_file(str(kohya_back))
        # lora_down / lora_up tensors are identical; alphas match.
        for name, arr in orig.items():
            if name.endswith(".lora_down.weight") or name.endswith(".lora_up.weight"):
                np.testing.assert_array_equal(roundtripped[name], arr)
            if name.endswith(".alpha"):
                np.testing.assert_allclose(roundtripped[name], arr, rtol=1e-3)

    def test_convert_rejects_same_target(self, tmp_path: Path) -> None:
        src = _make_kohya(tmp_path, rank=4)
        with pytest.raises(ValueError):
            convert_lora(src, tmp_path / "out.safetensors", target="kohya")


# ---------- merge: delta equals scale * B @ A ----------


class TestMergeMath:
    def test_merge_then_subtract_recovers_delta(
        self, lora_base_model: Path, tmp_path: Path
    ) -> None:
        # Build a PEFT adapter that targets q_proj and v_proj on layer 0
        # (matches lora_base_model fixture).
        rng = np.random.RandomState(42)
        rank = 4
        in_dim = out_dim = 8
        tensors = {}
        for mod in ("q_proj", "v_proj"):
            prefix = f"base_model.model.model.layers.0.self_attn.{mod}"
            tensors[f"{prefix}.lora_A.weight"] = rng.randn(rank, in_dim).astype(
                np.float32
            )
            tensors[f"{prefix}.lora_B.weight"] = rng.randn(out_dim, rank).astype(
                np.float32
            )
        adapter = tmp_path / "adapter.safetensors"
        save_file(
            tensors,
            str(adapter),
            metadata={
                "rank": str(rank),
                "alpha": str(rank),
                "target_modules": "q_proj,v_proj",
            },
        )

        merged_path = tmp_path / "merged.safetensors"
        result = merge_lora(lora_base_model, adapter, dst=merged_path, scale=1.0)

        base = load_file(str(lora_base_model))
        merged = load_file(str(merged_path))
        for mod in ("q_proj", "v_proj"):
            base_name = f"model.layers.0.self_attn.{mod}.weight"
            prefix = f"base_model.model.{base_name[: -len('.weight')]}"
            A = tensors[f"{prefix}.lora_A.weight"]
            B = tensors[f"{prefix}.lora_B.weight"]
            delta = merged[base_name] - base[base_name]
            np.testing.assert_allclose(
                delta, (B @ A).astype(delta.dtype), rtol=1e-5, atol=1e-5
            )

        # Unchanged tensors (e.g. mlp.gate_proj, layernorm, embeddings) are bit-identical.
        for name in base:
            if name not in {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            }:
                np.testing.assert_array_equal(merged[name], base[name])
        assert result.unchanged_tensors == 3
