"""Tests for sft lora stack (lossless weighted concat merge)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.lora.detect import detect_lora
from sft.ops.lora.stack import stack_loras

runner = CliRunner()


def _make_lora(
    path: Path,
    modules: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Build a tiny PEFT-format LoRA file from a {module_key: (A, B)} dict."""
    tensors: dict[str, np.ndarray] = {}
    for module_key, (A, B) in modules.items():
        tensors[f"{module_key}.lora_A.weight"] = A
        tensors[f"{module_key}.lora_B.weight"] = B
    save_file(tensors, str(path), metadata=metadata or {})
    return path


@pytest.fixture
def lora_a(tmp_path: Path) -> Path:
    """LoRA with q_proj and v_proj at rank 4, in/out dims = 8."""
    rng = np.random.RandomState(0)
    rank = 4
    return _make_lora(
        tmp_path / "a.safetensors",
        {
            "base_model.model.layers.0.q_proj": (
                rng.randn(rank, 8).astype(np.float32),
                rng.randn(8, rank).astype(np.float32),
            ),
            "base_model.model.layers.0.v_proj": (
                rng.randn(rank, 8).astype(np.float32),
                rng.randn(8, rank).astype(np.float32),
            ),
        },
        metadata={"rank": "4", "alpha": "8"},
    )


@pytest.fixture
def lora_b(tmp_path: Path) -> Path:
    """LoRA with q_proj and v_proj at rank 4, in/out dims = 8."""
    rng = np.random.RandomState(1)
    rank = 4
    return _make_lora(
        tmp_path / "b.safetensors",
        {
            "base_model.model.layers.0.q_proj": (
                rng.randn(rank, 8).astype(np.float32),
                rng.randn(8, rank).astype(np.float32),
            ),
            "base_model.model.layers.0.v_proj": (
                rng.randn(rank, 8).astype(np.float32),
                rng.randn(8, rank).astype(np.float32),
            ),
        },
        metadata={"rank": "4", "alpha": "8"},
    )


class TestLossless:
    def test_effective_delta_exact(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        """merged_B @ merged_A == coeff_a · (B_a @ A_a) + coeff_b · (B_b @ A_b) exactly."""
        dst = tmp_path / "out.safetensors"
        ca, cb = 0.7, 0.3
        stack_loras(lora_a, lora_b, dst, coeff_a=ca, coeff_b=cb)

        ta = load_file(str(lora_a))
        tb = load_file(str(lora_b))
        out = load_file(str(dst))

        for module in (
            "base_model.model.layers.0.q_proj",
            "base_model.model.layers.0.v_proj",
        ):
            delta_a = ta[f"{module}.lora_B.weight"] @ ta[f"{module}.lora_A.weight"]
            delta_b = tb[f"{module}.lora_B.weight"] @ tb[f"{module}.lora_A.weight"]
            expected = ca * delta_a + cb * delta_b
            actual = out[f"{module}.lora_B.weight"] @ out[f"{module}.lora_A.weight"]
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_merged_rank_is_sum(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        """Without --target-rank, output rank per pair equals r_a + r_b."""
        dst = tmp_path / "out.safetensors"
        stack_loras(lora_a, lora_b, dst)
        out = load_file(str(dst))
        for module in (
            "base_model.model.layers.0.q_proj",
            "base_model.model.layers.0.v_proj",
        ):
            A = out[f"{module}.lora_A.weight"]
            B = out[f"{module}.lora_B.weight"]
            assert A.shape[0] == 8  # 4 + 4
            assert B.shape[1] == 8

    def test_target_rank_truncates(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        """--target-rank applies SVD truncation after stacking."""
        dst = tmp_path / "out.safetensors"
        result = stack_loras(lora_a, lora_b, dst, target_rank=3)
        out = load_file(str(dst))
        for module in (
            "base_model.model.layers.0.q_proj",
            "base_model.model.layers.0.v_proj",
        ):
            A = out[f"{module}.lora_A.weight"]
            B = out[f"{module}.lora_B.weight"]
            assert A.shape[0] == 3
            assert B.shape[1] == 3
        assert result.target_rank == 3


class TestUnionModules:
    def test_modules_only_in_a_kept(self, tmp_path: Path) -> None:
        """Modules present only in file A are kept and scaled by coeff_a."""
        rng = np.random.RandomState(0)
        rank = 2
        only_a = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        shared = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        a = _make_lora(
            tmp_path / "a.safetensors",
            {"base_model.model.q": only_a, "base_model.model.shared": shared},
        )
        b = _make_lora(
            tmp_path / "b.safetensors",
            {"base_model.model.shared": shared},
        )
        dst = tmp_path / "out.safetensors"
        result = stack_loras(a, b, dst, coeff_a=0.5, coeff_b=1.0)
        assert result.n_a_only == 1
        assert result.n_b_only == 0
        assert result.n_both == 1

        out = load_file(str(dst))
        # The A-only module's A factor should be 0.5 * original
        expected_A = 0.5 * only_a[0]
        np.testing.assert_allclose(out["base_model.model.q.lora_A.weight"], expected_A)

    def test_modules_only_in_b_kept(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(0)
        rank = 2
        only_b = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        shared = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        a = _make_lora(
            tmp_path / "a.safetensors",
            {"base_model.model.shared": shared},
        )
        b = _make_lora(
            tmp_path / "b.safetensors",
            {"base_model.model.q": only_b, "base_model.model.shared": shared},
        )
        dst = tmp_path / "out.safetensors"
        result = stack_loras(a, b, dst, coeff_a=1.0, coeff_b=0.3)
        assert result.n_a_only == 0
        assert result.n_b_only == 1
        assert result.n_both == 1

        out = load_file(str(dst))
        np.testing.assert_allclose(
            out["base_model.model.q.lora_A.weight"], 0.3 * only_b[0]
        )


class TestShapeMismatch:
    def test_in_features_mismatch_skipped(self, tmp_path: Path) -> None:
        """A pair with different in_features between files is skipped with a count."""
        rng = np.random.RandomState(0)
        a_pair = (
            rng.randn(4, 8).astype(np.float32),
            rng.randn(8, 4).astype(np.float32),
        )
        b_pair = (
            rng.randn(4, 16).astype(np.float32),
            rng.randn(8, 4).astype(np.float32),
        )
        a = _make_lora(tmp_path / "a.safetensors", {"base_model.model.q": a_pair})
        b = _make_lora(tmp_path / "b.safetensors", {"base_model.model.q": b_pair})
        dst = tmp_path / "out.safetensors"
        result = stack_loras(a, b, dst)
        assert result.n_skipped_shape == 1
        assert result.n_both == 0


class TestPassthroughCollisions:
    def test_passthrough_only_in_one(self, tmp_path: Path) -> None:
        """Non-LoRA tensors only in one file pass through."""
        rng = np.random.RandomState(0)
        rank = 2
        pair = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        a = _make_lora(tmp_path / "a.safetensors", {"base_model.model.q": pair})
        b_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "extra_bias": np.ones(4, dtype=np.float32),
        }
        save_file(b_tensors, str(tmp_path / "b.safetensors"))
        b = tmp_path / "b.safetensors"

        dst = tmp_path / "out.safetensors"
        result = stack_loras(a, b, dst)
        out = load_file(str(dst))
        assert "extra_bias" in out
        assert result.n_passthrough == 1
        assert result.collisions == []

    def test_collision_warns_on_value_difference(self, tmp_path: Path) -> None:
        """Same name + shape + dtype but different values -> warn, keep A's."""
        rng = np.random.RandomState(0)
        rank = 2
        pair = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        a_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "shared_bias": np.zeros(4, dtype=np.float32),
        }
        b_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "shared_bias": np.ones(4, dtype=np.float32),
        }
        save_file(a_tensors, str(tmp_path / "a.safetensors"))
        save_file(b_tensors, str(tmp_path / "b.safetensors"))

        dst = tmp_path / "out.safetensors"
        result = stack_loras(
            tmp_path / "a.safetensors", tmp_path / "b.safetensors", dst
        )
        out = load_file(str(dst))
        # File A's version wins
        np.testing.assert_array_equal(out["shared_bias"], np.zeros(4, dtype=np.float32))
        assert any("shared_bias" in msg for msg in result.collisions)

    def test_collision_warns_on_shape_mismatch_and_skips(self, tmp_path: Path) -> None:
        """Same name but different shape/dtype -> warn and skip."""
        rng = np.random.RandomState(0)
        rank = 2
        pair = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        a_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "weird": np.zeros(4, dtype=np.float32),
        }
        b_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "weird": np.zeros(8, dtype=np.float32),
        }
        save_file(a_tensors, str(tmp_path / "a.safetensors"))
        save_file(b_tensors, str(tmp_path / "b.safetensors"))

        dst = tmp_path / "out.safetensors"
        result = stack_loras(
            tmp_path / "a.safetensors", tmp_path / "b.safetensors", dst
        )
        out = load_file(str(dst))
        # A's "weird" still passes through; B's is skipped from collision
        assert out["weird"].shape == (4,)
        assert any("weird" in msg and "shape/dtype" in msg for msg in result.collisions)

    def test_identical_passthrough_no_warning(self, tmp_path: Path) -> None:
        """Identical non-LoRA tensors in both files: copy once, no warning."""
        rng = np.random.RandomState(0)
        rank = 2
        pair = (
            rng.randn(rank, 4).astype(np.float32),
            rng.randn(4, rank).astype(np.float32),
        )
        bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "shared_bias": bias,
        }
        b_tensors = {
            "base_model.model.q.lora_A.weight": pair[0],
            "base_model.model.q.lora_B.weight": pair[1],
            "shared_bias": bias.copy(),
        }
        save_file(a_tensors, str(tmp_path / "a.safetensors"))
        save_file(b_tensors, str(tmp_path / "b.safetensors"))

        result = stack_loras(
            tmp_path / "a.safetensors",
            tmp_path / "b.safetensors",
            tmp_path / "out.safetensors",
        )
        assert result.collisions == []


MODULES = (
    "base_model.model.layers.0.q_proj",
    "base_model.model.layers.0.v_proj",
)


class TestConflictModes:
    """--mode transforms file B against file A (reference) but stays lossless."""

    def test_norm_scaler_lossless_and_matches_norm(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        ca, cb = 0.7, 0.3
        stack_loras(lora_a, lora_b, dst, coeff_a=ca, coeff_b=cb, mode="norm-scaler")

        ta, tb, out = (
            load_file(str(lora_a)),
            load_file(str(lora_b)),
            load_file(str(dst)),
        )
        for module in MODULES:
            dw_a = (
                ta[f"{module}.lora_B.weight"] @ ta[f"{module}.lora_A.weight"]
            ).astype(np.float64)
            dw_b = (
                tb[f"{module}.lora_B.weight"] @ tb[f"{module}.lora_A.weight"]
            ).astype(np.float64)
            s = np.linalg.norm(dw_a) / np.linalg.norm(dw_b)
            expected = ca * dw_a + cb * s * dw_b
            actual = out[f"{module}.lora_B.weight"] @ out[f"{module}.lora_A.weight"]
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            # scaled B delta matches A's norm
            assert np.isclose(np.linalg.norm(s * dw_b), np.linalg.norm(dw_a))

    def test_gram_schmidt_lossless_and_orthogonal(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        ca, cb = 0.7, 0.3
        stack_loras(lora_a, lora_b, dst, coeff_a=ca, coeff_b=cb, mode="gram-schmidt")

        ta, tb, out = (
            load_file(str(lora_a)),
            load_file(str(lora_b)),
            load_file(str(dst)),
        )
        for module in MODULES:
            dw_a = (
                ta[f"{module}.lora_B.weight"] @ ta[f"{module}.lora_A.weight"]
            ).astype(np.float64)
            dw_b = (
                tb[f"{module}.lora_B.weight"] @ tb[f"{module}.lora_A.weight"]
            ).astype(np.float64)
            c = np.sum(dw_a * dw_b) / np.sum(dw_a * dw_a)
            dw_b_ortho = dw_b - c * dw_a
            expected = ca * dw_a + cb * dw_b_ortho
            actual = out[f"{module}.lora_B.weight"] @ out[f"{module}.lora_A.weight"]
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            # the removed component is orthogonal to the reference delta
            overlap = abs(np.sum(dw_a * dw_b_ortho))
            assert overlap <= 1e-9 * np.linalg.norm(dw_a) * np.linalg.norm(dw_b)

    def test_mode_noop_on_single_side(self, tmp_path: Path) -> None:
        """Modules present in only one file are untouched by any mode."""
        rng = np.random.RandomState(0)
        only_a = (
            rng.randn(2, 4).astype(np.float32),
            rng.randn(4, 2).astype(np.float32),
        )
        a = _make_lora(tmp_path / "a.safetensors", {"base_model.model.q": only_a})
        b = _make_lora(
            tmp_path / "b.safetensors",
            {"base_model.model.other": only_a},
        )
        dst = tmp_path / "out.safetensors"
        stack_loras(a, b, dst, coeff_a=0.5, coeff_b=2.0, mode="gram-schmidt")
        out = load_file(str(dst))
        np.testing.assert_allclose(
            out["base_model.model.q.lora_A.weight"], 0.5 * only_a[0], atol=1e-6
        )
        np.testing.assert_allclose(
            out["base_model.model.other.lora_A.weight"], 2.0 * only_a[0], atol=1e-6
        )

    def test_metadata_records_mode(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        stack_loras(lora_a, lora_b, dst, mode="norm-scaler")
        info = detect_lora(dst)
        assert info is not None
        assert info.metadata.get("conflict_mode") == "norm-scaler"

    def test_invalid_mode_raises(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            stack_loras(lora_a, lora_b, tmp_path / "out.safetensors", mode="bogus")


class TestErrors:
    def test_rejects_non_lora(
        self, mini_model: Path, lora_a: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        with pytest.raises(ValueError, match="Not a PEFT LoRA"):
            stack_loras(mini_model, lora_a, dst)
        with pytest.raises(ValueError, match="Not a PEFT LoRA"):
            stack_loras(lora_a, mini_model, dst)


class TestCli:
    def test_cli_basic(self, lora_a: Path, lora_b: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app, ["lora", "stack", str(lora_a), str(lora_b), "-o", str(dst)]
        )
        assert result.exit_code == 0, result.output
        assert "Modules in both:" in result.output
        assert dst.exists()

    def test_cli_with_coeffs_and_target_rank(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app,
            [
                "lora",
                "stack",
                str(lora_a),
                str(lora_b),
                "-a",
                "0.7",
                "-b",
                "0.3",
                "--target-rank",
                "4",
                "-o",
                str(dst),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Truncated to rank:    4" in result.output

    def test_cli_dry_run(self, lora_a: Path, lora_b: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app,
            ["lora", "stack", str(lora_a), str(lora_b), "-o", str(dst), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "dry run" in result.output.lower()
        assert not dst.exists()

    def test_cli_smart_output(self, lora_a: Path, lora_b: Path) -> None:
        """When -o is omitted, output goes next to file A as <stem>.stack.safetensors."""
        result = runner.invoke(app, ["lora", "stack", str(lora_a), str(lora_b)])
        assert result.exit_code == 0, result.output
        expected = lora_a.parent / "a.stack.safetensors"
        assert expected.exists()

    def test_cli_gram_schmidt_mode(
        self, lora_a: Path, lora_b: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app,
            [
                "lora",
                "stack",
                str(lora_a),
                str(lora_b),
                "--mode",
                "gram-schmidt",
                "-o",
                str(dst),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Mode:                 gram-schmidt" in result.output
        assert dst.exists()

    def test_cli_invalid_mode(self, lora_a: Path, lora_b: Path) -> None:
        result = runner.invoke(
            app, ["lora", "stack", str(lora_a), str(lora_b), "--mode", "bogus"]
        )
        assert result.exit_code == 2
