"""Tests for the diff command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.diff import diff_files

runner = CliRunner()


def _make_file(
    path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_diff_identical(mini_model: Path) -> None:
    result = diff_files(mini_model, mini_model)
    assert result.added == []
    assert result.removed == []
    assert result.shape_changed == {}
    assert result.dtype_changed == {}
    assert len(result.unchanged) > 0
    assert result.value_diffs is None


def test_diff_added_tensors(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w1": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {
            "w1": np.zeros((4, 4), dtype=np.float32),
            "w2": np.ones((8,), dtype=np.float16),
        },
    )
    result = diff_files(base, target)
    assert result.added == ["w2"]
    assert result.removed == []
    assert "w1" in result.unchanged


def test_diff_removed_tensors(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {
            "w1": np.zeros((4, 4), dtype=np.float32),
            "w2": np.ones((8,), dtype=np.float16),
        },
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w1": np.zeros((4, 4), dtype=np.float32)},
    )
    result = diff_files(base, target)
    assert result.removed == ["w2"]
    assert result.added == []
    assert "w1" in result.unchanged


def test_diff_shape_changed(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": np.zeros((8, 8), dtype=np.float32)},
    )
    result = diff_files(base, target)
    assert "w" in result.shape_changed
    assert result.shape_changed["w"] == ((4, 4), (8, 8))
    assert result.added == []
    assert result.removed == []


def test_diff_dtype_changed(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": np.zeros((4, 4), dtype=np.float16)},
    )
    result = diff_files(base, target)
    assert "w" in result.dtype_changed
    assert result.dtype_changed["w"][0] == "F32"
    assert result.dtype_changed["w"][1] == "F16"


def test_diff_delta(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    w_base = rng.randn(8, 8).astype(np.float32)
    w_target = w_base + 0.01 * rng.randn(8, 8).astype(np.float32)

    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": w_base, "b": np.ones(4, dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": w_target, "b": np.ones(4, dtype=np.float32)},
    )
    result = diff_files(base, target, compute_delta=True)
    assert result.value_diffs is not None
    assert "w" in result.value_diffs
    assert result.value_diffs["w"].l2_norm > 0
    assert result.value_diffs["w"].cosine_sim > 0.99
    # Identical tensor should have l2_norm == 0 and cosine_sim == 1.0
    assert result.value_diffs["b"].l2_norm == pytest.approx(0.0)
    assert result.value_diffs["b"].cosine_sim == pytest.approx(1.0)


class TestRicherMetrics:
    """Tests for the upgraded diff metrics: max_abs/mean_abs/rel_L2 + status."""

    def test_equal_status(self, tmp_path: Path) -> None:
        """Bitwise-identical tensors get status='equal'."""
        rng = np.random.RandomState(0)
        w = rng.randn(8, 8).astype(np.float32)
        a = _make_file(tmp_path / "a.safetensors", {"w": w})
        b = _make_file(tmp_path / "b.safetensors", {"w": w})
        result = diff_files(a, b, compute_delta=True)
        vd = result.value_diffs["w"]
        assert vd.status == "equal"
        assert vd.max_abs == 0.0
        assert vd.mean_abs == 0.0
        assert vd.rel_l2 == 0.0

    def test_close_status_default_tolerances(self, tmp_path: Path) -> None:
        """Tiny perturbations within default rtol/atol get status='close'."""
        rng = np.random.RandomState(1)
        w_base = rng.randn(8, 8).astype(np.float64)
        # 1e-9 perturbation is well below default rtol=1e-5
        w_target = w_base + 1e-9 * rng.randn(8, 8)
        a = _make_file(tmp_path / "a.safetensors", {"w": w_base})
        b = _make_file(tmp_path / "b.safetensors", {"w": w_target})
        result = diff_files(a, b, compute_delta=True)
        vd = result.value_diffs["w"]
        assert vd.status == "close"
        assert vd.max_abs > 0
        assert vd.max_abs < 1e-8

    def test_differ_status(self, tmp_path: Path) -> None:
        """Substantial differences get status='differ'."""
        rng = np.random.RandomState(2)
        w_base = rng.randn(8, 8).astype(np.float32)
        w_target = w_base + 0.5 * rng.randn(8, 8).astype(np.float32)
        a = _make_file(tmp_path / "a.safetensors", {"w": w_base})
        b = _make_file(tmp_path / "b.safetensors", {"w": w_target})
        result = diff_files(a, b, compute_delta=True)
        vd = result.value_diffs["w"]
        assert vd.status == "differ"
        assert vd.max_abs > 0.1
        assert vd.rel_l2 > 0.0

    def test_rel_l2_zero_when_base_is_zero(self, tmp_path: Path) -> None:
        """rel_L2 should be 0 (not NaN) when ||a|| == 0."""
        a = _make_file(
            tmp_path / "a.safetensors", {"w": np.zeros((4, 4), dtype=np.float32)}
        )
        b = _make_file(
            tmp_path / "b.safetensors", {"w": np.ones((4, 4), dtype=np.float32)}
        )
        result = diff_files(a, b, compute_delta=True)
        vd = result.value_diffs["w"]
        assert vd.rel_l2 == 0.0
        assert vd.l2_norm == pytest.approx(4.0)
        assert vd.max_abs == 1.0
        assert vd.mean_abs == 1.0

    def test_rtol_atol_flags_change_classification(self, tmp_path: Path) -> None:
        """Loosening rtol promotes 'differ' tensors to 'close'."""
        rng = np.random.RandomState(3)
        w_base = rng.randn(8, 8).astype(np.float32)
        w_target = w_base + 1e-3 * rng.randn(8, 8).astype(np.float32)
        a = _make_file(tmp_path / "a.safetensors", {"w": w_base})
        b = _make_file(tmp_path / "b.safetensors", {"w": w_target})

        strict = diff_files(a, b, compute_delta=True)
        loose = diff_files(a, b, compute_delta=True, rtol=1e-1, atol=1e-2)
        assert strict.value_diffs["w"].status == "differ"
        assert loose.value_diffs["w"].status == "close"

    def test_by_status_helper(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(4)
        w = rng.randn(4, 4).astype(np.float32)
        a = _make_file(
            tmp_path / "a.safetensors", {"same": w, "diff": np.zeros_like(w)}
        )
        b = _make_file(tmp_path / "b.safetensors", {"same": w, "diff": np.ones_like(w)})
        result = diff_files(a, b, compute_delta=True)
        assert result.by_status("equal") == ["same"]
        assert result.by_status("differ") == ["diff"]


class TestCliRicherMetrics:
    """CLI-level tests for the new diff metric output."""

    def test_delta_shows_status_buckets(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(5)
        w = rng.randn(4, 4).astype(np.float32)
        a = _make_file(tmp_path / "a.safetensors", {"w": w, "stable": w.copy()})
        b = _make_file(
            tmp_path / "b.safetensors",
            {"w": w + 0.1, "stable": w.copy()},
        )
        result = runner.invoke(app, ["diff", str(a), str(b), "--delta"])
        assert result.exit_code == 0
        out = result.output
        assert "equal" in out
        assert "differ" in out
        assert "max_abs" in out
        assert "rel_L2" in out

    def test_rtol_flag(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(6)
        w = rng.randn(4, 4).astype(np.float32)
        a = _make_file(tmp_path / "a.safetensors", {"w": w})
        b = _make_file(tmp_path / "b.safetensors", {"w": w + 1e-3})
        result = runner.invoke(
            app, ["diff", str(a), str(b), "--delta", "--rtol", "1e-1"]
        )
        assert result.exit_code == 0
        # With loose rtol, the small perturbation classifies as close
        assert "close" in result.output

    def test_json_includes_new_fields(self, tmp_path: Path) -> None:
        a = _make_file(
            tmp_path / "a.safetensors",
            {"w": np.zeros((4, 4), dtype=np.float32)},
        )
        b = _make_file(
            tmp_path / "b.safetensors",
            {"w": np.ones((4, 4), dtype=np.float32)},
        )
        result = runner.invoke(app, ["diff", str(a), str(b), "--delta", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "rtol" in data
        assert "atol" in data
        assert "value_diffs" in data
        vd = data["value_diffs"]["w"]
        for key in ("status", "max_abs", "mean_abs", "l2_norm", "rel_l2", "cosine_sim"):
            assert key in vd


def test_diff_json(mini_model: Path) -> None:
    result = runner.invoke(app, ["diff", str(mini_model), str(mini_model), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, dict)
    assert data["added"] == []
    assert data["removed"] == []
    assert len(data["unchanged"]) > 0


def test_diff_cli_structural(tmp_path: Path) -> None:
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w1": np.zeros((4, 4), dtype=np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {
            "w1": np.zeros((4, 4), dtype=np.float32),
            "extra": np.ones((2,), dtype=np.float16),
        },
    )
    result = runner.invoke(app, ["diff", str(base), str(target)])
    assert result.exit_code == 0
    assert "Added (1):" in result.output
    assert "+ extra" in result.output
    assert "Unchanged: 1 tensors" in result.output


def test_diff_cli_delta(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    base = _make_file(
        tmp_path / "base.safetensors",
        {"w": rng.randn(4, 4).astype(np.float32)},
    )
    target = _make_file(
        tmp_path / "target.safetensors",
        {"w": rng.randn(4, 4).astype(np.float32)},
    )
    result = runner.invoke(app, ["diff", str(base), str(target), "--delta"])
    assert result.exit_code == 0
    # Output now contains the richer metric headers and a status column
    assert "max_abs" in result.output
    assert "cosine" in result.output
    assert "rtol" in result.output
