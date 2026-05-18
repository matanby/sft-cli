"""Tests for sft lora svd command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.lora.svd import _qr_svd, analyze_svd

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


# --- QR-accelerated SVD unit tests ---


class TestQrSvd:
    """Tests for the _qr_svd helper that accelerates SVD via QR factorization."""

    def test_matches_naive_svd(self):
        """QR-accelerated SVD produces the same singular values as naive B@A SVD."""
        rng = np.random.RandomState(42)
        rank, in_feat, out_feat = 8, 512, 256
        a = rng.randn(rank, in_feat).astype(np.float32)
        b = rng.randn(out_feat, rank).astype(np.float32)

        (s_qr,) = _qr_svd(a, b, compute_uv=False)

        delta = b @ a
        s_naive = np.linalg.svd(delta, compute_uv=False)[:rank]

        np.testing.assert_allclose(s_qr, s_naive, rtol=1e-5)

    def test_uv_reconstruction(self):
        """U @ diag(s) @ Vt from _qr_svd reconstructs B@A accurately."""
        rng = np.random.RandomState(123)
        rank, in_feat, out_feat = 4, 64, 32
        a = rng.randn(rank, in_feat).astype(np.float64)
        b = rng.randn(out_feat, rank).astype(np.float64)

        u, s, vt = _qr_svd(a, b, compute_uv=True)
        reconstructed = u @ np.diag(s) @ vt
        expected = b @ a

        np.testing.assert_allclose(reconstructed, expected, atol=1e-10)

    def test_truncated_reconstruction(self):
        """Truncating to fewer singular values gives the best rank-k approximation."""
        rng = np.random.RandomState(7)
        rank, in_feat, out_feat = 16, 128, 64
        a = rng.randn(rank, in_feat).astype(np.float64)
        b = rng.randn(out_feat, rank).astype(np.float64)

        u, s, vt = _qr_svd(a, b, compute_uv=True)
        k = 4
        approx = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]
        full = b @ a

        error = np.linalg.norm(full - approx, "fro") / np.linalg.norm(full, "fro")
        expected_error = np.sqrt(1.0 - (s[:k] ** 2).sum() / (s**2).sum())
        np.testing.assert_allclose(error, expected_error, rtol=1e-10)

    def test_rank_1(self):
        """Works correctly with rank-1 LoRA pairs."""
        rng = np.random.RandomState(0)
        a = rng.randn(1, 64).astype(np.float64)
        b = rng.randn(64, 1).astype(np.float64)

        (s_qr,) = _qr_svd(a, b, compute_uv=False)
        assert s_qr.shape == (1,)

        expected = np.linalg.norm(b @ a, "fro")
        np.testing.assert_allclose(s_qr[0], expected, rtol=1e-10)

    def test_zero_matrices(self):
        """Handles all-zero A or B gracefully."""
        a = np.zeros((4, 32), dtype=np.float64)
        b = np.zeros((32, 4), dtype=np.float64)

        (s,) = _qr_svd(a, b, compute_uv=False)
        assert np.all(s == 0) or s.sum() < 1e-15

    def test_singular_values_descending(self):
        """Singular values are returned in descending order."""
        rng = np.random.RandomState(99)
        a = rng.randn(8, 128).astype(np.float64)
        b = rng.randn(64, 8).astype(np.float64)

        (s,) = _qr_svd(a, b, compute_uv=False)
        assert np.all(s[:-1] >= s[1:])

    def test_different_dtypes(self):
        """Works with float16 and float32 inputs (upcasts internally)."""
        rng = np.random.RandomState(5)
        a = rng.randn(4, 64).astype(np.float16)
        b = rng.randn(64, 4).astype(np.float16)

        (s,) = _qr_svd(a, b, compute_uv=False)
        assert s.dtype == np.float64
        assert len(s) == 4
        assert np.all(s >= 0)
