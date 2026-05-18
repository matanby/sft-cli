"""Performance and scale smoke tests.

Marked ``slow`` so they don't slow down the default test loop. Thresholds
are intentionally generous: we're guarding against catastrophic regressions
(loading tensor data when only the header should be read; falling off the
QR-accelerated SVD fast path), not benchmarking.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from sft.index import TensorIndex

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def big_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A safetensors file with 10k zero-sized tensors — large header, tiny data."""
    p = tmp_path_factory.mktemp("big") / "big.safetensors"
    tensors = {
        f"model.layers.{i // 8}.module_{i % 8}.weight": np.zeros((1,), dtype=np.float32)
        for i in range(10_000)
    }
    save_file(tensors, str(p))
    return p


def test_header_only_index_is_fast(big_file: Path) -> None:
    """Parsing 10k tensors from header should comfortably fit in 500 ms.

    Slow here means we regressed into reading the data section.
    """
    t0 = time.perf_counter()
    index = TensorIndex.from_file(big_file)
    dt = time.perf_counter() - t0
    assert len(index.tensors) == 10_000
    assert dt < 0.5, f"header parse took {dt:.3f}s (>0.5s); header-only regression?"


def test_qr_svd_uses_thin_factor_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_qr_svd` on a thin (rank=8, dim=4096) input must NOT form the full
    (4096, 4096) product matrix before running SVD.
    """
    from sft.ops.lora import svd as svd_mod

    real_svd = np.linalg.svd
    seen_shapes: list[tuple[int, ...]] = []

    def spy(matrix, *args, **kwargs):
        seen_shapes.append(matrix.shape)
        return real_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", spy)

    rng = np.random.RandomState(0)
    rank = 8
    a = rng.randn(rank, 4096).astype(np.float32)
    b = rng.randn(4096, rank).astype(np.float32)

    (s,) = svd_mod._qr_svd(a, b, compute_uv=False)
    assert s.shape == (rank,)
    assert any(shape == (rank, rank) for shape in seen_shapes), (
        f"QR-SVD didn't run on a (rank, rank) matrix; saw shapes {seen_shapes}"
    )
    assert not any(shape == (4096, 4096) for shape in seen_shapes), (
        "QR-SVD fell back to full (4096, 4096) SVD"
    )
