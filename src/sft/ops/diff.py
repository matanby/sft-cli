"""Pure logic for comparing two safetensors files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.utils.glob import filter_tensors
from sft.utils.tensor_io import read_tensors

# Default tolerances mirror numpy.allclose defaults; used to classify
# value diffs as "close" vs "differ".
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8


@dataclass
class ValueDiff:
    """Numerical difference between two tensors with the same name/shape/dtype.

    `status` is one of "equal" (max_abs == 0), "close" (within rtol/atol per
    numpy.allclose), or "differ" (outside tolerances).
    """

    status: str  # "equal" | "close" | "differ"
    max_abs: float  # max |a - b|
    mean_abs: float  # mean |a - b|
    l2_norm: float  # ||a - b||  (Frobenius)
    rel_l2: float  # ||a - b|| / ||a||  (0 if ||a|| == 0)
    cosine_sim: float


@dataclass
class TensorDiff:
    """Result of comparing two safetensors files.

    Structural buckets (always populated): `added`, `removed`, `shape_changed`,
    `dtype_changed`, `unchanged`. When `compute_delta=True`, `value_diffs`
    maps each comparable tensor (in `unchanged`) to per-tensor metrics with a
    status classification (`equal`/`close`/`differ`).
    """

    added: list[str]
    removed: list[str]
    shape_changed: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    dtype_changed: dict[str, tuple[str, str]]
    unchanged: list[str]
    value_diffs: dict[str, ValueDiff] | None = field(default=None)
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    def by_status(self, status: str) -> list[str]:
        """Names of comparable tensors with the given value-diff status."""
        if self.value_diffs is None:
            return []
        return [n for n, vd in self.value_diffs.items() if vd.status == status]


def _value_diff(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> ValueDiff:
    """Compute all numeric diff metrics + tolerance-based status for one pair."""
    a64 = a.astype(np.float64).flatten()
    b64 = b.astype(np.float64).flatten()
    delta = a64 - b64

    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    mean_abs = float(np.mean(np.abs(delta))) if delta.size else 0.0
    l2 = float(np.linalg.norm(delta))
    norm_a = float(np.linalg.norm(a64))
    rel_l2 = l2 / norm_a if norm_a > 0 else 0.0

    norm_b = float(np.linalg.norm(b64))
    if norm_a == 0 or norm_b == 0:
        cosine = 0.0
    else:
        cosine = float(np.dot(a64, b64) / (norm_a * norm_b))

    if max_abs == 0:
        status = "equal"
    elif np.allclose(a64, b64, rtol=rtol, atol=atol):
        status = "close"
    else:
        status = "differ"

    return ValueDiff(
        status=status,
        max_abs=max_abs,
        mean_abs=mean_abs,
        l2_norm=l2,
        rel_l2=rel_l2,
        cosine_sim=cosine,
    )


def diff_files(
    path_a: Path,
    path_b: Path,
    *,
    compute_delta: bool = False,
    include: str | None = None,
    exclude: str | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> TensorDiff:
    """Compare two safetensors files and return a structured diff.

    When `compute_delta=True`, also computes per-tensor numeric metrics
    (`max_abs`, `mean_abs`, `l2_norm`, `rel_l2`, `cosine_sim`) for every
    tensor that shares name, shape, and dtype, and classifies each as
    `equal` / `close` (per `rtol`/`atol`) / `differ`.
    """
    index_a = TensorIndex.from_file(path_a)
    index_b = TensorIndex.from_file(path_b)

    info_a = {t.full_name: t for t in index_a.tensors}
    info_b = {t.full_name: t for t in index_b.tensors}

    all_names = sorted(set(info_a) | set(info_b))
    all_names = filter_tensors(all_names, include=include, exclude=exclude)

    added: list[str] = []
    removed: list[str] = []
    shape_changed: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    dtype_changed: dict[str, tuple[str, str]] = {}
    unchanged: list[str] = []
    comparable: list[str] = []

    for name in all_names:
        in_a = name in info_a
        in_b = name in info_b

        if in_b and not in_a:
            added.append(name)
        elif in_a and not in_b:
            removed.append(name)
        else:
            ta, tb = info_a[name], info_b[name]
            if ta.shape != tb.shape:
                shape_changed[name] = (ta.shape, tb.shape)
            elif ta.dtype != tb.dtype:
                dtype_changed[name] = (ta.dtype, tb.dtype)
            else:
                unchanged.append(name)
                comparable.append(name)

    value_diffs: dict[str, ValueDiff] | None = None
    if compute_delta and comparable:
        tensors_a = read_tensors(path_a)
        tensors_b = read_tensors(path_b)
        value_diffs = {}
        for name in comparable:
            value_diffs[name] = _value_diff(
                tensors_a[name], tensors_b[name], rtol=rtol, atol=atol
            )

    return TensorDiff(
        added=added,
        removed=removed,
        shape_changed=shape_changed,
        dtype_changed=dtype_changed,
        unchanged=unchanged,
        value_diffs=value_diffs,
        rtol=rtol,
        atol=atol,
    )
