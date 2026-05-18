"""Pure logic for comparing two safetensors files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.utils.glob import filter_tensors
from sft.utils.tensor_io import read_tensors


@dataclass
class ValueDiff:
    """Numerical difference between two tensors with the same name/shape/dtype."""

    l2_norm: float
    cosine_sim: float


@dataclass
class TensorDiff:
    """Result of comparing two safetensors files."""

    added: list[str]
    removed: list[str]
    shape_changed: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    dtype_changed: dict[str, tuple[str, str]]
    unchanged: list[str]
    value_diffs: dict[str, ValueDiff] | None = field(default=None)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _l2_norm_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm(a.flatten().astype(np.float64) - b.flatten().astype(np.float64))
    )


def diff_files(
    path_a: Path,
    path_b: Path,
    *,
    compute_delta: bool = False,
    include: str | None = None,
    exclude: str | None = None,
) -> TensorDiff:
    """Compare two safetensors files and return a structured diff."""
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
            l2 = _l2_norm_delta(tensors_a[name], tensors_b[name])
            cos = _cosine_similarity(tensors_a[name], tensors_b[name])
            value_diffs[name] = ValueDiff(l2_norm=l2, cosine_sim=cos)

    return TensorDiff(
        added=added,
        removed=removed,
        shape_changed=shape_changed,
        dtype_changed=dtype_changed,
        unchanged=unchanged,
        value_diffs=value_diffs,
    )
