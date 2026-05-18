"""Pure logic for computing per-tensor statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.utils.formatting import format_dtype
from sft.utils.glob import filter_tensors
from sft.utils.tensor_io import read_tensors


@dataclass
class TensorStats:
    """Statistics for a single tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float
    sparsity: float
    nan_count: int
    inf_count: int


def compute_stats(
    path: Path,
    include: str | None = None,
    exclude: str | None = None,
) -> list[TensorStats]:
    """Compute per-tensor statistics for a safetensors file."""
    index = TensorIndex.from_file(path)
    all_names = [t.full_name for t in index.tensors]
    names = filter_tensors(all_names, include=include, exclude=exclude)

    info_by_name = {t.full_name: t for t in index.tensors}
    tensors = read_tensors(path)

    results: list[TensorStats] = []
    for name in names:
        arr = tensors[name]
        info = info_by_name[name]
        dtype_label = format_dtype(info.dtype)

        arr_float = arr.astype(np.float64)
        nan_count = int(np.isnan(arr_float).sum())
        inf_count = int(np.isinf(arr_float).sum())

        finite = arr_float[np.isfinite(arr_float)]
        if finite.size > 0:
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
        else:
            mean = std = min_val = max_val = float("nan")

        numel = arr.size
        sparsity = float(np.sum(arr == 0) / numel) if numel > 0 else 0.0

        results.append(
            TensorStats(
                name=name,
                dtype=dtype_label,
                shape=tuple(arr.shape),
                mean=mean,
                std=std,
                min=min_val,
                max=max_val,
                sparsity=sparsity,
                nan_count=nan_count,
                inf_count=inf_count,
            )
        )

    return results
