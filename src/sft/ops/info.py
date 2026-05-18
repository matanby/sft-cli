"""Pure logic for the info command — summarise a .safetensors file."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sft.index import TensorIndex


@dataclass
class DtypeStats:
    """Aggregated statistics for a single dtype."""

    dtype: str
    count: int
    total_bytes: int
    total_params: int


@dataclass
class FileSummary:
    """Complete summary of a .safetensors file."""

    file_name: str
    file_size: int
    total_tensors: int
    total_parameters: int
    total_tensor_bytes: int
    dtypes: list[DtypeStats] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def summarize(path: Path) -> FileSummary:
    """Build a FileSummary from a .safetensors file (header-only parse)."""
    index = TensorIndex.from_file(path)

    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    dtype_params: Counter[str] = Counter()
    total_params = 0

    for t in index.tensors:
        dtype_counts[t.dtype] += 1
        dtype_bytes[t.dtype] += t.nbytes
        dtype_params[t.dtype] += t.numel
        total_params += t.numel

    dtypes = sorted(
        (
            DtypeStats(
                dtype=dt,
                count=dtype_counts[dt],
                total_bytes=dtype_bytes[dt],
                total_params=dtype_params[dt],
            )
            for dt in dtype_counts
        ),
        key=lambda s: s.total_bytes,
        reverse=True,
    )

    return FileSummary(
        file_name=path.name,
        file_size=path.stat().st_size,
        total_tensors=index.total_tensors,
        total_parameters=total_params,
        total_tensor_bytes=index.total_bytes,
        dtypes=dtypes,
        metadata=dict(index.metadata),
    )
