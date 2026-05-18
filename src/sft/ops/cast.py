"""Pure logic for the cast command — dtype conversion of safetensors tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.utils.dtypes import SAFETENSORS_TO_NUMPY, cast_tensor, resolve_dtype
from sft.utils.glob import filter_tensors
from sft.utils.tensor_io import copy_with_transform


@dataclass
class CastResult:
    """Result of a cast operation."""

    cast_count: int
    skipped_count: int
    output_path: Path


def cast_file(
    src: Path,
    dst: Path,
    target_dtype: str,
    include: str | None = None,
    exclude: str | None = None,
    dry_run: bool = False,
) -> CastResult:
    """Cast tensors in a safetensors file to a target dtype.

    Tensors matching include/exclude filters are cast; others pass through unchanged.
    """
    target_np = resolve_dtype(target_dtype)

    index = TensorIndex.from_file(src)
    all_names = [t.full_name for t in index.tensors]
    names_to_cast = set(filter_tensors(all_names, include, exclude))

    dtype_by_name = {
        t.full_name: SAFETENSORS_TO_NUMPY.get(t.dtype) for t in index.tensors
    }

    cast_count = 0
    skipped_count = 0
    for name in all_names:
        if name in names_to_cast and dtype_by_name.get(name) != target_np:
            cast_count += 1
        else:
            skipped_count += 1

    if not dry_run:

        def transform(name: str, tensor: np.ndarray) -> np.ndarray:
            if name in names_to_cast:
                return cast_tensor(tensor, target_np)
            return tensor

        copy_with_transform(src, dst, transform)

    return CastResult(
        cast_count=cast_count,
        skipped_count=skipped_count,
        output_path=dst,
    )
