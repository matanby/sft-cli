"""Pure logic for the cat command — merge multiple safetensors files into one."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class CatResult:
    """Result of merging safetensors files."""

    total_tensors: int
    total_files: int
    duplicates: list[str] = field(default_factory=list)
    output_path: Path | None = None


def cat_files(
    sources: list[Path],
    dst: Path | None,
    *,
    allow_duplicates: bool = False,
    dry_run: bool = False,
) -> CatResult:
    """Merge multiple safetensors files into one.

    Raises ValueError if duplicate tensor names are found and allow_duplicates is False.
    """
    merged_tensors: dict[str, np.ndarray] = {}
    merged_metadata: dict[str, str] = {}
    seen_names: dict[str, str] = {}  # tensor_name -> source filename
    duplicates: list[str] = []

    for path in sources:
        index = TensorIndex.from_file(path)
        if index.metadata:
            merged_metadata.update(index.metadata)

        tensors = read_tensors(path)
        for name, array in tensors.items():
            if name in seen_names:
                if name not in duplicates:
                    duplicates.append(name)
                if not allow_duplicates:
                    continue
            seen_names[name] = path.name
            merged_tensors[name] = array

    if duplicates and not allow_duplicates:
        raise ValueError(
            f"Duplicate tensor names across files: {', '.join(duplicates)}"
        )

    output_path = dst if dst is not None else Path("merged.safetensors")

    if not dry_run:
        write_file(output_path, merged_tensors, metadata=merged_metadata or None)

    return CatResult(
        total_tensors=len(merged_tensors),
        total_files=len(sources),
        duplicates=duplicates,
        output_path=output_path if not dry_run else None,
    )
