"""Pure logic for the ls command — tabular summary of multiple .safetensors files."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from sft.index import TensorIndex


@dataclass
class FileSummaryRow:
    """One row in the ls output table."""

    file_name: str
    total_tensors: int
    total_params: int
    total_bytes: int
    dtypes: set[str] = field(default_factory=set)


def list_files(paths: list[Path]) -> list[FileSummaryRow]:
    """Build a summary row for each safetensors file.

    Files that fail to parse are skipped with a warning on stderr.
    """
    rows: list[FileSummaryRow] = []
    for path in paths:
        try:
            index = TensorIndex.from_file(path)
        except Exception as exc:
            print(f"Warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue

        total_params = sum(t.numel for t in index.tensors)
        dtypes = {t.dtype for t in index.tensors}

        rows.append(
            FileSummaryRow(
                file_name=path.name,
                total_tensors=index.total_tensors,
                total_params=total_params,
                total_bytes=index.total_bytes,
                dtypes=dtypes,
            )
        )
    return rows
