"""Pure logic for the metadata command — view/edit safetensors file metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sft.index import TensorIndex
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class MetadataResult:
    metadata: dict[str, str]
    output_path: Path | None


def get_metadata(path: Path) -> dict[str, str]:
    """Read metadata from a safetensors file (header-only, no tensor data loaded)."""
    index = TensorIndex.from_file(path)
    return dict(index.metadata)


def set_metadata(
    src: Path,
    dst: Path,
    set_keys: dict[str, str] | None = None,
    unset_keys: list[str] | None = None,
) -> MetadataResult:
    """Apply set/unset changes to metadata and write a new file with all tensors preserved."""
    index = TensorIndex.from_file(src)
    metadata = dict(index.metadata)

    if set_keys:
        metadata.update(set_keys)

    if unset_keys:
        for key in unset_keys:
            metadata.pop(key, None)

    tensors = read_tensors(src)
    write_file(dst, tensors, metadata=metadata)

    return MetadataResult(metadata=metadata, output_path=dst)
