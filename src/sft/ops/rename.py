"""Pure logic for the rename command — regex-based tensor key renaming."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sft.index import TensorIndex
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class RenameMapping:
    old_name: str
    new_name: str


@dataclass
class RenameResult:
    mappings: list[RenameMapping] = field(default_factory=list)
    unchanged: int = 0
    output_path: Path | None = None


def rename_tensors(
    src: Path,
    dst: Path | None,
    substitutions: list[tuple[str, str]],
    *,
    dry_run: bool = False,
) -> RenameResult:
    """Rename tensor keys by applying regex substitutions in order.

    Each substitution is a (pattern, replacement) pair applied via re.sub().
    """
    index = TensorIndex.from_file(src)
    tensor_names = [t.full_name for t in index.tensors]

    mappings: list[RenameMapping] = []
    unchanged = 0

    for name in tensor_names:
        new_name = name
        for pattern, replacement in substitutions:
            new_name = re.sub(pattern, replacement, new_name)
        if new_name != name:
            mappings.append(RenameMapping(old_name=name, new_name=new_name))
        else:
            unchanged += 1

    if dry_run:
        return RenameResult(mappings=mappings, unchanged=unchanged)

    tensors = read_tensors(src)
    new_tensors: dict = {}
    for name, tensor in tensors.items():
        new_name = name
        for pattern, replacement in substitutions:
            new_name = re.sub(pattern, replacement, new_name)
        new_tensors[new_name] = tensor

    write_file(dst, new_tensors, metadata=index.metadata or None)

    return RenameResult(mappings=mappings, unchanged=unchanged, output_path=dst)
