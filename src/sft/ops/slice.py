"""Pure logic for slice and strip — filtering tensors by glob pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sft.index import TensorIndex
from sft.utils.glob import filter_tensors
from sft.utils.output import resolve_output
from sft.utils.tensor_io import copy_with_transform, iter_tensor_names


@dataclass
class SliceResult:
    """Result of a slice or strip operation."""

    included: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    output_path: Path | None = None


def slice_file(
    src: Path,
    dst: Path | None,
    include: str | None,
    exclude: str | None,
    dry_run: bool = False,
) -> SliceResult:
    """Keep tensors matching include, remove those matching exclude.

    Preserves metadata from the source file.
    """
    all_names = iter_tensor_names(src)
    kept = filter_tensors(all_names, include=include, exclude=exclude)
    kept_set = set(kept)
    removed = [n for n in all_names if n not in kept_set]

    output_path = resolve_output(dst, src, "sliced")

    if dry_run:
        return SliceResult(included=kept, excluded=removed, output_path=None)

    index = TensorIndex.from_file(src)
    copy_with_transform(
        src,
        output_path,
        transform=lambda _name, tensor: tensor,
        include_names=kept,
        metadata=index.metadata or None,
    )

    return SliceResult(included=kept, excluded=removed, output_path=output_path)


def strip_file(
    src: Path,
    dst: Path | None,
    exclude: str,
    dry_run: bool = False,
) -> SliceResult:
    """Remove tensors matching the exclude pattern, keep everything else.

    This is the inverse of slice: instead of selecting what to keep,
    you specify what to remove.
    """
    output_path = resolve_output(dst, src, "stripped")

    all_names = iter_tensor_names(src)
    kept = filter_tensors(all_names, include=None, exclude=exclude)
    kept_set = set(kept)
    removed = [n for n in all_names if n not in kept_set]

    if dry_run:
        return SliceResult(included=kept, excluded=removed, output_path=None)

    index = TensorIndex.from_file(src)
    copy_with_transform(
        src,
        output_path,
        transform=lambda _name, tensor: tensor,
        include_names=kept,
        metadata=index.metadata or None,
    )

    return SliceResult(included=kept, excluded=removed, output_path=output_path)
