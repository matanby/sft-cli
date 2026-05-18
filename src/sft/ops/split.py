"""Pure logic for the split command — shard a safetensors file by size threshold."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from sft.index import TensorIndex
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class ShardInfo:
    """Describes one output shard."""

    path: Path
    tensor_names: list[str]
    total_bytes: int


@dataclass
class SplitResult:
    """Result of a split operation."""

    shards: list[ShardInfo]
    index_path: Path | None


_SIZE_UNITS: dict[str, int] = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
}


def parse_size(size_str: str) -> int:
    """Parse a human-readable size string like '4GB', '500MB', '1024B' into bytes."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", size_str.strip())
    if not m:
        raise ValueError(
            f"Invalid size format: {size_str!r}. Expected e.g. '4GB', '500MB', '1024B'."
        )
    value, unit = float(m.group(1)), m.group(2).upper()
    if unit not in _SIZE_UNITS:
        raise ValueError(
            f"Unknown size unit: {unit!r}. Supported: {', '.join(_SIZE_UNITS)}."
        )
    return int(value * _SIZE_UNITS[unit])


def _build_output_pattern(src: Path, output_pattern: str | None) -> str:
    """Resolve the output filename pattern, filling in {stem}."""
    if output_pattern is None:
        return f"{src.stem}-{{index}}.safetensors"
    return output_pattern


def _shard_path(src: Path, pattern: str, shard_idx: int, total_shards: int) -> Path:
    width = len(str(total_shards))
    index_str = str(shard_idx + 1).zfill(width)
    filename = pattern.replace("{index}", index_str)
    return src.parent / filename


def split_file(
    src: Path,
    max_bytes: int,
    output_pattern: str | None = None,
    dry_run: bool = False,
) -> SplitResult:
    """Split a safetensors file into shards not exceeding *max_bytes* each.

    Greedy bin-packing: tensors are added to the current shard until adding
    the next one would exceed the limit.  A tensor larger than *max_bytes*
    gets its own shard.
    """
    index = TensorIndex.from_file(src)
    pattern = _build_output_pattern(src, output_pattern)

    if "{index}" not in pattern:
        raise ValueError("Output pattern must contain '{index}'.")

    bins: list[list[tuple[str, int]]] = []
    current_bin: list[tuple[str, int]] = []
    current_size = 0

    for ti in index.tensors:
        if current_bin and current_size + ti.nbytes > max_bytes:
            bins.append(current_bin)
            current_bin = []
            current_size = 0
        current_bin.append((ti.full_name, ti.nbytes))
        current_size += ti.nbytes

    if current_bin:
        bins.append(current_bin)

    shards: list[ShardInfo] = []
    for i, bin_items in enumerate(bins):
        names = [name for name, _ in bin_items]
        total = sum(nb for _, nb in bin_items)
        path = _shard_path(src, pattern, i, len(bins))
        shards.append(ShardInfo(path=path, tensor_names=names, total_bytes=total))

    if dry_run:
        return SplitResult(shards=shards, index_path=None)

    all_tensors = read_tensors(src)

    for shard in shards:
        shard_tensors = {name: all_tensors[name] for name in shard.tensor_names}
        write_file(shard.path, shard_tensors)

    index_data = {
        "metadata": {"total_size": index.total_bytes},
        "weight_map": {},
    }
    for shard in shards:
        for name in shard.tensor_names:
            index_data["weight_map"][name] = shard.path.name

    index_path = src.parent / f"{src.name}.index.json"
    index_path.write_text(json.dumps(index_data, indent=2) + "\n")

    return SplitResult(shards=shards, index_path=index_path)
