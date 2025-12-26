"""Data model and parsing for safetensors files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def natural_sort_key(s: str) -> list:
    """Generate a sort key for natural (human) sorting.

    Splits string into text and numeric parts so that numbers are
    compared numerically: 'layer.2' < 'layer.10' instead of 'layer.10' < 'layer.2'.
    """
    parts = re.split(r"(\d+)", s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


@dataclass
class TensorInfo:
    """Information about a single tensor."""

    full_name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    @property
    def rank(self) -> int:
        """Return the number of dimensions (rank) of the tensor."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Return the number of elements in the tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class TensorIndex:
    """Index of all tensors in a safetensors file."""

    tensors: list[TensorInfo]
    metadata: dict[str, Any]
    file_path: Path

    @classmethod
    def from_file(cls, path: Path) -> TensorIndex:
        """Parse a safetensors file and extract tensor metadata (header only).

        This uses direct header parsing to avoid loading any tensor data.
        """
        import json
        import struct

        tensors: list[TensorInfo] = []
        metadata: dict[str, Any] = {}

        with open(path, "rb") as f:
            # Read header size (first 8 bytes, little-endian u64)
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                raise ValueError("Invalid safetensors file: too short")

            header_size = struct.unpack("<Q", header_size_bytes)[0]

            # Read and parse header JSON
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))

        # Extract metadata if present
        if "__metadata__" in header:
            metadata = header.pop("__metadata__")

        # Extract tensor info
        for name, info in header.items():
            dtype_str = info["dtype"]
            shape = tuple(info["shape"])

            # Calculate byte offsets
            data_offsets = info["data_offsets"]
            nbytes = data_offsets[1] - data_offsets[0]

            tensors.append(
                TensorInfo(
                    full_name=name,
                    shape=shape,
                    dtype=dtype_str,
                    nbytes=nbytes,
                )
            )

        # Sort tensors by name for consistent ordering (natural sort)
        tensors.sort(key=lambda t: natural_sort_key(t.full_name))

        return cls(tensors=tensors, metadata=metadata, file_path=path)

    @property
    def total_tensors(self) -> int:
        """Return the total number of tensors."""
        return len(self.tensors)

    @property
    def total_bytes(self) -> int:
        """Return the total size of all tensors in bytes."""
        return sum(t.nbytes for t in self.tensors)


@dataclass
class PrefixTreeNode:
    """A node in the prefix tree representing a namespace."""

    name: str
    children: dict[str, PrefixTreeNode] = field(default_factory=dict)
    tensor_ids: list[int] = field(default_factory=list)
    aggregate_count: int = 0
    aggregate_bytes: int = 0

    def add_tensor(self, parts: list[str], tensor_id: int, nbytes: int) -> None:
        """Add a tensor to this node or a descendant."""
        if not parts:
            # This tensor belongs directly to this node
            self.tensor_ids.append(tensor_id)
            return

        # Navigate/create child node
        child_name = parts[0]
        if child_name not in self.children:
            self.children[child_name] = PrefixTreeNode(name=child_name)

        self.children[child_name].add_tensor(parts[1:], tensor_id, nbytes)

    def compute_aggregates(self, tensors: list[TensorInfo]) -> None:
        """Compute aggregate counts and bytes for this node and descendants."""
        # First, compute for all children
        for child in self.children.values():
            child.compute_aggregates(tensors)

        # Aggregate from direct tensors
        direct_count = len(self.tensor_ids)
        direct_bytes = sum(tensors[tid].nbytes for tid in self.tensor_ids)

        # Aggregate from children
        child_count = sum(c.aggregate_count for c in self.children.values())
        child_bytes = sum(c.aggregate_bytes for c in self.children.values())

        self.aggregate_count = direct_count + child_count
        self.aggregate_bytes = direct_bytes + child_bytes


class PrefixTree:
    """A tree structure built from tensor names using a delimiter."""

    def __init__(self, index: TensorIndex, delimiter: str = ".") -> None:
        """Build a prefix tree from a tensor index."""
        self.index = index
        self.delimiter = delimiter
        self.root = PrefixTreeNode(name="")

        # Build the tree
        for i, tensor in enumerate(index.tensors):
            parts = tensor.full_name.split(delimiter)
            self.root.add_tensor(parts, i, tensor.nbytes)

        # Compute aggregates
        self.root.compute_aggregates(index.tensors)

    def get_tensors_under(self, prefix: str) -> list[TensorInfo]:
        """Get all tensors under a given prefix."""
        if not prefix:
            return self.index.tensors

        # Navigate to the prefix node
        parts = prefix.split(self.delimiter)
        node = self.root

        for part in parts:
            if part in node.children:
                node = node.children[part]
            else:
                return []

        # Collect all tensor IDs under this node
        tensor_ids = self._collect_tensor_ids(node)
        return [self.index.tensors[tid] for tid in tensor_ids]

    def _collect_tensor_ids(self, node: PrefixTreeNode) -> list[int]:
        """Recursively collect all tensor IDs under a node."""
        ids = list(node.tensor_ids)
        for child in node.children.values():
            ids.extend(self._collect_tensor_ids(child))
        return ids
