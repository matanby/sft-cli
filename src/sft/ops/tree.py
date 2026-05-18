"""Pure logic for rendering a PrefixTree as an ASCII tree string."""

from __future__ import annotations

from pathlib import Path

from sft.index import PrefixTree, PrefixTreeNode, TensorIndex, natural_sort_key
from sft.utils.formatting import format_bytes, format_dtype, format_shape


def _render_node(
    node: PrefixTreeNode,
    index: TensorIndex,
    prefix: str,
    is_last: bool,
    is_root: bool,
    depth: int,
    max_depth: int | None,
) -> list[str]:
    """Recursively render a node and its children as ASCII tree lines."""
    lines: list[str] = []
    tensors = index.tensors

    if is_root:
        connector = ""
        child_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

    # Build the label for this node
    has_children = bool(node.children)
    has_direct_tensors = bool(node.tensor_ids)
    is_leaf_tensor = not has_children and len(node.tensor_ids) == 1

    if is_leaf_tensor:
        t = tensors[node.tensor_ids[0]]
        label = (
            f"{node.name} "
            f"{format_shape(t.shape)} "
            f"{format_dtype(t.dtype)} "
            f"({format_bytes(t.nbytes)})"
        )
    elif has_children or has_direct_tensors:
        if node.aggregate_count > 1:
            noun = "tensor" if node.aggregate_count == 1 else "tensors"
            label = (
                f"{node.name} "
                f"({node.aggregate_count} {noun}, "
                f"{format_bytes(node.aggregate_bytes)})"
            )
        else:
            label = node.name
    else:
        label = node.name

    if not is_root or node.name:
        lines.append(f"{prefix}{connector}{label}")

    # Check depth limit before recursing
    if max_depth is not None and depth >= max_depth:
        # Show ellipsis if there are children we're hiding
        if has_children:
            lines.append(f"{child_prefix}└── ...")
        return lines

    # Collect and sort children
    sorted_children: list[tuple[str, PrefixTreeNode]] = sorted(
        node.children.items(), key=lambda kv: natural_sort_key(kv[0])
    )

    # Also include direct tensors as "leaf" entries (if node has both
    # children and direct tensors, which is uncommon but possible)
    direct_tensor_entries: list[tuple[str, int]] = []
    if has_children and has_direct_tensors:
        for tid in node.tensor_ids:
            t = tensors[tid]
            parts = t.full_name.split(".")
            leaf_name = parts[-1] if parts else t.full_name
            direct_tensor_entries.append((leaf_name, tid))

    total_entries = len(sorted_children) + len(direct_tensor_entries)

    for i, (_, child) in enumerate(sorted_children):
        is_last_entry = i == total_entries - 1
        lines.extend(
            _render_node(
                child, index, child_prefix, is_last_entry, False, depth + 1, max_depth
            )
        )

    for i, (leaf_name, tid) in enumerate(direct_tensor_entries):
        t = tensors[tid]
        is_last_entry = i == len(direct_tensor_entries) - 1
        entry_connector = "└── " if is_last_entry else "├── "
        leaf_label = (
            f"{leaf_name} "
            f"{format_shape(t.shape)} "
            f"{format_dtype(t.dtype)} "
            f"({format_bytes(t.nbytes)})"
        )
        lines.append(f"{child_prefix}{entry_connector}{leaf_label}")

    return lines


def render_tree(path: Path, max_depth: int | None = None) -> str:
    """Render a safetensors file's tensor hierarchy as an ASCII tree.

    Returns the full tree as a single string.
    """
    index = TensorIndex.from_file(path)
    tree = PrefixTree(index)

    # The root node has name="" and its children are the top-level namespaces.
    # We render each top-level child as a root-level entry.
    sorted_roots = sorted(
        tree.root.children.items(), key=lambda kv: natural_sort_key(kv[0])
    )

    lines: list[str] = []
    for i, (_, child) in enumerate(sorted_roots):
        is_last = i == len(sorted_roots) - 1
        lines.extend(_render_node(child, index, "", is_last, True, 0, max_depth))

    return "\n".join(lines)
