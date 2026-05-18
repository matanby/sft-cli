"""CLI wrapper for the tree command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.index import PrefixTree, PrefixTreeNode, TensorIndex
from sft.ops.tree import render_tree


def _node_to_dict(
    node: PrefixTreeNode,
    index: TensorIndex,
    depth: int,
    max_depth: int | None,
) -> dict:
    out: dict = {
        "name": node.name,
        "aggregate_count": node.aggregate_count,
        "aggregate_bytes": node.aggregate_bytes,
    }
    truncated = max_depth is not None and depth >= max_depth
    if node.tensor_ids:
        out["tensors"] = [
            {
                "name": index.tensors[tid].full_name,
                "shape": list(index.tensors[tid].shape),
                "dtype": index.tensors[tid].dtype,
                "nbytes": index.tensors[tid].nbytes,
            }
            for tid in node.tensor_ids
        ]
    if node.children:
        if truncated:
            out["truncated"] = True
        else:
            out["children"] = [
                _node_to_dict(child, index, depth + 1, max_depth)
                for _, child in sorted(node.children.items())
            ]
    return out


def _render_tree_json(file: Path, max_depth: int | None) -> str:
    index = TensorIndex.from_file(file)
    tree = PrefixTree(index)
    data = {
        "file": file.name,
        "total_tensors": index.total_tensors,
        "total_bytes": index.total_bytes,
        "max_depth": max_depth,
        "children": [
            _node_to_dict(child, index, 0, max_depth)
            for _, child in sorted(tree.root.children.items())
        ],
    }
    return json.dumps(data, indent=2)


@app.command("tree", rich_help_panel="Inspect", no_args_is_help=True)
def tree_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file.",
        resolve_path=True,
    ),
    depth: int | None = typer.Option(
        None,
        "--depth",
        "-d",
        help="Limit tree depth.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output the tree as JSON.",
    ),
) -> None:
    """Print tensor namespace hierarchy as an ASCII tree.

    Tensor names are split on '.' to form a directory-like structure.

    Examples:
      sft tree model.safetensors
      sft tree model.safetensors --depth 3
      sft tree model.safetensors --json
    """
    file = validate_safetensors(file)
    if json_output:
        typer.echo(_render_tree_json(file, max_depth=depth))
    else:
        output = render_tree(file, max_depth=depth)
        typer.echo(output)


@app.command("t", hidden=True)
def tree_alias(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file.",
        resolve_path=True,
    ),
    depth: int | None = typer.Option(
        None,
        "--depth",
        "-d",
        help="Limit tree depth.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output the tree as JSON.",
    ),
) -> None:
    """Alias for tree-cmd."""
    tree_cmd(file, depth, json_output)
