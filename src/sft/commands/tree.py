"""CLI wrapper for the tree command."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.tree import render_tree


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
) -> None:
    """Print tensor namespace hierarchy as an ASCII tree.

    Tensor names are split on '.' to form a directory-like structure.

    Examples:
      sft tree model.safetensors
      sft tree model.safetensors --depth 3
    """
    file = validate_safetensors(file)
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
) -> None:
    """Alias for tree-cmd."""
    tree_cmd(file, depth)
