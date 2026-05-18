"""CLI wrapper for the slice command — extract tensors matching a pattern."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.slice import slice_file


@app.command("slice", rich_help_panel="Transform", no_args_is_help=True)
def slice_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file.",
        resolve_path=True,
    ),
    include: str | None = typer.Option(
        None,
        "--include",
        help="Glob pattern for tensor names to keep.",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Glob pattern for tensor names to remove.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path (default: {stem}.sliced.safetensors).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be included/removed without writing.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output a JSON report.",
    ),
) -> None:
    """Extract tensors matching a pattern into a new file.

    Use --include to keep only matching tensors, --exclude to remove
    matching tensors, or both to include first then exclude.

    Examples:
      sft slice model.safetensors --include='**.weight'
      sft slice model.safetensors --exclude='**.bias' -o no_bias.safetensors
      sft slice model.safetensors --include='model.layers.0.**' --dry-run
    """
    file = validate_safetensors(file)

    if include is None and exclude is None:
        msg = "At least one of --include or --exclude is required."
        if json_output:
            typer.echo(json.dumps({"error": msg}, indent=2))
        else:
            typer.secho(f"Error: {msg}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    result = slice_file(file, output, include=include, exclude=exclude, dry_run=dry_run)

    if json_output:
        data = {
            "dry_run": dry_run,
            "included": result.included,
            "excluded": result.excluded,
            "output_path": str(result.output_path) if not dry_run else None,
        }
        typer.echo(json.dumps(data, indent=2))
        return

    if dry_run:
        typer.echo(f"Would keep {len(result.included)} tensor(s):")
        for name in result.included:
            typer.echo(f"  + {name}")
        if result.excluded:
            typer.echo(f"Would remove {len(result.excluded)} tensor(s):")
            for name in result.excluded:
                typer.echo(f"  - {name}")
    else:
        typer.echo(f"Wrote {len(result.included)} tensor(s) to {result.output_path}")
