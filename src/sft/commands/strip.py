"""CLI wrapper for the strip command — remove tensors matching a pattern."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.slice import strip_file


@app.command("strip", rich_help_panel="Transform", no_args_is_help=True)
def strip_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file.",
        resolve_path=True,
    ),
    exclude: str = typer.Option(
        ...,
        "--exclude",
        help="Glob pattern for tensor names to remove.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path (default: {stem}.stripped.safetensors).",
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
    """Remove tensors matching a pattern from a file.

    Glob patterns:
      *name*       matches any tensor containing "name"
      **.weight    matches tensors ending with ".weight" at any depth
      model.*.*    matches exactly two segments under "model"

    Examples:
      sft strip model.safetensors --exclude='*lora_A*'
      sft strip model.safetensors --exclude='**.bias' -o slim.safetensors
      sft strip model.safetensors --exclude='*lora_A*' --dry-run
    """
    file = validate_safetensors(file)

    result = strip_file(file, output, exclude=exclude, dry_run=dry_run)

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
            typer.echo(f"Would strip {len(result.excluded)} tensor(s):")
            for name in result.excluded:
                typer.echo(f"  - {name}")
    else:
        typer.echo(f"Wrote {len(result.included)} tensor(s) to {result.output_path}")
