"""CLI wrapper for the strip command — remove tensors matching a pattern."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.slice import strip_file


@app.command("strip", rich_help_panel="Transform")
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
) -> None:
    """Remove tensors matching a pattern from a file."""
    file = validate_safetensors(file)

    result = strip_file(file, output, exclude=exclude, dry_run=dry_run)

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
