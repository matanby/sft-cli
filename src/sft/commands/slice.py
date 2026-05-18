"""CLI wrapper for the slice command — extract tensors matching a pattern."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.slice import slice_file


@app.command("slice", rich_help_panel="Transform")
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
) -> None:
    """Extract tensors matching a pattern into a new file."""
    file = validate_safetensors(file)

    if include is None and exclude is None:
        typer.secho(
            "Error: At least one of --include or --exclude is required.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    result = slice_file(file, output, include=include, exclude=exclude, dry_run=dry_run)

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
