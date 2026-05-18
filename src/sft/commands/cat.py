"""CLI wrapper for the cat command — merge multiple safetensors files."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.cat import cat_files


@app.command("cat", rich_help_panel="Transform")
def cat(
    files: list[Path] = typer.Argument(
        ...,
        help="Two or more .safetensors files to merge.",
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path (default: merged.safetensors in CWD).",
    ),
    allow_duplicates: bool = typer.Option(
        False,
        "--allow-duplicates",
        help="On tensor name collision, keep the version from the last file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what the merged file would contain without writing.",
    ),
) -> None:
    """Merge multiple .safetensors files into one."""
    validated = [validate_safetensors(f) for f in files]

    try:
        result = cat_files(
            validated,
            output,
            allow_duplicates=allow_duplicates,
            dry_run=dry_run,
        )
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if dry_run:
        typer.echo(
            f"Would merge {result.total_files} files → {result.total_tensors} tensors"
        )
        if result.duplicates:
            typer.echo(f"Duplicate tensors: {', '.join(result.duplicates)}")
    else:
        typer.echo(
            f"Merged {result.total_files} files → "
            f"{result.output_path} ({result.total_tensors} tensors)"
        )
