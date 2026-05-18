"""CLI wrapper for the cat command — merge multiple safetensors files."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.cat import cat_files


@app.command("cat", rich_help_panel="Transform", no_args_is_help=True)
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output a JSON report.",
    ),
) -> None:
    """Merge multiple .safetensors files into one.

    Concatenates tensors from all input files. Fails on name collisions
    unless --allow-duplicates is set (last file wins).

    Examples:
      sft cat shard_01.safetensors shard_02.safetensors -o merged.safetensors
      sft cat *.safetensors --allow-duplicates
      sft cat part1.safetensors part2.safetensors --dry-run
    """
    validated = [validate_safetensors(f) for f in files]

    try:
        result = cat_files(
            validated,
            output,
            allow_duplicates=allow_duplicates,
            dry_run=dry_run,
        )
    except ValueError as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        data = {
            "dry_run": dry_run,
            "total_files": result.total_files,
            "total_tensors": result.total_tensors,
            "duplicates": result.duplicates,
            "output_path": str(result.output_path) if result.output_path else None,
        }
        typer.echo(json.dumps(data, indent=2))
        return

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
