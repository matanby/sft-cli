"""CLI wrapper for the split command — shard a safetensors file by size."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.split import parse_size, split_file


@app.command("split", rich_help_panel="Transform")
def split(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to split.",
        resolve_path=True,
    ),
    max_size: str = typer.Option(
        ...,
        "--max-size",
        help="Maximum shard size (e.g. '4GB', '500MB', '1024B').",
    ),
    output: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filename pattern (must contain '{index}'). "
        "Default: {stem}-{index}.safetensors",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show shard distribution without writing files.",
    ),
) -> None:
    """Split a .safetensors file into smaller shards by size."""
    file = validate_safetensors(file)

    try:
        max_bytes = parse_size(max_size)
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    try:
        result = split_file(
            src=file,
            max_bytes=max_bytes,
            output_pattern=output,
            dry_run=dry_run,
        )
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if dry_run:
        typer.echo(f"Would create {len(result.shards)} shard(s):")
        for shard in result.shards:
            typer.echo(
                f"  {shard.path.name}: {len(shard.tensor_names)} tensors, "
                f"{shard.total_bytes:,} bytes"
            )
    else:
        typer.echo(
            f"Split into {len(result.shards)} shard(s), index: {result.index_path}"
        )
        for shard in result.shards:
            typer.echo(
                f"  {shard.path.name}: {len(shard.tensor_names)} tensors, "
                f"{shard.total_bytes:,} bytes"
            )
