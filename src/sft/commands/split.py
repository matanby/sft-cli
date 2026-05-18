"""CLI wrapper for the split command — shard a safetensors file by size."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.split import parse_size, split_file


@app.command("split", rich_help_panel="Transform", no_args_is_help=True)
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output a JSON report.",
    ),
) -> None:
    """Split a .safetensors file into smaller shards by size.

    Creates numbered shard files and a JSON index file.
    Size format: number followed by B, KB, MB, or GB.

    Examples:
      sft split model.safetensors --max-size 4GB
      sft split model.safetensors --max-size 500MB --dry-run
      sft split model.safetensors --max-size 2GB -o 'shard-{index}.safetensors'
    """
    file = validate_safetensors(file)

    def _fail(msg: str) -> None:
        if json_output:
            typer.echo(json.dumps({"error": msg}, indent=2))
        else:
            typer.secho(f"Error: {msg}", fg=typer.colors.RED, err=True)

    try:
        max_bytes = parse_size(max_size)
    except ValueError as exc:
        _fail(str(exc))
        raise typer.Exit(code=1) from None

    try:
        result = split_file(
            src=file,
            max_bytes=max_bytes,
            output_pattern=output,
            dry_run=dry_run,
        )
    except ValueError as exc:
        _fail(str(exc))
        raise typer.Exit(code=1) from None

    if json_output:
        data = {
            "dry_run": dry_run,
            "max_bytes": max_bytes,
            "index_path": str(result.index_path) if result.index_path else None,
            "shards": [
                {
                    "path": str(shard.path),
                    "tensor_names": shard.tensor_names,
                    "total_bytes": shard.total_bytes,
                }
                for shard in result.shards
            ],
        }
        typer.echo(json.dumps(data, indent=2))
        return

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
