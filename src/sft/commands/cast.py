"""CLI wrapper for the cast command — dtype conversion of safetensors tensors."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.cast import cast_file
from sft.utils.dtypes import VALID_DTYPES
from sft.utils.output import resolve_output


@app.command("cast", rich_help_panel="Transform", no_args_is_help=True)
def cast(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to cast.",
        resolve_path=True,
    ),
    dtype: str = typer.Option(
        ...,
        "--dtype",
        help=f"Target dtype ({', '.join(VALID_DTYPES)}).",
    ),
    include: str | None = typer.Option(
        None,
        "--include",
        help="Only cast tensors matching this glob pattern.",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Skip tensors matching this glob pattern.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.{dtype}.safetensors).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be cast without writing.",
    ),
) -> None:
    """Cast tensor dtypes in a .safetensors file.

    Convert all (or selected) tensors to a different dtype. Useful for
    quantising models to fp16/bf16 or promoting to fp32.

    Examples:
      sft cast model.safetensors --dtype fp16
      sft cast model.safetensors --dtype bf16 --exclude='*.norm.*'
      sft cast model.safetensors --dtype fp32 --include='**.weight' --dry-run
    """
    file = validate_safetensors(file)
    dst = resolve_output(output, file, dtype)

    try:
        result = cast_file(
            src=file,
            dst=dst,
            target_dtype=dtype,
            include=include,
            exclude=exclude,
            dry_run=dry_run,
        )
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if dry_run:
        typer.echo(
            f"Would cast {result.cast_count} tensor(s), skip {result.skipped_count}"
        )
    else:
        typer.echo(
            f"Cast {result.cast_count} tensor(s) to {dtype}, "
            f"skipped {result.skipped_count} → {result.output_path.name}"
        )
