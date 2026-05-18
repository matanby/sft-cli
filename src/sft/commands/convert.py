"""CLI wrapper for the convert command — convert PyTorch checkpoints to safetensors."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app


@app.command("convert", rich_help_panel="Convert")
def convert(
    file: Path = typer.Argument(
        ...,
        help="Path to a PyTorch checkpoint file (.pt, .pth, .bin).",
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.safetensors).",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Cast tensors to this dtype during conversion (e.g. fp16, fp32).",
    ),
) -> None:
    """Convert a PyTorch checkpoint file to safetensors format."""
    if not file.exists():
        typer.secho(f"Error: File not found: {file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    dst = output if output is not None else file.parent / f"{file.stem}.safetensors"

    try:
        from sft.ops.convert import convert_to_safetensors

        result = convert_to_safetensors(src=file, dst=dst, dtype=dtype)
    except ImportError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    dtype_msg = f" (cast to {dtype})" if dtype else ""
    typer.echo(
        f"Converted {result.tensors_count} tensors from {result.source_format}"
        f"{dtype_msg} → {result.output_path.name}"
    )
