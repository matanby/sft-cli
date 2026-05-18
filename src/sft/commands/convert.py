"""CLI wrapper for the convert command — convert PyTorch checkpoints to safetensors."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app


@app.command("convert", rich_help_panel="Convert", no_args_is_help=True)
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output a JSON report.",
    ),
) -> None:
    """Convert a PyTorch checkpoint file to safetensors format.

    Supports .pt, .pth, and .bin files. Requires PyTorch to be installed.

    Examples:
      sft convert model.pt
      sft convert checkpoint.bin -o model.safetensors
      sft convert model.pth --dtype fp16
    """
    if not file.exists():
        if json_output:
            typer.echo(json.dumps({"error": f"File not found: {file}"}, indent=2))
        else:
            typer.secho(f"Error: File not found: {file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    dst = output if output is not None else file.parent / f"{file.stem}.safetensors"

    try:
        from sft.ops.convert import convert_to_safetensors

        result = convert_to_safetensors(src=file, dst=dst, dtype=dtype)
    except (ImportError, ValueError) as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}, indent=2))
        else:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        data = {
            "tensors_count": result.tensors_count,
            "source_format": result.source_format,
            "dtype": dtype,
            "output_path": str(result.output_path),
        }
        typer.echo(json.dumps(data, indent=2))
        return

    dtype_msg = f" (cast to {dtype})" if dtype else ""
    typer.echo(
        f"Converted {result.tensors_count} tensors from {result.source_format}"
        f"{dtype_msg} → {result.output_path.name}"
    )
