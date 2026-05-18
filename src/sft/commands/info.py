"""CLI wrapper for the info command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.ops.info import summarize
from sft.utils.formatting import format_bytes, format_dtype, format_number


def run(
    file: Path,
    *,
    json_output: bool = False,
) -> None:
    """Print a non-interactive summary of a .safetensors file."""
    summary = summarize(file)

    if json_output:
        _print_json(summary)
    else:
        _print_table(summary)


def _print_json(summary) -> None:
    from sft.utils.formatting import format_dtype

    data = {
        "file": summary.file_name,
        "file_size": summary.file_size,
        "tensors": summary.total_tensors,
        "total_parameters": summary.total_parameters,
        "total_tensor_bytes": summary.total_tensor_bytes,
        "dtypes": {format_dtype(d.dtype): d.count for d in summary.dtypes},
        "metadata": summary.metadata,
    }
    typer.echo(json.dumps(data, indent=2))


def _print_table(summary) -> None:
    params_formatted = (
        f"{summary.total_parameters:,} ({format_number(summary.total_parameters)})"
    )
    lines = [
        f"{'File:':<13}{summary.file_name}",
        f"{'File size:':<13}{format_bytes(summary.file_size)}",
        f"{'Tensors:':<13}{summary.total_tensors}",
        f"{'Parameters:':<13}{params_formatted}",
    ]

    lines.append("")
    lines.append("Dtype breakdown:")
    for d in summary.dtypes:
        pct = (
            d.total_bytes / summary.total_tensor_bytes * 100
            if summary.total_tensor_bytes
            else 0
        )
        lines.append(
            f"  {format_dtype(d.dtype):<10}{d.count} tensors    "
            f"{format_bytes(d.total_bytes):>9}  {pct:5.1f}%"
        )

    if summary.metadata:
        lines.append("")
        lines.append("Metadata:")
        for k, v in summary.metadata.items():
            lines.append(f"  {k}: {v}")

    typer.echo("\n".join(lines))
