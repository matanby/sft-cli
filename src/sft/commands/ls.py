"""CLI wrapper for the ls command — tabular summary of .safetensors files."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.ls import FileSummaryRow, list_files
from sft.utils.formatting import format_bytes, format_dtype, format_number


class SortField(str, Enum):
    name = "name"
    size = "size"
    params = "params"
    tensors = "tensors"


_SORT_KEYS: dict[SortField, callable] = {
    SortField.name: lambda r: r.file_name.lower(),
    SortField.size: lambda r: r.total_bytes,
    SortField.params: lambda r: r.total_params,
    SortField.tensors: lambda r: r.total_tensors,
}


def _format_dtypes(dtypes: set[str]) -> str:
    return ", ".join(sorted(format_dtype(d) for d in dtypes))


def _print_table(rows: list[FileSummaryRow]) -> None:
    headers = ("File", "Tensors", "Params", "Size", "Dtypes")
    formatted = [
        (
            r.file_name,
            format_number(r.total_tensors),
            format_number(r.total_params),
            format_bytes(r.total_bytes),
            _format_dtypes(r.dtypes),
        )
        for r in rows
    ]

    col_widths = [len(h) for h in headers]
    for row in formatted:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad_row(cells: tuple[str, ...]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.ljust(col_widths[i]))
        return "  ".join(parts)

    typer.echo(_pad_row(headers))
    for row in formatted:
        typer.echo(_pad_row(row))


def _to_json(rows: list[FileSummaryRow]) -> None:
    data = [
        {
            "file": r.file_name,
            "tensors": r.total_tensors,
            "params": r.total_params,
            "bytes": r.total_bytes,
            "dtypes": sorted(format_dtype(d) for d in r.dtypes),
        }
        for r in rows
    ]
    typer.echo(json.dumps(data, indent=2))


@app.command(rich_help_panel="Inspect", no_args_is_help=True)
def ls(
    files: list[Path] = typer.Argument(
        ...,
        help="One or more .safetensors files to summarise.",
        resolve_path=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON array.",
    ),
    sort: SortField | None = typer.Option(
        None,
        "--sort",
        help="Sort rows by: name, size, params, tensors.",
    ),
) -> None:
    """Print a tabular summary of one or more .safetensors files.

    Shows tensor count, parameter count, file size, and dtypes for each file.

    Examples:
      sft ls model.safetensors
      sft ls *.safetensors --sort=size
      sft ls shard_*.safetensors --json
    """
    validated = [validate_safetensors(f, json_output=json_output) for f in files]
    rows = list_files(validated)

    if not rows:
        raise typer.Exit(code=1)

    if sort is not None:
        rows.sort(key=_SORT_KEYS[sort])

    if json_output:
        _to_json(rows)
    else:
        _print_table(rows)
