"""CLI wrapper for the stat command — per-tensor statistics."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.stat import compute_stats


def _print_table(stats_list) -> None:
    headers = (
        "Tensor",
        "dtype",
        "shape",
        "mean",
        "std",
        "min",
        "max",
        "sparsity",
        "nan",
        "inf",
    )

    def _fmt(s):
        from sft.utils.formatting import format_shape

        return (
            s.name,
            s.dtype,
            format_shape(s.shape),
            f"{s.mean:.4f}",
            f"{s.std:.4f}",
            f"{s.min:.4f}",
            f"{s.max:.4f}",
            f"{s.sparsity:.1%}",
            str(s.nan_count),
            str(s.inf_count),
        )

    rows = [_fmt(s) for s in stats_list]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad_row(cells):
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    typer.echo(_pad_row(headers))
    for row in rows:
        typer.echo(_pad_row(row))


def _print_json(stats_list) -> None:
    data = [
        {
            "name": s.name,
            "dtype": s.dtype,
            "shape": list(s.shape),
            "mean": s.mean,
            "std": s.std,
            "min": s.min,
            "max": s.max,
            "sparsity": s.sparsity,
            "nan_count": s.nan_count,
            "inf_count": s.inf_count,
        }
        for s in stats_list
    ]
    typer.echo(json.dumps(data, indent=2))


@app.command("stat", rich_help_panel="Inspect")
def stat(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file.",
        resolve_path=True,
    ),
    include: str | None = typer.Option(
        None,
        "--include",
        help="Glob pattern to include tensors.",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Glob pattern to exclude tensors.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Exit code 1 if any NaN or Inf found.",
    ),
) -> None:
    """Compute per-tensor statistics (mean, std, min, max, sparsity, NaN/Inf)."""
    file = validate_safetensors(file)
    results = compute_stats(file, include=include, exclude=exclude)

    if json_output:
        _print_json(results)
    else:
        _print_table(results)

    if check:
        has_bad = any(s.nan_count > 0 or s.inf_count > 0 for s in results)
        if has_bad:
            bad_names = [s.name for s in results if s.nan_count > 0 or s.inf_count > 0]
            typer.secho(
                f"NaN/Inf detected in: {', '.join(bad_names)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
