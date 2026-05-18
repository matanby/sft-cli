"""CLI wrapper for the diff command — compare two safetensors files."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.diff import TensorDiff, diff_files
from sft.utils.formatting import format_bytes, format_dtype, format_shape


def _print_structural(
    diff: TensorDiff, name_a: str, name_b: str, info_a: dict, info_b: dict
) -> None:
    total_a = len(info_a)
    total_b = len(info_b)
    typer.echo(
        f"Compared: {name_a} ({total_a} tensors) vs {name_b} ({total_b} tensors)"
    )
    typer.echo()

    typer.echo(f"Added ({len(diff.added)}):")
    for name in diff.added:
        t = info_b[name]
        shape = format_shape(t.shape)
        dtype = format_dtype(t.dtype)
        size = format_bytes(t.nbytes)
        typer.echo(f"  + {name}  {shape}  {dtype}  {size}")

    typer.echo()
    typer.echo(f"Removed ({len(diff.removed)}):")
    for name in diff.removed:
        t = info_a[name]
        shape = format_shape(t.shape)
        dtype = format_dtype(t.dtype)
        size = format_bytes(t.nbytes)
        typer.echo(f"  - {name}  {shape}  {dtype}  {size}")

    typer.echo()
    typer.echo(f"Shape changed ({len(diff.shape_changed)}):")
    for name, (sa, sb) in diff.shape_changed.items():
        typer.echo(f"  ~ {name}  {format_shape(sa)} → {format_shape(sb)}")

    typer.echo()
    typer.echo(f"Dtype changed ({len(diff.dtype_changed)}):")
    for name, (da, db) in diff.dtype_changed.items():
        typer.echo(f"  ~ {name}  {format_dtype(da)} → {format_dtype(db)}")

    typer.echo()
    typer.echo(f"Unchanged: {len(diff.unchanged)} tensors")


def _print_delta(diff: TensorDiff) -> None:
    if diff.value_diffs is None:
        return

    header = f"{'Tensor':<45}{'L2 norm(Δ)':>12}  {'cosine sim':>10}"
    typer.echo(header)
    changed = 0
    total_cos = 0.0
    for name, vd in diff.value_diffs.items():
        typer.echo(f"{name:<45}{vd.l2_norm:>12.4f}  {vd.cosine_sim:>10.4f}")
        total_cos += vd.cosine_sim
        if vd.l2_norm > 0:
            changed += 1

    total = len(diff.value_diffs)
    avg_cos = total_cos / total if total else 0.0
    typer.echo(
        f"Summary: {changed}/{total} tensors changed, avg cosine similarity {avg_cos:.4f}"
    )


def _to_json(diff: TensorDiff) -> None:
    data: dict = {
        "added": diff.added,
        "removed": diff.removed,
        "shape_changed": {
            k: {"a": list(sa), "b": list(sb)}
            for k, (sa, sb) in diff.shape_changed.items()
        },
        "dtype_changed": {
            k: {"a": format_dtype(da), "b": format_dtype(db)}
            for k, (da, db) in diff.dtype_changed.items()
        },
        "unchanged": diff.unchanged,
    }
    if diff.value_diffs is not None:
        data["value_diffs"] = {
            name: {"l2_norm": vd.l2_norm, "cosine_sim": vd.cosine_sim}
            for name, vd in diff.value_diffs.items()
        }
    typer.echo(json.dumps(data, indent=2))


@app.command("diff", rich_help_panel="Transform")
def diff_cmd(
    file_a: Path = typer.Argument(
        ...,
        help="First .safetensors file (base).",
        resolve_path=True,
    ),
    file_b: Path = typer.Argument(
        ...,
        help="Second .safetensors file (target).",
        resolve_path=True,
    ),
    delta: bool = typer.Option(
        False,
        "--delta",
        help="Compute value-level differences (L2 norm, cosine similarity).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as machine-readable JSON.",
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
) -> None:
    """Compare two .safetensors files and show structural or value differences."""
    file_a = validate_safetensors(file_a)
    file_b = validate_safetensors(file_b)

    from sft.index import TensorIndex

    index_a = TensorIndex.from_file(file_a)
    index_b = TensorIndex.from_file(file_b)
    info_a = {t.full_name: t for t in index_a.tensors}
    info_b = {t.full_name: t for t in index_b.tensors}

    result = diff_files(
        file_a,
        file_b,
        compute_delta=delta,
        include=include,
        exclude=exclude,
    )

    if json_output:
        _to_json(result)
    elif delta:
        _print_delta(result)
    else:
        _print_structural(result, file_a.name, file_b.name, info_a, info_b)
