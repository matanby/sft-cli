"""CLI wrapper for the diff command — compare two safetensors files."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.diff import DEFAULT_ATOL, DEFAULT_RTOL, TensorDiff, diff_files
from sft.utils.formatting import format_bytes, format_dtype, format_shape

_STATUS_COLORS = {
    "equal": typer.colors.GREEN,
    "close": typer.colors.CYAN,
    "differ": typer.colors.RED,
}


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

    n_equal = len(diff.by_status("equal"))
    n_close = len(diff.by_status("close"))
    n_differ = len(diff.by_status("differ"))

    typer.echo(f"rtol={diff.rtol:g}  atol={diff.atol:g}")
    typer.echo()
    typer.secho(f"  equal  {n_equal}", fg=_STATUS_COLORS["equal"])
    typer.secho(f"  close  {n_close}", fg=_STATUS_COLORS["close"])
    typer.secho(f"  differ {n_differ}", fg=_STATUS_COLORS["differ"])
    typer.echo()

    header = (
        f"{'status':<8}{'tensor':<48}"
        f"{'max_abs':>11}  {'mean_abs':>11}  {'rel_L2':>10}  {'cosine':>8}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, vd in diff.value_diffs.items():
        color = _STATUS_COLORS.get(vd.status)
        line = (
            f"{vd.status:<8}{name:<48}"
            f"{vd.max_abs:>11.3e}  {vd.mean_abs:>11.3e}  "
            f"{vd.rel_l2:>10.3e}  {vd.cosine_sim:>8.4f}"
        )
        typer.secho(line, fg=color)


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
        data["rtol"] = diff.rtol
        data["atol"] = diff.atol
        data["value_diffs"] = {
            name: {
                "status": vd.status,
                "max_abs": vd.max_abs,
                "mean_abs": vd.mean_abs,
                "l2_norm": vd.l2_norm,
                "rel_l2": vd.rel_l2,
                "cosine_sim": vd.cosine_sim,
            }
            for name, vd in diff.value_diffs.items()
        }
    typer.echo(json.dumps(data, indent=2))


@app.command("diff", rich_help_panel="Transform", no_args_is_help=True)
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
        help="Compute value-level differences (max_abs, mean_abs, L2, rel_L2, cosine).",
    ),
    rtol: float = typer.Option(
        DEFAULT_RTOL,
        "--rtol",
        help="Relative tolerance for the 'close' classification (used with --delta).",
    ),
    atol: float = typer.Option(
        DEFAULT_ATOL,
        "--atol",
        help="Absolute tolerance for the 'close' classification (used with --delta).",
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
    """Compare two .safetensors files and show structural or value differences.

    Shows added/removed tensors, shape changes, and dtype changes.

    With --delta, also computes per-tensor numerical metrics for every tensor
    that shares name + shape + dtype between both files: max_abs, mean_abs,
    L2 (Frobenius of the difference), rel_L2 (L2 / ||a||), and cosine sim.
    Each comparable tensor is classified as `equal` (bitwise identical),
    `close` (within rtol/atol via numpy.allclose), or `differ`.

    Examples:
      sft diff base.safetensors finetuned.safetensors
      sft diff v1.safetensors v2.safetensors --delta
      sft diff v1.safetensors v2.safetensors --delta --rtol 1e-3
      sft diff a.safetensors b.safetensors --include='**.weight' --json
    """
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
        rtol=rtol,
        atol=atol,
    )

    if json_output:
        _to_json(result)
    elif delta:
        _print_delta(result)
    else:
        _print_structural(result, file_a.name, file_b.name, info_a, info_b)
