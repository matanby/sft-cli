"""CLI wrapper for the check command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.ops.check import CheckResult, check_file


def run(file: Path, *, skip_values: bool = False, json_output: bool = False) -> None:
    """Validate a .safetensors file and print results."""
    result = check_file(file, skip_values=skip_values)
    if json_output:
        _print_json(result)
    else:
        _print_result(result)

    if not result.healthy:
        raise typer.Exit(code=1)


def _print_json(result: CheckResult) -> None:
    data = {
        "healthy": result.healthy,
        "header_ok": result.header_ok,
        "header_error": result.header_error or None,
        "num_tensors": result.num_tensors,
        "offsets_ok": result.offsets_ok,
        "offsets_error": result.offsets_error or None,
        "dtypes": result.dtypes,
        "values_checked": result.values_checked,
        "nan_tensors": result.nan_tensors,
        "inf_tensors": result.inf_tensors,
    }
    typer.echo(json.dumps(data, indent=2))


def _print_result(result: CheckResult) -> None:
    if result.header_ok:
        _ok(f"Header: valid JSON, {result.num_tensors} tensors")
    else:
        _fail(f"Header: {result.header_error}")
        _fail("File has issues")
        return

    if result.offsets_ok:
        _ok("Offsets: all data offsets within file bounds")
    else:
        _fail(f"Offsets: {result.offsets_error}")

    dtype_str = ", ".join(result.dtypes) if result.dtypes else "none"
    _ok(f"Dtypes: all tensors {dtype_str}")

    if not result.values_checked:
        _ok("Values: skipped (--skip-values)")
    elif result.nan_tensors or result.inf_tensors:
        parts: list[str] = []
        for name in result.nan_tensors:
            parts.append(f"found NaN in {name}")
        for name in result.inf_tensors:
            parts.append(f"found Inf in {name}")
        _fail(f"Values: {', '.join(parts)}")
    else:
        _ok("Values: no NaN or Inf detected")

    if result.healthy:
        _ok("File healthy")
    else:
        _fail("File has issues")


def _ok(msg: str) -> None:
    typer.echo(f"✓ {msg}")


def _fail(msg: str) -> None:
    typer.echo(f"✗ {msg}")
