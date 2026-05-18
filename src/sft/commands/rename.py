"""CLI wrapper for the rename command — regex-based tensor key renaming."""

from __future__ import annotations

from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.rename import rename_tensors
from sft.utils.output import resolve_output


@app.command("rename", rich_help_panel="Transform")
def rename(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to rename keys in.",
        resolve_path=True,
    ),
    sub: list[str] = typer.Option(
        [],
        "--sub",
        help="Regex substitution: --sub PATTERN REPLACEMENT (can repeat).",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.renamed.safetensors).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show old → new name mappings without writing.",
    ),
) -> None:
    """Rename tensor keys using regex substitution."""
    file = validate_safetensors(file)

    if len(sub) == 0:
        typer.secho(
            "Error: at least one --sub pair is required.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    if len(sub) % 2 != 0:
        typer.secho(
            "Error: --sub requires pairs of PATTERN REPLACEMENT.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    substitutions = [(sub[i], sub[i + 1]) for i in range(0, len(sub), 2)]
    dst = resolve_output(output, file, "renamed")

    result = rename_tensors(
        src=file, dst=dst, substitutions=substitutions, dry_run=dry_run
    )

    if dry_run:
        if result.mappings:
            for m in result.mappings:
                typer.echo(f"  {m.old_name} → {m.new_name}")
            typer.echo(
                f"\nWould rename {len(result.mappings)} tensor(s), {result.unchanged} unchanged"
            )
        else:
            typer.secho(
                "Warning: no tensors matched the substitution pattern(s).",
                fg=typer.colors.YELLOW,
            )
            typer.echo(f"{result.unchanged} tensor(s) unchanged")
    else:
        if not result.mappings:
            typer.secho(
                "Warning: no tensors matched the substitution pattern(s).",
                fg=typer.colors.YELLOW,
            )
        typer.echo(
            f"Renamed {len(result.mappings)} tensor(s), "
            f"{result.unchanged} unchanged → {result.output_path.name}"
        )
