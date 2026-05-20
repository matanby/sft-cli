"""CLI wrapper for the metadata command — view/edit safetensors file metadata."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.ops.metadata import get_metadata, set_metadata
from sft.utils.output import resolve_output


def _parse_key_value(raw: str) -> tuple[str, str]:
    """Parse a 'key=value' string, raising on bad format."""
    if "=" not in raw:
        raise typer.BadParameter(f"Expected key=value, got '{raw}'")
    key, _, value = raw.partition("=")
    return key, value


@app.command("metadata", rich_help_panel="Transform", no_args_is_help=True)
def metadata(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file.",
        resolve_path=True,
    ),
    set_pairs: list[str] | None = typer.Option(
        None,
        "--set",
        help="Set a metadata key-value pair (key=value). Can be repeated.",
    ),
    unset_keys: list[str] | None = typer.Option(
        None,
        "--unset",
        help="Remove a metadata key. Can be repeated.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output metadata as JSON.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.metadata.safetensors).",
    ),
) -> None:
    """View or edit metadata in a .safetensors file.

    Without --set or --unset, prints existing metadata. With --set/--unset,
    writes a new file with updated metadata.

    Examples:
      sft metadata model.safetensors
      sft metadata model.safetensors --json
      sft metadata model.safetensors --set format=pt --set version=2.0
      sft metadata model.safetensors --unset format
    """
    file = validate_safetensors(file, json_output=json_output)

    is_write = bool(set_pairs) or bool(unset_keys)

    if not is_write:
        md = get_metadata(file)
        if json_output:
            typer.echo(json.dumps(md, indent=2))
        elif md:
            for key, value in md.items():
                typer.echo(f"{key}: {value}")
        else:
            typer.echo("No metadata found.")
        return

    parsed_set: dict[str, str] = {}
    for pair in set_pairs or []:
        try:
            k, v = _parse_key_value(pair)
        except typer.BadParameter as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        parsed_set[k] = v

    dst = resolve_output(output, file, "metadata")

    result = set_metadata(
        src=file,
        dst=dst,
        set_keys=parsed_set or None,
        unset_keys=list(unset_keys) if unset_keys else None,
    )

    typer.echo(f"Wrote {len(result.metadata)} metadata key(s) → {dst.name}")
