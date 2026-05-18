"""CLI entry point for sft — the Swiss army knife for .safetensors files."""

from __future__ import annotations

from pathlib import Path

import typer

from sft import __version__

app = typer.Typer(
    name="sft",
    help="The Swiss army knife for .safetensors files.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sft {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """The Swiss army knife for .safetensors files."""


def validate_safetensors(path: Path) -> Path:
    """Validate that a path points to a readable .safetensors file."""
    if not path.exists():
        typer.secho(f"Error: File not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if path.suffix.lower() != ".safetensors":
        typer.secho(
            f"Error: Expected a .safetensors file, got '{path.suffix}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return path


@app.command(rich_help_panel="Inspect")
def browse(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to browse.",
        resolve_path=True,
    ),
) -> None:
    """Open an interactive TUI browser for a .safetensors file."""
    file = validate_safetensors(file)
    from sft.browser import SftApp

    SftApp(file).run()


# Short alias for browse
@app.command("b", hidden=True)
def browse_alias(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to browse.",
        resolve_path=True,
    ),
) -> None:
    """Alias for browse."""
    browse(file)


@app.command(rich_help_panel="Inspect")
def info(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to inspect.",
        resolve_path=True,
    ),
    json: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """Print a non-interactive summary of a .safetensors file."""
    file = validate_safetensors(file)
    from sft.commands.info import run

    run(file, json_output=json)


# Short alias for info
@app.command("i", hidden=True)
def info_alias(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to inspect.",
        resolve_path=True,
    ),
    json: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """Alias for info."""
    info(file, json=json)


import sft.commands.cast  # noqa: F401, E402
import sft.commands.cat  # noqa: F401, E402
import sft.commands.convert  # noqa: F401, E402
import sft.commands.diff  # noqa: F401, E402
import sft.commands.lora  # noqa: F401, E402
import sft.commands.ls  # noqa: F401, E402
import sft.commands.metadata  # noqa: F401, E402
import sft.commands.rename  # noqa: F401, E402
import sft.commands.slice  # noqa: F401, E402
import sft.commands.split  # noqa: F401, E402
import sft.commands.stat  # noqa: F401, E402
import sft.commands.strip  # noqa: F401, E402
import sft.commands.tree  # noqa: F401, E402


@app.command(rich_help_panel="Inspect")
def check(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to check.",
        resolve_path=True,
    ),
    skip_values: bool = typer.Option(
        False,
        "--skip-values",
        help="Skip NaN/Inf scan (faster for huge files).",
    ),
) -> None:
    """Validate a .safetensors file's integrity and check for NaN/Inf values."""
    file = validate_safetensors(file)
    from sft.commands.check import run

    run(file, skip_values=skip_values)


if __name__ == "__main__":
    app()
