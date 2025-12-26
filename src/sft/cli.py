"""CLI entry point for sft."""

from pathlib import Path

import typer

from sft import __version__

app = typer.Typer(
    name="sft",
    help="An interactive terminal browser for .safetensors files.",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"sft {__version__}")
        raise typer.Exit()


@app.command()
def main(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to browse.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Open an interactive browser for a .safetensors file."""
    # Validate file extension
    if file.suffix.lower() != ".safetensors":
        typer.secho(
            f"Error: Expected a .safetensors file, got '{file.suffix}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Import here to avoid slow startup for --help/--version
    from sft.browser import SftApp

    # Launch the TUI
    app_instance = SftApp(file)
    app_instance.run()


if __name__ == "__main__":
    app()
