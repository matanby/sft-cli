"""CLI entry point for sft."""

from pathlib import Path

import typer

from sft import __version__

SUPPORTED_EXTENSIONS = {".safetensors", ".pt", ".pth"}

app = typer.Typer(
    name="sft",
    help="An interactive terminal browser for .safetensors and .pt/.pth files.",
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
        help="Path to a .safetensors, .pt, or .pth file to browse.",
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
    """Open an interactive browser for a .safetensors, .pt, or .pth file."""
    # Validate file extension
    if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        typer.secho(
            f"Error: Expected a .safetensors, .pt, or .pth file, got '{file.suffix}'",
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
