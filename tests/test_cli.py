"""Tests for CLI entry point and subcommand routing."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "sft" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer's no_args_is_help uses exit code 0 on some versions, 2 on others
    assert result.exit_code in (0, 2)
    assert "safetensors" in result.output.lower() or "usage" in result.output.lower()


def test_browse_subcommand_validates_extension(tmp_path: Path):
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("not a safetensors file")
    result = runner.invoke(app, ["browse", str(bad_file)])
    assert result.exit_code == 1
    assert "safetensors" in result.output.lower()


def test_browse_alias_validates_extension(tmp_path: Path):
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("not a safetensors file")
    result = runner.invoke(app, ["b", str(bad_file)])
    assert result.exit_code == 1
    assert "safetensors" in result.output.lower()


def test_browse_subcommand_rejects_missing_file():
    result = runner.invoke(app, ["browse", "/nonexistent/model.safetensors"])
    assert result.exit_code != 0


# ------------------------------------------------------------------
# _entry() bare-file shim
# ------------------------------------------------------------------


def _entry_rewrite(argv: list[str]) -> None:
    """Mirror the rewrite logic from _entry for isolated testing."""
    args = argv[1:]
    if (
        args
        and not args[0].startswith("-")
        and args[0].lower().endswith(".safetensors")
    ):
        argv.insert(1, "browse")


def test_entry_rewrite_inserts_browse():
    argv = ["sft", "model.safetensors"]
    _entry_rewrite(argv)
    assert argv == ["sft", "browse", "model.safetensors"]


def test_entry_rewrite_no_op_for_subcommand():
    argv = ["sft", "info", "model.safetensors"]
    _entry_rewrite(argv)
    assert argv == ["sft", "info", "model.safetensors"]


def test_entry_rewrite_no_op_for_flags():
    argv = ["sft", "--version"]
    _entry_rewrite(argv)
    assert argv == ["sft", "--version"]


def test_entry_rewrite_no_op_for_non_safetensors():
    argv = ["sft", "model.gguf"]
    _entry_rewrite(argv)
    assert argv == ["sft", "model.gguf"]


def test_entry_rewrite_case_insensitive():
    argv = ["sft", "Model.SafeTensors"]
    _entry_rewrite(argv)
    assert argv == ["sft", "browse", "Model.SafeTensors"]


def test_bare_safetensors_validates_extension(tmp_path: Path):
    """Running via the CliRunner with a bare .safetensors path (simulating
    the argv that _entry would produce after rewrite)."""
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("nope")
    # After _entry rewrites, the runner would see ["browse", path]
    result = runner.invoke(app, ["browse", str(bad_file)])
    assert result.exit_code == 1
    assert "safetensors" in result.output.lower()
