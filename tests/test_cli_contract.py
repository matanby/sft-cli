"""CLI contract suite: every command in the `sft` Typer app honors a stable
contract on `--help`, `no_args_is_help`, `--json`, `--dry-run`, and default
output paths.

This file is parametrized over the full command surface so adding a new
command only requires registering it in COMMANDS / WRITE_COMMANDS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


# Every user-facing command path (omitting hidden aliases like `b`, `i`, `t`).
# A "path" is the chain of subcommand tokens needed to reach the leaf command.
ALL_COMMANDS: list[list[str]] = [
    # Top-level Inspect / Transform / Convert
    ["browse"],
    ["info"],
    ["check"],
    ["ls"],
    ["tree"],
    ["stat"],
    ["metadata"],
    ["cast"],
    ["cat"],
    ["convert"],
    ["rename"],
    ["slice"],
    ["split"],
    ["strip"],
    ["diff"],
    # LoRA group
    ["lora", "extract"],
    ["lora", "info"],
    ["lora", "resize"],
    ["lora", "add"],
    ["lora", "stack"],
    ["lora", "svd"],
    ["lora", "compat"],
    ["lora", "convert"],
    ["lora", "merge"],
    # Skill group (uninstall and install are write-side and tested elsewhere;
    # status / path / update are listed here for --help coverage only).
    ["skill", "install"],
    ["skill", "uninstall"],
    ["skill", "status"],
    ["skill", "update"],
    ["skill", "path"],
]


# Commands that accept --json on the happy path and a single .safetensors arg.
JSON_HAPPY_COMMANDS: list[list[str]] = [
    ["info"],
    ["ls"],
    ["tree"],
    ["stat"],
    ["metadata"],
    ["check"],
    ["lora", "info"],
]


# Write-side commands that should honor --dry-run, with the args to drive them.
# Each entry: (path, build_args(model_path, tmp_path) -> list[str]).
def _strip_args(model: Path, tmp: Path) -> list[str]:
    return [
        str(model),
        "--exclude",
        "**.q_proj.weight",
        "-o",
        str(tmp / "out.safetensors"),
    ]


def _slice_args(model: Path, tmp: Path) -> list[str]:
    return [
        str(model),
        "--include",
        "**.q_proj.weight",
        "-o",
        str(tmp / "out.safetensors"),
    ]


def _cast_args(model: Path, tmp: Path) -> list[str]:
    return [str(model), "--dtype", "fp32", "-o", str(tmp / "out.safetensors")]


def _rename_args(model: Path, tmp: Path) -> list[str]:
    return [
        str(model),
        "--sub",
        r"^model\.",
        "--sub",
        "renamed.",
        "-o",
        str(tmp / "out.safetensors"),
    ]


def _split_args(model: Path, tmp: Path) -> list[str]:
    pattern = str(tmp / "shard-{index}.safetensors")
    return [str(model), "--max-size", "1KB", "-o", pattern]


WRITE_COMMANDS = [
    (["strip"], _strip_args),
    (["slice"], _slice_args),
    (["cast"], _cast_args),
    (["rename"], _rename_args),
    (["split"], _split_args),
]


# ---------- 1. Every command supports --help ----------


@pytest.mark.parametrize("path", ALL_COMMANDS, ids=lambda p: " ".join(p))
def test_command_supports_help(path: list[str]) -> None:
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output
    # The leaf command name appears in the usage line.
    assert path[-1] in result.output


# ---------- 2. Commands declared with no_args_is_help actually behave so ----------

# Skip subcommand-groups that intentionally accept zero args (auto-detect or
# print their own listing): skill install/uninstall/update/status/path.
_NO_ARGS_EXEMPT = {
    ("skill", "install"),
    ("skill", "uninstall"),
    ("skill", "update"),
    ("skill", "status"),
    ("skill", "path"),
}
NO_ARGS_COMMANDS = [p for p in ALL_COMMANDS if tuple(p) not in _NO_ARGS_EXEMPT]


@pytest.mark.parametrize("path", NO_ARGS_COMMANDS, ids=lambda p: " ".join(p))
def test_command_no_args_shows_help(path: list[str]) -> None:
    result = runner.invoke(app, path)
    # `no_args_is_help=True` -> Typer exits non-zero and prints usage.
    assert result.exit_code != 0
    assert "Usage" in result.output or "usage" in result.output


# ---------- 3. --json happy-path returns parseable JSON ----------


@pytest.mark.parametrize("path", JSON_HAPPY_COMMANDS, ids=lambda p: " ".join(p))
def test_command_json_happy_path_parses(
    path: list[str], mini_model: Path, lora_adapter: Path
) -> None:
    target = lora_adapter if path[0] == "lora" else mini_model
    args = [*path, str(target), "--json"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    # Stdout must be parseable as JSON (object or array).
    data = json.loads(result.output)
    assert isinstance(data, (dict, list))


# ---------- 4. --json error contract (currently a partial contract) ----------


@pytest.mark.xfail(
    reason=(
        "Documented in AGENTS.md but not yet enforced in every command: "
        "errors raised under --json should be emitted as JSON "
        "({'error': '...'}). validate_safetensors() prints a plain "
        "stderr message and exits. Tracked as a gap; this xfail "
        "flips to pass when the contract is fully implemented."
    ),
    strict=False,
)
@pytest.mark.parametrize("path", JSON_HAPPY_COMMANDS, ids=lambda p: " ".join(p))
def test_command_json_error_is_json(path: list[str], tmp_path: Path) -> None:
    fake = tmp_path / "does_not_exist.safetensors"
    args = [*path, str(fake), "--json"]
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    # Strict contract: error output is itself JSON.
    data = json.loads(result.output)
    assert "error" in data


# ---------- 5. --dry-run on write commands: no file is written ----------


@pytest.mark.parametrize(
    "path,arg_builder",
    WRITE_COMMANDS,
    ids=[" ".join(p) for p, _ in WRITE_COMMANDS],
)
def test_write_command_dry_run_does_not_write(
    path: list[str],
    arg_builder,
    mini_model: Path,
    tmp_path: Path,
) -> None:
    args = arg_builder(mini_model, tmp_path)
    result = runner.invoke(app, [*path, *args, "--dry-run"])
    assert result.exit_code == 0, result.output
    # No new .safetensors files should appear in tmp_path under dry-run.
    written = [p for p in tmp_path.rglob("*.safetensors") if p != mini_model]
    assert written == [], f"dry-run wrote files: {written}"


# ---------- 6. Default output never overwrites the input ----------


@pytest.mark.parametrize(
    "path,arg_builder",
    [pair for pair in WRITE_COMMANDS if pair[0] not in (["split"],)],
    ids=[" ".join(p) for p, _ in WRITE_COMMANDS if p not in (["split"],)],
)
def test_write_command_default_output_differs_from_input(
    path: list[str],
    arg_builder,
    mini_model: Path,
    tmp_path: Path,
) -> None:
    # Build args, then strip the `-o ...` pair so the command falls back to
    # the default output path computed by resolve_output().
    args = arg_builder(mini_model, tmp_path)
    if "-o" in args:
        i = args.index("-o")
        args = args[:i] + args[i + 2 :]
    result = runner.invoke(app, [*path, *args])
    assert result.exit_code == 0, result.output
    # The original file is untouched and a sibling file with a suffix exists.
    assert mini_model.exists()
    siblings = [
        p
        for p in mini_model.parent.iterdir()
        if p.suffix == ".safetensors" and p != mini_model
    ]
    assert siblings, "expected a default-named output sibling to be created"


# ---------- 7. Header-only inspect commands don't read tensor data ----------


def test_inspect_commands_are_header_only(
    mini_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`info`, `ls`, `tree`, `metadata` should never call `.get_tensor()`
    or otherwise materialize tensor data. We assert this by spying on
    `safetensors.safe_open` and rejecting any call to `get_tensor` /
    `get_slice` on the returned file handle.
    """
    import safetensors

    real_safe_open = safetensors.safe_open
    forbidden_calls: list[str] = []

    class SpyFile:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

        def keys(self):
            return self._inner.keys()

        def metadata(self):
            return self._inner.metadata()

        def get_tensor(self, name):
            forbidden_calls.append(f"get_tensor({name!r})")
            return self._inner.get_tensor(name)

        def get_slice(self, name):
            forbidden_calls.append(f"get_slice({name!r})")
            return self._inner.get_slice(name)

    def fake_safe_open(*args, **kwargs):
        return SpyFile(real_safe_open(*args, **kwargs))

    monkeypatch.setattr(safetensors, "safe_open", fake_safe_open)

    for cmd in (["info"], ["ls"], ["tree"], ["metadata"]):
        forbidden_calls.clear()
        result = runner.invoke(app, [*cmd, str(mini_model)])
        assert result.exit_code == 0, (cmd, result.output)
        assert forbidden_calls == [], (
            f"{cmd} should be header-only but called: {forbidden_calls}"
        )
