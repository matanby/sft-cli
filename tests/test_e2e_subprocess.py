"""End-to-end tests that invoke the real ``sft`` console script as a subprocess.

These complement the in-process CliRunner tests by exercising:
  * the ``_entry()`` shim that rewrites ``sft model.safetensors`` to
    ``sft browse model.safetensors``;
  * argument quoting, exit codes and stdout/stderr separation as the user
    actually sees them;
  * shell pipelines combining multiple commands.

Skips automatically if no console script is on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

pytestmark = pytest.mark.slow


# ---------- Fixture: locate the installed sft executable ----------


@pytest.fixture(scope="session")
def sft_cmd() -> list[str]:
    """Return the argv prefix for invoking the installed `sft` command."""
    # Prefer the script installed by uv pip install -e .; fall back to
    # `python -m sft.cli` which exercises the same entry point.
    found = shutil.which("sft")
    if found:
        return [found]
    return [sys.executable, "-m", "sft.cli"]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
        **kw,
    )


# ---------- Console script presence ----------


def test_sft_help_via_subprocess(sft_cmd: list[str]) -> None:
    result = _run([*sft_cmd, "--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout


def test_sft_version_via_subprocess(sft_cmd: list[str]) -> None:
    result = _run([*sft_cmd, "--version"])
    assert result.returncode == 0, result.stderr
    assert "sft" in result.stdout


# ---------- _entry() shim: bare-file invocation rewrites to `browse` ----------


def test_entry_shim_rewrites_bare_safetensors_to_browse(
    sft_cmd: list[str], tmp_path: Path
) -> None:
    """`sft model.safetensors` should auto-rewrite to `sft browse model.safetensors`.

    Browse normally launches a TUI which we cannot drive from a pipe, but we
    can verify the rewrite by intercepting ``SftApp.run``: the cli.browse()
    function imports SftApp from sft.browser then calls .run() — if the
    shim worked, we land in that code path. The cleanest probe is to set
    an env var that browse honors; lacking that, we run the script directly
    and confirm it doesn't error out as "Missing argument FILE" (the
    no-args browse failure mode).
    """
    model = tmp_path / "mini.safetensors"
    save_file({"w": np.ones((2, 2), dtype=np.float32)}, str(model))

    # `browse` opens a Textual App that blocks on a TTY. If the shim worked
    # the process either hangs in the app (subprocess timeout) or exits
    # with a Textual-style error -- the one outcome it MUST NOT produce is
    # Typer's "no such command" / "unexpected extra argument" parse error,
    # which would indicate the shim didn't rewrite the argv.
    try:
        result = subprocess.run(
            [*sft_cmd, str(model)],
            capture_output=True,
            text=True,
            env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
            stdin=subprocess.DEVNULL,
            timeout=3,
        )
        combined = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or b"").decode("utf-8", "ignore") + (
            exc.stderr or b""
        ).decode("utf-8", "ignore")
    assert "Got unexpected extra argument" not in combined
    assert "No such command" not in combined


# ---------- Real JSON parses out of stdout ----------


def test_info_json_subprocess(sft_cmd: list[str], tmp_path: Path) -> None:
    model = tmp_path / "tiny.safetensors"
    save_file(
        {"w": np.ones((3, 3), dtype=np.float32)},
        str(model),
        metadata={"k": "v"},
    )
    result = _run([*sft_cmd, "info", str(model), "--json"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["tensors"] == 1
    assert data["metadata"]["k"] == "v"


# ---------- Multi-command shell pipeline: extract -> resize -> info ----------


def test_pipeline_extract_resize_info(sft_cmd: list[str], tmp_path: Path) -> None:
    """A realistic multi-step flow exercised through three separate subprocess
    invocations. Asserts each step writes the expected output and the final
    JSON reflects the resized rank.
    """
    rng = np.random.RandomState(0)

    base_path = tmp_path / "base.safetensors"
    ft_path = tmp_path / "ft.safetensors"
    W = rng.randn(8, 8).astype(np.float32)
    delta = 0.05 * rng.randn(8, 8).astype(np.float32)
    save_file({"model.layers.0.self_attn.q_proj.weight": W}, str(base_path))
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": W + delta},
        str(ft_path),
    )

    adapter_path = tmp_path / "adapter.safetensors"
    r = _run(
        [
            *sft_cmd,
            "lora",
            "extract",
            str(base_path),
            str(ft_path),
            "--rank",
            "4",
            "-o",
            str(adapter_path),
        ]
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert adapter_path.exists()

    resized_path = tmp_path / "resized.safetensors"
    r = _run(
        [
            *sft_cmd,
            "lora",
            "resize",
            str(adapter_path),
            "--rank",
            "2",
            "-o",
            str(resized_path),
        ]
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert resized_path.exists()

    r = _run([*sft_cmd, "lora", "info", str(resized_path), "--json"])
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info["rank"] == 2


# ---------- Missing-file path: exit non-zero with a useful message ----------


def test_missing_file_exits_nonzero(sft_cmd: list[str], tmp_path: Path) -> None:
    fake = tmp_path / "does_not_exist.safetensors"
    r = _run([*sft_cmd, "info", str(fake)])
    assert r.returncode != 0
    combined = (r.stdout or "") + (r.stderr or "")
    assert "not found" in combined.lower() or "no such" in combined.lower()
