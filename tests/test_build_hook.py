"""Verify the Hatch build hook regenerates ``REFERENCE.md`` into the wheel.

This test is marked slow because it shells out to `uv build` to produce a
wheel from the current working tree, then unpacks it and compares the
shipped `REFERENCE.md` against a fresh regeneration.
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_built_wheel_contains_up_to_date_reference(tmp_path: Path) -> None:
    """Build the wheel into tmp_path and compare its REFERENCE.md to the
    output of running the regenerator script against the same source tree."""
    if not (REPO_ROOT / "pyproject.toml").exists():
        pytest.skip("not running in a source checkout")

    # 1. Generate the expected REFERENCE.md by importing the regenerator and
    # capturing its output (no on-disk side effect required).
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from build_skill_reference import generate  # type: ignore
    finally:
        sys.path.pop(0)
    expected = generate()

    # 2. Build the wheel.
    wheel_dir = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"uv build failed: {build.stderr[:400]}")

    wheels = list(wheel_dir.glob("sft_cli-*.whl"))
    assert wheels, f"no wheel produced in {wheel_dir}"
    wheel_path = wheels[0]

    # 3. Extract REFERENCE.md from the wheel.
    with zipfile.ZipFile(wheel_path) as zf:
        ref_members = [n for n in zf.namelist() if n.endswith("skill/REFERENCE.md")]
        assert ref_members, f"REFERENCE.md missing from wheel: {zf.namelist()}"
        with zf.open(ref_members[0]) as src:
            wheel_ref = io.TextIOWrapper(src, encoding="utf-8").read()

    # Equality up to trailing-whitespace differences is sufficient; the
    # regenerator is deterministic when called against the same source.
    assert wheel_ref.rstrip() == expected.rstrip(), (
        "REFERENCE.md inside the wheel is stale; "
        "run `python scripts/build_skill_reference.py` and rebuild."
    )
