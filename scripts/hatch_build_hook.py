"""Hatch build hook that attempts to regenerate `src/sft/skill/REFERENCE.md`.

Wired in pyproject.toml via `[tool.hatch.build.hooks.custom]`.

The hook runs in the build environment, which is typically isolated and does
not have the runtime deps (`typer`, `click`) or `sft` itself importable. When
those imports fail we silently keep the existing committed REFERENCE.md — devs
regenerate it manually via `python scripts/build_skill_reference.py` (and the
pre-commit hook). This keeps `uv build` working out of the box without forcing
runtime deps into `build-system.requires`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SkillReferenceHook(BuildHookInterface):
    PLUGIN_NAME = "sft-skill-reference"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        root = Path(self.root)
        scripts_dir = root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from build_skill_reference import main  # noqa: WPS433
        except Exception as exc:
            self.app.display_info(
                f"[sft-skill-reference] skipping regeneration ({exc.__class__.__name__}: {exc}); "
                "shipping the committed REFERENCE.md as-is."
            )
            return

        try:
            written = main(root / "src" / "sft" / "skill" / "REFERENCE.md")
        except Exception as exc:
            self.app.display_info(
                f"[sft-skill-reference] regeneration failed ({exc}); "
                "shipping the committed REFERENCE.md as-is."
            )
            return

        self.app.display_info(f"[sft-skill-reference] regenerated {written}")
