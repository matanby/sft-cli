# AGENTS.md — guidance for AI agents working on `sft-cli`

This file is read by Claude Code, Cursor, Codex CLI, and other agents to understand the repo. It is meant for agents **modifying this codebase**, not end-users of the CLI.

## What this project is

`sft-cli` is a CLI ("`sft`") for inspecting and editing `.safetensors` files. It is published on PyPI as `sft-cli` and the user-facing command is `sft`.

## Repo layout

- `src/sft/cli.py` — Typer app entry point, the auto-`browse` shim, and top-level commands (`browse`, `info`, `check`). Every other subcommand is registered by importing its module at the bottom of this file.
- `src/sft/commands/<name>.py` — thin CLI wrappers (Typer surface, argument parsing, formatting, `--json`). They delegate to `src/sft/ops/<name>.py`.
- `src/sft/ops/<name>.py` — pure business logic. No Typer, no printing; returns dataclasses. Always testable in isolation.
- `src/sft/utils/` — shared helpers (formatting, dtype mapping, glob filtering, tensor IO, cross-platform linking for the skill installer).
- `src/sft/browser.py` — the interactive Textual TUI (the `browse` command). Self-contained, do not import from other commands.
- `src/sft/index.py` — the `TensorIndex` / `PrefixTree` data model, parsed header-only from `.safetensors` files.
- `src/sft/skill/` — the agent skill that ships inside the wheel and is installed by `sft skill install`. Two files: `SKILL.md` (hand-written: trigger description, command map, cookbook of cross-command patterns) and `REFERENCE.md` (**auto-generated** — never edit by hand).
- `scripts/build_skill_reference.py` — regenerates `src/sft/skill/REFERENCE.md` from the live Typer CLI.
- `scripts/hatch_build_hook.py` — hatch build hook that runs the regenerator at wheel-build time (tolerates missing deps in isolated builds).
- `tests/` — pytest tests, one file per command. Shared fixtures in `tests/conftest.py`.

## Conventions

1. **`--json` everywhere on commands an agent might parse.** Every command supports `--json` and outputs a stable JSON contract. When you add a new command, add `--json` from day one and update `SKILL.md`. The pattern is:

   ```python
   if json_output:
       typer.echo(json.dumps(data, indent=2))
       return
   # human-readable fallback below
   ```

   Errors raised via `--json` should also be JSON: `typer.echo(json.dumps({"error": str(e)}, indent=2))` before `raise typer.Exit(code=1)`.

2. **Separation of concerns.** A new feature `foo` means:
   - Pure logic in `src/sft/ops/foo.py` (returns a dataclass).
   - CLI wrapper in `src/sft/commands/foo.py` (imports `ops/foo.py`, owns flags and printing).
   - Register by adding `import sft.commands.foo` to the bottom of `src/sft/cli.py` (the imports there look stylistically dead but they trigger `@app.command(...)` decorators — keep them sorted alphabetically).
   - Tests in `tests/test_foo.py` (CLI tests use `typer.testing.CliRunner`).

3. **The auto-`browse` shim** in `_entry()` rewrites `sft x.safetensors` → `sft browse x.safetensors`. Do not add subcommands whose names end in `.safetensors`.

4. **`validate_safetensors(path)`** in `src/sft/cli.py` is the canonical "is this a `.safetensors` file?" check. Use it from every command that takes a `.safetensors` argument.

5. **Output paths** use the `resolve_output(output, src, suffix)` helper in `src/sft/utils/output.py` to default to `{stem}.{suffix}.safetensors`. Write commands should never overwrite the source file.

6. **Dry-run.** Every write-side command exposes `--dry-run` and the underlying op must support `dry_run=True` to return a result without touching disk.

7. **Header-only reads.** Prefer `TensorIndex.from_file(path)` (header-only, milliseconds) over loading tensor data unless you specifically need values. Header-only operations are why `sft` exists.

## Updating the agent skill

The shipped skill lives in `src/sft/skill/`:

- `SKILL.md` — hand-written: trigger description (YAML frontmatter), golden `--json` rule, command map, "when to use" guidance, and an inline cookbook of cross-command patterns and `jq` recipes.
- `REFERENCE.md` — **auto-generated** from Typer introspection. Regenerate with:

  ```bash
  python scripts/build_skill_reference.py
  ```

  This also runs as a hatch build hook during `uv build`, so the wheel always contains an up-to-date reference. The script is best-effort: if it can't import `sft` (e.g. inside an isolated build env), it leaves the committed file alone.

When you add a new command or flag, also:

- Update the command-map table in `SKILL.md` if the command is something agents should reach for.
- If the command enables a cross-command workflow or a useful `--json` parsing pattern that isn't obvious from the single-command help, add it to the "Cookbook" section in `SKILL.md`. Resist the urge to write per-command tutorials — `REFERENCE.md` already covers single-command usage.

## Dev commands

```bash
uv sync                                          # install deps
uv run pytest -q                                  # full test suite
uv run pytest tests/test_<command>.py -q         # one file
uv run ruff check                                # lint
uv run ruff format                               # format
python scripts/build_skill_reference.py          # regenerate skill reference
uv build                                          # produce wheel + sdist (regenerates skill ref via hook)
uv run sft skill install --dry-run               # preview a skill install locally
```

## Known stylistic notes

- Top-level imports in `src/sft/cli.py` use `# noqa: F401, E402` because they're after function definitions and are imported for their side effect (registering commands). Keep that pattern.
- We support Python ≥ 3.9. Avoid `match` statements, walrus inside comprehensions, and other newer syntax in shared code paths. Use `from __future__ import annotations` at the top of every module.
- The pre-commit config runs `ruff` (lint + format). Run `uv run ruff check` and `uv run ruff format` before committing.

## Out-of-scope for changes

- Do **not** add an MCP server module. The skill + reliable `--json` is the intended agent integration story.
- Do **not** silently rewrite `src/sft/skill/REFERENCE.md` by hand; always regenerate via the script.
- Do **not** invent new install locations for the skill installer without updating `AGENT_DIRS` in `src/sft/commands/skill.py` and the corresponding docs in `SKILL.md` and `README.md`.
