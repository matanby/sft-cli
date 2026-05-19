---
name: release-sft-cli
description: Release a new version of sft-cli to GitHub and PyPI. Use when the user asks to release, publish, ship, or bump the version of sft-cli, or create a GitHub release / PyPI deploy for this repo.
---

# Release sft-cli

End-to-end workflow to cut a GitHub release and publish `sft-cli` to PyPI.

## Prerequisites

- `gh` CLI authenticated for `matanby/sft-cli`
- `uv` available locally (`uv sync --dev`)
- Release authority: only proceed when the user explicitly requests a release

## Version bump type

If the user did **not** specify `patch`, `minor`, or `major`, ask before doing anything else.

Read current version from `src/sft/__init__.py` (`__version__`). Compute the next version (semver, no leading `v` in the file):

| Bump  | Example `0.2.0` → |
|-------|-------------------|
| patch | `0.2.1`           |
| minor | `0.3.0`           |
| major | `1.0.0`           |

Git tag format is always `v{version}` (e.g. `v0.3.0`). The tag must match `__version__` without the `v`.

## How CI and PyPI relate

| Workflow | File | Trigger |
|----------|------|---------|
| **CI** | `.github/workflows/ci.yml` | push/PR to `main` — lint, test matrix, slow tests, wheel smoke |
| **Publish to PyPI** | `.github/workflows/publish.yml` | GitHub **release published** — `uv build` then OIDC publish to https://pypi.org/project/sft-cli/ |

Publishing is **not** triggered by tags alone; the release must be **published** (not left as draft). Draft releases do not run `publish.yml`.

Version at build time comes from `src/sft/__init__.py` via Hatch (`[tool.hatch.version]` in `pyproject.toml`).

## Release checklist

Copy and track:

```
- [ ] Bump type confirmed (patch / minor / major)
- [ ] Working tree clean or user resolved uncommitted/unpushed work
- [ ] On main, synced with origin
- [ ] Local CI checks pass (or main CI already green for release commit)
- [ ] __version__ bumped and committed
- [ ] main pushed; CI green on release commit
- [ ] Release notes drafted
- [ ] GitHub release created (published)
- [ ] Publish to PyPI workflow succeeded
- [ ] PyPI shows new version
```

---

## Step 1 — Preflight: git state

Run in parallel:

```bash
git status
git branch --show-current
git log @{u}..HEAD --oneline 2>/dev/null   # unpushed commits
git log HEAD..@{u} --oneline 2>/dev/null   # behind remote
```

**Uncommitted changes:** Stop and ask the user what to do (commit, stash, or discard). Do not release with surprise local edits unless they approve.

**Unpushed commits:** Stop and ask whether to push first or wait.

**Wrong branch:** Switch to `main` only with user approval; releases target `main`.

**Behind `origin/main`:** `git pull` (fast-forward) before continuing.

---

## Step 2 — Confirm bump and previous tag

```bash
grep __version__ src/sft/__init__.py
git tag -l 'v*' --sort=-v:refname | head -5
PREV=$(git tag -l 'v*' --sort=-v:refname | head -1)   # e.g. v0.2.0
echo "Previous tag: $PREV"
```

State the computed new version and tag; get explicit user confirmation.

---

## Step 3 — CI must pass **before** tag/release

Run the same checks as CI locally on the commit you will release (current `main` HEAD, or after the version bump commit):

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Optional slow tier (matches CI `test-slow`):

```bash
uv pip install -e .
uv run pytest -q -m slow
```

If anything fails, fix first — do not create a release.

After pushing the version bump, wait for GitHub CI on `main`:

```bash
gh run list --branch main --workflow ci.yml --limit 1
gh run watch <run-id> --exit-status
```

All jobs (lint, test matrix, test-slow, build) must succeed before `gh release create`.

---

## Step 4 — Bump version

Edit `src/sft/__init__.py`:

```python
__version__ = "X.Y.Z"
```

Commit only the version bump (unless the user asked to include other changes):

```bash
git add src/sft/__init__.py
git commit -m "Bump version to X.Y.Z"
```

Push to `origin/main`, then re-run Step 3 CI watch on the new commit.

**Do not create the git tag locally** unless the user asks; `gh release create` creates the tag from the target commit.

---

## Step 5 — Draft release notes

Inspect the previous release for tone and structure:

```bash
gh release view v0.2.0   # or latest tag
```

Gather changes since the previous tag:

```bash
git log ${PREV}..HEAD --oneline
git log ${PREV}..HEAD --format='%s%n%b---'
```

Exclude the version-bump commit from feature bullets. Group commits into themed sections (match prior releases):

- **What's new in vX.Y.Z** — one-paragraph summary
- **New commands** / **LoRA toolkit** / **Interactive browser** / **Agent skill** — as applicable
- **Performance**, **Fixes**, **Tests & CI**, **Repo hygiene**
- **Full changelog** link:

  `https://github.com/matanby/sft-cli/compare/${PREV}...vX.Y.Z`

Write user-facing notes (what changed for CLI users), not a raw commit dump. Mention new commands, flags, breaking changes, and notable fixes. See `v0.2.0` for the expected depth on minor releases; patch releases can be shorter but still list fixes and small additions.

Save notes to a temp file (e.g. `/tmp/sft-release-notes.md`) for `gh release create --notes-file`.

---

## Step 6 — Create GitHub release

Target the version-bump commit on `main` (usually `HEAD` after push):

```bash
NEW=vX.Y.Z
gh release create "$NEW" \
  --title "$NEW" \
  --notes-file /tmp/sft-release-notes.md
```

Do **not** use `--draft` unless the user wants to delay PyPI. Use `--target` only if releasing an older commit (unusual).

Return the release URL to the user.

---

## Step 7 — Monitor PyPI publish workflow

The `Publish to PyPI` workflow starts on release publish:

```bash
gh run list --workflow publish.yml --limit 3
gh run watch <run-id> --exit-status
```

Both jobs must succeed: **build** (wheel/sdist) and **publish-pypi** (OIDC to PyPI).

On failure, inspect logs:

```bash
gh run view <run-id> --log-failed
```

Common issues: PyPI environment approval pending, duplicate version already on PyPI, build hook failure.

---

## Step 8 — Verify PyPI

```bash
pip index versions sft-cli 2>/dev/null | head -1
# or
curl -s https://pypi.org/pypi/sft-cli/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
```

Confirm the version matches `__version__`. Optionally smoke-install:

```bash
pip install 'sft-cli==X.Y.Z' && sft --version
```

Report success with: GitHub release URL, Actions run URL, and PyPI version.

---

## Rules

- **Never** force-push `main`, skip hooks, or publish without user approval for the version and notes.
- **Never** overwrite an existing PyPI version; bump again if a publish partially failed after upload.
- **Always** ask about uncommitted/unpushed work before bumping.
- **Always** ensure CI is green before `gh release create`.
- Regenerate `src/sft/skill/REFERENCE.md` only if CLI surface changed in this release (via `python scripts/build_skill_reference.py`); the Hatch build hook does this at wheel build time — no manual edit of `REFERENCE.md`.

## Quick reference

| Item | Value |
|------|-------|
| PyPI package | `sft-cli` |
| Console script | `sft` |
| Version file | `src/sft/__init__.py` |
| Tag pattern | `vMAJOR.MINOR.PATCH` |
| Repo | `https://github.com/matanby/sft-cli` |
