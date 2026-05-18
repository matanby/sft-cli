"""`sft skill` — install the sft agent skill for AI agents.

Materialises ``src/sft/skill/`` from the installed package into each agent's
well-known skills directory. The resolved skill source path is a stable
location inside the installed package, so a symlink/junction install will
automatically pick up content changes when ``sft-cli`` is upgraded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer

from sft.cli import app
from sft.utils.linking import (
    InstallKind,
    LinkMode,
    inspect,
    install_link,
    is_fresh,
)
from sft.utils.linking import uninstall as link_uninstall

SKILL_DIRNAME = "sft"

#: Mapping from agent key → (user-scope dir, project-scope dir relative to cwd).
AGENT_DIRS: dict[str, tuple[str, str]] = {
    "agents": ("~/.agents/skills", ".agents/skills"),
    "claude": ("~/.claude/skills", ".claude/skills"),
    "cursor": ("~/.cursor/skills", ".cursor/skills"),
    "codex": ("~/.codex/skills", ".codex/skills"),
}

ALL_AGENTS = list(AGENT_DIRS.keys())

skill_app = typer.Typer(
    name="skill",
    help="Install the sft agent skill for AI coding agents (Claude / Cursor / Codex / open standard).",
    no_args_is_help=True,
)
app.add_typer(skill_app, rich_help_panel="Agents")


# ----------------------------------------------------------------------
# Resolution helpers
# ----------------------------------------------------------------------


def _resolve_source() -> Path:
    """Return the on-disk path to the shipped skill directory.

    Uses importlib.resources so this works whether sft is installed via
    ``uv tool install``, ``pip install``, or an editable install. Falls
    back to the in-repo path for unusual setups.
    """
    try:
        from importlib.resources import files
    except ImportError:  # pragma: no cover — Python <3.9, blocked by pyproject
        files = None  # type: ignore[assignment]

    if files is not None:
        try:
            res = files("sft").joinpath("skill")
            path = Path(str(res))
            if path.is_dir():
                return path
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            pass

    # Editable / dev fallback: skill lives next to this module.
    here = Path(__file__).resolve().parent.parent / "skill"
    if here.is_dir():
        return here

    raise RuntimeError(
        "Could not locate the sft skill directory inside the installed package."
    )


def _resolve_target(agent: str, scope: str) -> Path:
    user_str, project_str = AGENT_DIRS[agent]
    if scope == "user":
        base = Path(user_str).expanduser()
    elif scope == "project":
        base = Path(project_str)
    else:
        raise typer.BadParameter(f"Unknown scope: {scope}")
    return base / SKILL_DIRNAME


def _parse_agents(agent_values: list[str] | None) -> list[str]:
    """Resolve --agent values to a concrete agent list.

    Accepts repeated --agent flags, ``all``, and ``auto``. ``auto`` selects
    every known agent dir that already exists on disk (user scope), plus
    ``agents`` (the open standard) which we always install for. ``all``
    forces every known agent dir.
    """
    if not agent_values:
        return _auto_detect_agents()

    out: list[str] = []
    seen: set[str] = set()
    for value in agent_values:
        for sub in value.split(","):
            sub = sub.strip()
            if not sub:
                continue
            if sub == "all":
                for name in ALL_AGENTS:
                    if name not in seen:
                        out.append(name)
                        seen.add(name)
                continue
            if sub == "auto":
                for name in _auto_detect_agents():
                    if name not in seen:
                        out.append(name)
                        seen.add(name)
                continue
            if sub not in AGENT_DIRS:
                raise typer.BadParameter(
                    f"Unknown --agent {sub!r}; choose from "
                    f"{', '.join([*ALL_AGENTS, 'auto', 'all'])}."
                )
            if sub not in seen:
                out.append(sub)
                seen.add(sub)
    return out


def _auto_detect_agents() -> list[str]:
    """Pick agents whose home directory exists; always include the open standard."""
    selected = ["agents"]  # ~/.agents/skills is the cross-agent standard
    for name in ALL_AGENTS:
        if name == "agents":
            continue
        home_marker = Path(AGENT_DIRS[name][0]).expanduser().parent
        if home_marker.exists():
            selected.append(name)
    return selected


def _parse_mode(value: str) -> LinkMode:
    try:
        return LinkMode(value)
    except ValueError as exc:
        raise typer.BadParameter("--mode must be one of: auto, link, copy") from exc


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


@dataclass
class _ActionRow:
    agent: str
    scope: str
    target: Path
    action: str
    kind: str | None = None
    detail: str | None = None


def _emit(rows: list[_ActionRow], json_output: bool, header: str) -> None:
    if json_output:
        payload = [
            {
                "agent": r.agent,
                "scope": r.scope,
                "target": str(r.target),
                "action": r.action,
                "kind": r.kind,
                "detail": r.detail,
            }
            for r in rows
        ]
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(header)
    if not rows:
        typer.echo("  (no actions)")
        return
    width = max(len(r.agent) for r in rows)
    for r in rows:
        kind_str = f" [{r.kind}]" if r.kind else ""
        detail_str = f"  — {r.detail}" if r.detail else ""
        typer.echo(
            f"  {r.agent.ljust(width)}  {r.action}{kind_str}  {r.target}{detail_str}"
        )


@skill_app.command("install")
def install_cmd(
    agent: list[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help=(
            "Which agent(s) to install for. Repeatable, comma-separated, or "
            "'all'/'auto'. Default: auto-detect installed agents (always "
            "includes ~/.agents/ for the open standard)."
        ),
    ),
    scope: str = typer.Option(
        "user",
        "--scope",
        "-s",
        case_sensitive=False,
        help="'user' (default, ~/...) or 'project' (./...).",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        "-m",
        case_sensitive=False,
        help="'auto' (symlink → junction → copy), 'link' (force symlink), or 'copy'.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Replace an existing install at the target.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would happen without changing anything.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a JSON report.",
    ),
) -> None:
    """Install the sft agent skill for one or more agents.

    Examples:
      sft skill install                            # auto-detect installed agents
      sft skill install --agent claude --agent cursor
      sft skill install --scope project            # install into ./.agents/skills/sft/
      sft skill install --mode copy --force
    """
    source = _resolve_source()
    agents = _parse_agents(agent)
    link_mode = _parse_mode(mode.lower())
    scope_str = scope.lower()
    if scope_str not in ("user", "project"):
        raise typer.BadParameter("--scope must be 'user' or 'project'.")

    rows: list[_ActionRow] = []
    had_error = False

    for ag in agents:
        target = _resolve_target(ag, scope_str)
        current = inspect(target)

        if dry_run:
            if current.exists and not force:
                rows.append(
                    _ActionRow(
                        agent=ag,
                        scope=scope_str,
                        target=target,
                        action="skip",
                        kind=current.kind.value if current.kind else None,
                        detail="already installed (use --force to replace)",
                    )
                )
            else:
                rows.append(
                    _ActionRow(
                        agent=ag,
                        scope=scope_str,
                        target=target,
                        action="would-install",
                        detail=f"mode={link_mode.value}",
                    )
                )
            continue

        if current.exists and not force:
            rows.append(
                _ActionRow(
                    agent=ag,
                    scope=scope_str,
                    target=target,
                    action="skip",
                    kind=current.kind.value if current.kind else None,
                    detail="already installed (use --force to replace)",
                )
            )
            continue

        try:
            record = install_link(source, target, mode=link_mode, force=force)
        except Exception as exc:  # noqa: BLE001 — surface to the user
            had_error = True
            rows.append(
                _ActionRow(
                    agent=ag,
                    scope=scope_str,
                    target=target,
                    action="error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        rows.append(
            _ActionRow(
                agent=ag,
                scope=scope_str,
                target=record.path,
                action="installed",
                kind=record.kind.value,
                detail=(
                    f"target={record.target}"
                    if record.target is not None
                    else f"source={source}"
                ),
            )
        )

    header = (
        f"sft skill install ({scope_str} scope, source: {source})"
        if not dry_run
        else f"sft skill install --dry-run ({scope_str} scope, source: {source})"
    )
    _emit(rows, json_output, header)

    if had_error:
        raise typer.Exit(code=1)


@skill_app.command("uninstall")
def uninstall_cmd(
    agent: list[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help="Which agent(s) to uninstall from. Same syntax as install.",
    ),
    scope: str = typer.Option("user", "--scope", "-s", case_sensitive=False),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Remove the sft skill from one or more agents."""
    agents = _parse_agents(agent)
    scope_str = scope.lower()
    if scope_str not in ("user", "project"):
        raise typer.BadParameter("--scope must be 'user' or 'project'.")

    rows: list[_ActionRow] = []
    for ag in agents:
        target = _resolve_target(ag, scope_str)
        current = inspect(target)
        if not current.exists:
            rows.append(
                _ActionRow(
                    agent=ag,
                    scope=scope_str,
                    target=target,
                    action="absent",
                )
            )
            continue
        if dry_run:
            rows.append(
                _ActionRow(
                    agent=ag,
                    scope=scope_str,
                    target=target,
                    action="would-remove",
                    kind=current.kind.value if current.kind else None,
                )
            )
            continue
        link_uninstall(target)
        rows.append(
            _ActionRow(
                agent=ag,
                scope=scope_str,
                target=target,
                action="removed",
                kind=current.kind.value if current.kind else None,
            )
        )

    _emit(rows, json_output, f"sft skill uninstall ({scope_str} scope)")


@skill_app.command("status")
def status_cmd(
    scope: str = typer.Option("user", "--scope", "-s", case_sensitive=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show installation status for every known agent."""
    scope_str = scope.lower()
    if scope_str not in ("user", "project"):
        raise typer.BadParameter("--scope must be 'user' or 'project'.")

    source = _resolve_source()
    rows: list[_ActionRow] = []
    for ag in ALL_AGENTS:
        target = _resolve_target(ag, scope_str)
        current = inspect(target)
        if not current.exists:
            rows.append(
                _ActionRow(agent=ag, scope=scope_str, target=target, action="absent")
            )
            continue

        fresh = is_fresh(target, source)
        detail_parts: list[str] = []
        if current.target is not None:
            detail_parts.append(f"target={current.target}")
        if current.kind is InstallKind.COPY:
            detail_parts.append("fresh" if fresh else "STALE (run `sft skill update`)")
        rows.append(
            _ActionRow(
                agent=ag,
                scope=scope_str,
                target=target,
                action="present",
                kind=current.kind.value if current.kind else None,
                detail=", ".join(detail_parts) or None,
            )
        )

    _emit(rows, json_output, f"sft skill status ({scope_str} scope, source: {source})")


@skill_app.command("update")
def update_cmd(
    agent: list[str] = typer.Option(None, "--agent", "-a"),
    scope: str = typer.Option("user", "--scope", "-s", case_sensitive=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Refresh existing installs.

    No-op for working symlinks/junctions (the target already points at the
    current source). For copies, replaces the copy with a fresh one.
    """
    source = _resolve_source()
    agents = _parse_agents(agent)
    scope_str = scope.lower()
    if scope_str not in ("user", "project"):
        raise typer.BadParameter("--scope must be 'user' or 'project'.")

    rows: list[_ActionRow] = []
    had_error = False

    for ag in agents:
        target = _resolve_target(ag, scope_str)
        current = inspect(target)

        if not current.exists:
            rows.append(
                _ActionRow(agent=ag, scope=scope_str, target=target, action="absent")
            )
            continue

        if current.kind in (InstallKind.SYMLINK, InstallKind.JUNCTION):
            rows.append(
                _ActionRow(
                    agent=ag,
                    scope=scope_str,
                    target=target,
                    action="ok",
                    kind=current.kind.value,
                    detail="live link",
                )
            )
            continue

        try:
            record = install_link(source, target, mode=LinkMode.COPY, force=True)
        except Exception as exc:  # noqa: BLE001
            had_error = True
            rows.append(
                _ActionRow(
                    agent=ag,
                    scope=scope_str,
                    target=target,
                    action="error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        rows.append(
            _ActionRow(
                agent=ag,
                scope=scope_str,
                target=record.path,
                action="refreshed",
                kind=record.kind.value,
            )
        )

    _emit(rows, json_output, f"sft skill update ({scope_str} scope, source: {source})")
    if had_error:
        raise typer.Exit(code=1)


@skill_app.command("path")
def path_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
    """Print the resolved source skill directory inside the installed package."""
    source = _resolve_source()
    if json_output:
        typer.echo(json.dumps({"source": str(source)}, indent=2))
    else:
        typer.echo(str(source))
