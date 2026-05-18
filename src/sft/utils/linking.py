"""Cross-platform directory linking with symlink / junction / copy fallback.

Used by `sft skill install` to materialise a skill directory inside an
agent's well-known skills folder. We strongly prefer a live link so that
upgrading `sft-cli` (which moves package contents in place at a stable
package path) is reflected in the installed skill without re-running
`sft skill install`. Copies are a last-resort fallback.

Strategy on each platform:

- POSIX:  os.symlink(target, link).
- Windows: try os.symlink (works under Developer Mode or admin),
           fall back to a directory junction via `mklink /J`,
           fall back to copytree.
- Cross-drive on Windows: junctions are not supported across drives;
  fall through to copytree with a warning.

Each installation drops a small sentinel file named `.sft-skill-install.json`
into the link/copy so `sft skill status` can introspect later, including
distinguishing a fresh copy from a stale one after a CLI upgrade.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

SENTINEL_NAME = ".sft-skill-install.json"


class InstallKind(str, Enum):
    SYMLINK = "symlink"
    JUNCTION = "junction"
    COPY = "copy"


class LinkMode(str, Enum):
    AUTO = "auto"
    LINK = "link"
    COPY = "copy"


@dataclass
class InstallRecord:
    """Describes an installed skill directory."""

    path: Path
    kind: InstallKind
    target: Path | None  # resolved target for links, None for copies
    fresh: bool  # for copies: True iff source mtime <= sentinel's recorded mtime


@dataclass
class InspectResult:
    """Result of inspecting a potential install location."""

    exists: bool
    kind: InstallKind | None
    target: Path | None
    sentinel: dict | None


def _write_sentinel(path: Path, kind: InstallKind, source: Path) -> None:
    """Write the install sentinel inside the link/copy."""
    payload = {
        "kind": kind.value,
        "source": str(source),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    with contextlib.suppress(OSError):
        (path / SENTINEL_NAME).write_text(json.dumps(payload, indent=2) + "\n")


def inspect(path: Path) -> InspectResult:
    """Inspect an install location without modifying it.

    Returns ``exists=False`` if nothing is there, otherwise a populated
    InspectResult. Junctions are reported as ``JUNCTION`` on Windows;
    everywhere else they appear as symlinks.
    """
    if not path.exists() and not path.is_symlink():
        return InspectResult(exists=False, kind=None, target=None, sentinel=None)

    sentinel: dict | None = None
    sentinel_path = path / SENTINEL_NAME
    if sentinel_path.exists():
        try:
            sentinel = json.loads(sentinel_path.read_text())
        except (OSError, json.JSONDecodeError):
            sentinel = None

    if path.is_symlink():
        try:
            target = Path(os.readlink(path))
        except OSError:
            target = None
        kind: InstallKind = InstallKind.SYMLINK
        if sys.platform.startswith("win") and _is_junction(path):
            kind = InstallKind.JUNCTION
        return InspectResult(exists=True, kind=kind, target=target, sentinel=sentinel)

    if sys.platform.startswith("win") and _is_junction(path):
        target = _readlink_or_none(path)
        return InspectResult(
            exists=True, kind=InstallKind.JUNCTION, target=target, sentinel=sentinel
        )

    return InspectResult(
        exists=True, kind=InstallKind.COPY, target=None, sentinel=sentinel
    )


def _is_junction(path: Path) -> bool:
    """True if *path* is a Windows directory junction (reparse point)."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import stat as _stat

        st = path.lstat()
        return bool(st.st_file_attributes & _stat.FILE_ATTRIBUTE_REPARSE_POINT)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def _readlink_or_none(path: Path) -> Path | None:
    try:
        return Path(os.readlink(path))
    except OSError:
        return None


def install_link(
    source: Path,
    dest: Path,
    *,
    mode: LinkMode = LinkMode.AUTO,
    force: bool = False,
) -> InstallRecord:
    """Materialise *source* at *dest* using the configured strategy.

    Raises:
        FileExistsError: if *dest* exists and ``force`` is False.
        RuntimeError: if every fallback fails.
    """
    source = source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source skill path is not a directory: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() or dest.is_symlink():
        if not force:
            raise FileExistsError(
                f"Destination already exists: {dest} (pass force=True to replace)"
            )
        _remove(dest)

    if mode is LinkMode.COPY:
        return _install_copy(source, dest)

    if mode is LinkMode.LINK:
        return _install_symlink(source, dest)

    return _install_auto(source, dest)


def _install_auto(source: Path, dest: Path) -> InstallRecord:
    """Try symlink, then junction (Windows), then copy."""
    try:
        return _install_symlink(source, dest)
    except OSError:
        pass

    if sys.platform.startswith("win"):
        try:
            return _install_junction(source, dest)
        except (OSError, subprocess.CalledProcessError):
            pass

    return _install_copy(source, dest)


def _install_symlink(source: Path, dest: Path) -> InstallRecord:
    os.symlink(source, dest, target_is_directory=True)
    _write_sentinel(dest, InstallKind.SYMLINK, source)
    return InstallRecord(path=dest, kind=InstallKind.SYMLINK, target=source, fresh=True)


def _install_junction(source: Path, dest: Path) -> InstallRecord:
    if not sys.platform.startswith("win"):
        raise OSError("Junctions are only supported on Windows.")
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(source)],
        check=True,
        capture_output=True,
    )
    _write_sentinel(dest, InstallKind.JUNCTION, source)
    return InstallRecord(
        path=dest, kind=InstallKind.JUNCTION, target=source, fresh=True
    )


def _install_copy(source: Path, dest: Path) -> InstallRecord:
    shutil.copytree(source, dest)
    _write_sentinel(dest, InstallKind.COPY, source)
    return InstallRecord(path=dest, kind=InstallKind.COPY, target=None, fresh=True)


def _remove(path: Path) -> None:
    """Remove *path* whether it is a symlink, junction, file, or directory."""
    if path.is_symlink() or (sys.platform.startswith("win") and _is_junction(path)):
        try:
            path.unlink()
            return
        except (OSError, IsADirectoryError):
            # Junctions on some Python/Windows combos need rmdir.
            try:
                os.rmdir(path)
                return
            except OSError:
                pass
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def uninstall(path: Path) -> bool:
    """Remove an installed skill directory. Returns True iff something was removed."""
    if not (path.exists() or path.is_symlink()):
        return False
    _remove(path)
    return True


def is_fresh(record_path: Path, source: Path) -> bool:
    """Heuristic freshness check for copies.

    For symlinks/junctions this is always True (content follows the source).
    For copies, compares source mtime against the sentinel's recorded
    `installed_at` and against the copy's SKILL.md mtime.
    """
    inspected = inspect(record_path)
    if inspected.kind in (InstallKind.SYMLINK, InstallKind.JUNCTION):
        return True
    if inspected.kind is None:
        return False

    sentinel = inspected.sentinel or {}
    installed_at = sentinel.get("installed_at")
    if not installed_at:
        return False
    try:
        installed_dt = datetime.fromisoformat(installed_at)
    except ValueError:
        return False
    if installed_dt.tzinfo is None:
        installed_dt = installed_dt.replace(tzinfo=timezone.utc)

    try:
        source_skill = source / "SKILL.md"
        source_mtime = datetime.fromtimestamp(
            source_skill.stat().st_mtime, tz=timezone.utc
        )
    except OSError:
        return True

    return source_mtime <= installed_dt
