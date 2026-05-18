"""Tests for the ``sft skill`` command group and the linking utility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sft.cli import app
from sft.commands import skill as skill_cmd
from sft.utils import linking

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``~`` to a temp dir so install paths are sandboxed."""
    monkeypatch.setenv("HOME", str(tmp_path))
    if sys.platform.startswith("win"):
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_skill_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pretend the package's skill dir is a temp dir with SKILL.md + REFERENCE.md."""
    source = tmp_path / "_skill_src"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: sft\ndescription: test\n---\nbody")
    (source / "REFERENCE.md").write_text("# reference")
    monkeypatch.setattr(skill_cmd, "_resolve_source", lambda: source)
    return source


# ---------------------------------------------------------------------------
# linking helper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink test")
def test_install_link_creates_symlink_on_posix(
    tmp_path: Path, fake_skill_source: Path
) -> None:
    dest = tmp_path / "agent" / "skills" / "sft"
    record = linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.AUTO)
    assert record.kind is linking.InstallKind.SYMLINK
    assert dest.is_symlink()
    assert (dest / "SKILL.md").exists()
    assert (dest / "SKILL.md").read_text().startswith("---")


def test_install_link_copy_mode_creates_directory(
    tmp_path: Path, fake_skill_source: Path
) -> None:
    dest = tmp_path / "agent" / "sft"
    record = linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.COPY)
    assert record.kind is linking.InstallKind.COPY
    assert not dest.is_symlink()
    assert dest.is_dir()
    assert (dest / "SKILL.md").exists()
    sentinel = dest / linking.SENTINEL_NAME
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["kind"] == "copy"
    assert payload["source"] == str(fake_skill_source.resolve())


def test_install_link_refuses_existing_without_force(
    tmp_path: Path, fake_skill_source: Path
) -> None:
    dest = tmp_path / "agent" / "sft"
    linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.COPY)
    with pytest.raises(FileExistsError):
        linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.COPY)


def test_install_link_force_replaces_existing(
    tmp_path: Path, fake_skill_source: Path
) -> None:
    dest = tmp_path / "agent" / "sft"
    linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.COPY)
    record = linking.install_link(
        fake_skill_source, dest, mode=linking.LinkMode.COPY, force=True
    )
    assert record.kind is linking.InstallKind.COPY


def test_inspect_reports_absent(tmp_path: Path) -> None:
    result = linking.inspect(tmp_path / "nothing" / "here")
    assert result.exists is False


def test_inspect_reports_copy(tmp_path: Path, fake_skill_source: Path) -> None:
    dest = tmp_path / "agent" / "sft"
    linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.COPY)
    result = linking.inspect(dest)
    assert result.exists
    assert result.kind is linking.InstallKind.COPY
    assert result.sentinel is not None


def test_uninstall_removes_link_and_copy(
    tmp_path: Path, fake_skill_source: Path
) -> None:
    dest = tmp_path / "agent" / "sft"
    linking.install_link(fake_skill_source, dest, mode=linking.LinkMode.COPY)
    assert linking.uninstall(dest) is True
    assert not dest.exists()
    assert linking.uninstall(dest) is False  # idempotent


# ---------------------------------------------------------------------------
# CLI: sft skill install / uninstall / status / update / path
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_skill_source")
def test_skill_install_auto_detect(fake_home: Path) -> None:
    (fake_home / ".cursor").mkdir()

    result = runner.invoke(app, ["skill", "install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    agents_acted_on = {row["agent"]: row for row in payload}
    # `agents` is always on; `cursor` was detected; the others should be skipped.
    assert "agents" in agents_acted_on
    assert "cursor" in agents_acted_on
    assert "claude" not in agents_acted_on
    assert "codex" not in agents_acted_on

    for row in payload:
        assert row["action"] == "installed"
    assert (fake_home / ".agents" / "skills" / "sft" / "SKILL.md").exists()
    assert (fake_home / ".cursor" / "skills" / "sft" / "SKILL.md").exists()


@pytest.mark.usefixtures("fake_skill_source")
def test_skill_install_explicit_agent(fake_home: Path) -> None:
    result = runner.invoke(app, ["skill", "install", "--agent", "claude", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["agent"] for row in payload] == ["claude"]
    assert (fake_home / ".claude" / "skills" / "sft" / "SKILL.md").exists()


@pytest.mark.usefixtures("fake_home", "fake_skill_source")
def test_skill_install_idempotent_and_force() -> None:
    runner.invoke(app, ["skill", "install", "--agent", "agents"])
    second = runner.invoke(app, ["skill", "install", "--agent", "agents", "--json"])
    assert second.exit_code == 0
    payload = json.loads(second.output)
    assert payload[0]["action"] == "skip"

    forced = runner.invoke(
        app, ["skill", "install", "--agent", "agents", "--force", "--json"]
    )
    assert forced.exit_code == 0
    payload = json.loads(forced.output)
    assert payload[0]["action"] == "installed"


@pytest.mark.usefixtures("fake_skill_source")
def test_skill_install_mode_copy(fake_home: Path) -> None:
    result = runner.invoke(
        app, ["skill", "install", "--agent", "agents", "--mode", "copy", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["kind"] == "copy"
    dest = fake_home / ".agents" / "skills" / "sft"
    assert dest.is_dir()
    assert not dest.is_symlink()
    assert (dest / linking.SENTINEL_NAME).exists()


@pytest.mark.usefixtures("fake_home", "fake_skill_source")
def test_skill_install_project_scope(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "myrepo"
    project.mkdir()
    monkeypatch.chdir(project)

    result = runner.invoke(
        app,
        ["skill", "install", "--agent", "agents", "--scope", "project", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert (project / ".agents" / "skills" / "sft" / "SKILL.md").exists()


@pytest.mark.usefixtures("fake_skill_source")
def test_skill_status_and_uninstall(fake_home: Path) -> None:
    runner.invoke(app, ["skill", "install", "--agent", "agents"])

    status = runner.invoke(app, ["skill", "status", "--json"])
    assert status.exit_code == 0
    rows = {row["agent"]: row for row in json.loads(status.output)}
    assert rows["agents"]["action"] == "present"
    assert rows["claude"]["action"] == "absent"

    removed = runner.invoke(app, ["skill", "uninstall", "--agent", "agents", "--json"])
    assert removed.exit_code == 0
    payload = json.loads(removed.output)
    assert payload[0]["action"] == "removed"

    # Sibling agent dirs (if any pre-existing files) should be untouched.
    assert not (fake_home / ".agents" / "skills" / "sft").exists()


@pytest.mark.usefixtures("fake_skill_source")
def test_skill_install_dry_run_does_not_write(fake_home: Path) -> None:
    result = runner.invoke(
        app, ["skill", "install", "--agent", "agents", "--dry-run", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["action"] == "would-install"
    assert not (fake_home / ".agents" / "skills" / "sft").exists()


def test_skill_path_prints_source(fake_skill_source: Path) -> None:
    result = runner.invoke(app, ["skill", "path"])
    assert result.exit_code == 0
    assert str(fake_skill_source) in result.output


# ---------------------------------------------------------------------------
# Sanity checks on shipped SKILL.md
# ---------------------------------------------------------------------------


def test_shipped_skill_has_required_frontmatter() -> None:
    """The real skill source (not the test fixture) must have valid frontmatter."""
    from sft.commands.skill import _resolve_source

    skill = _resolve_source() / "SKILL.md"
    content = skill.read_text()
    assert content.startswith("---\n")
    block, _, _ = content[4:].partition("\n---")
    assert "name:" in block
    assert "description:" in block
    # Trigger words required for the skill to activate on relevant prompts.
    desc = block.lower()
    assert "safetensors" in desc
    assert "lora" in desc


def test_shipped_reference_is_present() -> None:
    from sft.commands.skill import _resolve_source

    assert (_resolve_source() / "REFERENCE.md").exists()


# ---------------------------------------------------------------------------
# Per-agent install + uninstall round-trip (parametrized over AGENT_DIRS)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_skill_source", "fake_home")
@pytest.mark.parametrize("agent", list(skill_cmd.AGENT_DIRS.keys()))
def test_skill_install_uninstall_per_agent_round_trip(agent: str) -> None:
    """Install -> status -> uninstall completes cleanly for every supported agent."""
    # Install
    install = runner.invoke(app, ["skill", "install", "--agent", agent, "--json"])
    assert install.exit_code == 0, install.output
    install_payload = json.loads(install.output)
    assert install_payload[0]["agent"] == agent
    assert install_payload[0]["action"] == "installed"

    expected_dir = (
        Path(skill_cmd.AGENT_DIRS[agent][0]).expanduser() / skill_cmd.SKILL_DIRNAME
    )
    assert (expected_dir / "SKILL.md").exists()

    # Status sees it as present
    status = runner.invoke(app, ["skill", "status", "--json"])
    rows = {row["agent"]: row for row in json.loads(status.output)}
    assert rows[agent]["action"] == "present"

    # Uninstall removes it, and a second uninstall is a clean no-op.
    removed = runner.invoke(app, ["skill", "uninstall", "--agent", agent, "--json"])
    assert removed.exit_code == 0
    assert json.loads(removed.output)[0]["action"] == "removed"
    assert not (expected_dir / "SKILL.md").exists()

    second = runner.invoke(app, ["skill", "uninstall", "--agent", agent, "--json"])
    assert second.exit_code == 0
    # Idempotent: second remove reports no work.
    second_payload = json.loads(second.output)
    assert second_payload[0]["action"] in {"skip", "absent", "noop"}
