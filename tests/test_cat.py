"""Tests for the cat command — merging safetensors files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.ops.cat import cat_files

runner = CliRunner()


def _make_file(
    path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_cat_two_files(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"weight": np.ones((2, 3), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"bias": np.zeros((3,), dtype=np.float32)},
    )
    dst = tmp_path / "out.safetensors"
    result = cat_files([f1, f2], dst)

    assert result.total_tensors == 2
    assert result.total_files == 2
    assert result.duplicates == []
    assert dst.exists()

    merged = load_file(str(dst))
    assert "weight" in merged
    assert "bias" in merged
    np.testing.assert_array_equal(merged["weight"], np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(merged["bias"], np.zeros((3,), dtype=np.float32))


def test_cat_duplicates_error(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"shared": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"shared": np.zeros((2,), dtype=np.float32)},
    )
    dst = tmp_path / "out.safetensors"

    import pytest

    with pytest.raises(ValueError, match="Duplicate tensor names"):
        cat_files([f1, f2], dst)


def test_cat_allow_duplicates(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"shared": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"shared": np.zeros((2,), dtype=np.float32)},
    )
    dst = tmp_path / "out.safetensors"
    result = cat_files([f1, f2], dst, allow_duplicates=True)

    assert result.total_tensors == 1
    assert "shared" in result.duplicates

    merged = load_file(str(dst))
    np.testing.assert_array_equal(merged["shared"], np.zeros((2,), dtype=np.float32))


def test_cat_default_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"x": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"y": np.zeros((2,), dtype=np.float32)},
    )
    result = cat_files([f1, f2], dst=None)

    assert result.output_path == Path("merged.safetensors")
    assert (tmp_path / "merged.safetensors").exists()


def test_cat_explicit_output(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"x": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"y": np.zeros((2,), dtype=np.float32)},
    )
    custom = tmp_path / "custom.safetensors"
    result = cat_files([f1, f2], custom)

    assert result.output_path == custom
    assert custom.exists()


def test_cat_dry_run(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"x": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"y": np.zeros((2,), dtype=np.float32)},
    )
    dst = tmp_path / "out.safetensors"
    result = cat_files([f1, f2], dst, dry_run=True)

    assert result.total_tensors == 2
    assert result.output_path is None
    assert not dst.exists()


def test_cat_merges_metadata(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"x": np.ones((2,), dtype=np.float32)},
        metadata={"author": "alice", "version": "1"},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"y": np.zeros((2,), dtype=np.float32)},
        metadata={"version": "2", "format": "sft"},
    )
    dst = tmp_path / "out.safetensors"
    cat_files([f1, f2], dst)

    from sft.index import TensorIndex

    index = TensorIndex.from_file(dst)
    assert index.metadata["author"] == "alice"
    assert index.metadata["version"] == "2"  # later file overrides
    assert index.metadata["format"] == "sft"


def test_cat_cli_basic(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"w": np.ones((2, 2), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"v": np.zeros((3,), dtype=np.float32)},
    )
    out = tmp_path / "result.safetensors"
    result = runner.invoke(app, ["cat", str(f1), str(f2), "-o", str(out)])
    assert result.exit_code == 0
    assert "Merged 2 files" in result.output
    assert out.exists()


def test_cat_cli_duplicates_error(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"shared": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"shared": np.zeros((2,), dtype=np.float32)},
    )
    out = tmp_path / "result.safetensors"
    result = runner.invoke(app, ["cat", str(f1), str(f2), "-o", str(out)])
    assert result.exit_code == 1


def test_cat_cli_dry_run(tmp_path: Path) -> None:
    f1 = _make_file(
        tmp_path / "a.safetensors",
        {"x": np.ones((2,), dtype=np.float32)},
    )
    f2 = _make_file(
        tmp_path / "b.safetensors",
        {"y": np.zeros((2,), dtype=np.float32)},
    )
    result = runner.invoke(app, ["cat", str(f1), str(f2), "--dry-run"])
    assert result.exit_code == 0
    assert "Would merge" in result.output
