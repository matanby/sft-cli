"""Tests for slice and strip operations."""

from __future__ import annotations

from pathlib import Path

from safetensors.numpy import load_file

from sft.ops.slice import slice_file, strip_file


def test_slice_include(mini_model: Path, tmp_path: Path) -> None:
    """--include 'model.layers.0.**' keeps only layer 0 tensors."""
    dst = tmp_path / "out.safetensors"
    result = slice_file(mini_model, dst, include="model.layers.0.**", exclude=None)

    assert result.output_path == dst
    assert len(result.included) == 6
    assert all("model.layers.0." in n for n in result.included)
    assert len(result.excluded) == 8

    tensors = load_file(str(dst))
    assert len(tensors) == 6
    assert all("model.layers.0." in n for n in tensors)


def test_slice_exclude(mini_model: Path, tmp_path: Path) -> None:
    """--exclude '**.inv_freq' removes rotary embedding tensors."""
    dst = tmp_path / "out.safetensors"
    result = slice_file(mini_model, dst, include=None, exclude="**.inv_freq")

    assert result.output_path == dst
    assert len(result.included) == 13
    assert "model.layers.0.rotary_emb.inv_freq" not in result.included
    assert "model.layers.0.rotary_emb.inv_freq" in result.excluded

    tensors = load_file(str(dst))
    assert len(tensors) == 13
    assert "model.layers.0.rotary_emb.inv_freq" not in tensors


def test_slice_smart_output(mini_model: Path) -> None:
    """No -o generates {stem}.sliced.safetensors."""
    result = slice_file(mini_model, dst=None, include="model.layers.0.**", exclude=None)

    expected = mini_model.parent / f"{mini_model.stem}.sliced.safetensors"
    assert result.output_path == expected
    assert expected.exists()

    tensors = load_file(str(expected))
    assert len(tensors) == 6


def test_slice_dry_run(mini_model: Path) -> None:
    """--dry-run doesn't write a file."""
    result = slice_file(
        mini_model, dst=None, include="model.layers.0.**", exclude=None, dry_run=True
    )

    assert result.output_path is None
    assert len(result.included) == 6
    assert len(result.excluded) == 8

    expected = mini_model.parent / f"{mini_model.stem}.sliced.safetensors"
    assert not expected.exists()


def test_strip_exclude(mini_model: Path, tmp_path: Path) -> None:
    """--exclude '**.inv_freq' strips matching tensors."""
    dst = tmp_path / "out.safetensors"
    result = strip_file(mini_model, dst, exclude="**.inv_freq")

    assert result.output_path == dst
    assert len(result.included) == 13
    assert "model.layers.0.rotary_emb.inv_freq" not in result.included
    assert "model.layers.0.rotary_emb.inv_freq" in result.excluded

    tensors = load_file(str(dst))
    assert len(tensors) == 13


def test_strip_smart_output(mini_model: Path) -> None:
    """No -o generates {stem}.stripped.safetensors."""
    result = strip_file(mini_model, dst=None, exclude="**.inv_freq")

    expected = mini_model.parent / f"{mini_model.stem}.stripped.safetensors"
    assert result.output_path == expected
    assert expected.exists()

    tensors = load_file(str(expected))
    assert len(tensors) == 13


def test_roundtrip(mini_model: Path, tmp_path: Path) -> None:
    """Slice then verify tensor count is correct."""
    dst = tmp_path / "sliced.safetensors"
    result = slice_file(mini_model, dst, include="model.layers.**", exclude=None)

    original = load_file(str(mini_model))
    sliced = load_file(str(dst))

    layer_tensors = [n for n in original if n.startswith("model.layers.")]
    assert len(sliced) == len(layer_tensors)
    assert len(result.included) == len(layer_tensors)
    assert len(result.included) + len(result.excluded) == len(original)
