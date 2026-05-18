"""Tests for the convert command — PyTorch to safetensors conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from safetensors.numpy import load_file
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from sft.cli import app  # noqa: E402
from sft.ops.convert import convert_to_safetensors  # noqa: E402

runner = CliRunner()


def _make_pt(path: Path, state_dict: dict) -> Path:
    torch.save(state_dict, path)
    return path


def test_convert_basic(tmp_path: Path) -> None:
    src = _make_pt(
        tmp_path / "model.pt",
        {"weight": torch.randn(4, 4), "bias": torch.randn(4)},
    )
    dst = tmp_path / "model.safetensors"
    result = convert_to_safetensors(src, dst)

    assert result.tensors_count == 2
    assert result.output_path == dst
    assert dst.exists()

    tensors = load_file(str(dst))
    assert "weight" in tensors
    assert "bias" in tensors
    assert tensors["weight"].shape == (4, 4)
    assert tensors["bias"].shape == (4,)


def test_convert_smart_output(tmp_path: Path) -> None:
    src = _make_pt(
        tmp_path / "checkpoint.pt",
        {"weight": torch.randn(2, 2)},
    )
    result = runner.invoke(app, ["convert", str(src)])
    assert result.exit_code == 0

    expected = tmp_path / "checkpoint.safetensors"
    assert expected.exists()
    tensors = load_file(str(expected))
    assert "weight" in tensors


def test_convert_explicit_output(tmp_path: Path) -> None:
    src = _make_pt(
        tmp_path / "model.bin",
        {"weight": torch.randn(3, 3)},
    )
    out = tmp_path / "custom.safetensors"
    result = runner.invoke(app, ["convert", str(src), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()

    tensors = load_file(str(out))
    assert "weight" in tensors
    assert tensors["weight"].shape == (3, 3)


def test_convert_with_dtype(tmp_path: Path) -> None:
    src = _make_pt(
        tmp_path / "model.pt",
        {"weight": torch.randn(4, 4)},  # float32 by default
    )
    dst = tmp_path / "model.safetensors"
    result = convert_to_safetensors(src, dst, dtype="fp16")

    assert result.tensors_count == 1
    tensors = load_file(str(dst))
    assert tensors["weight"].dtype == np.float16


def test_convert_with_dtype_cli(tmp_path: Path) -> None:
    src = _make_pt(
        tmp_path / "model.pt",
        {"weight": torch.randn(4, 4)},
    )
    result = runner.invoke(app, ["convert", str(src), "--dtype", "fp16"])
    assert result.exit_code == 0
    assert "cast to fp16" in result.output

    tensors = load_file(str(tmp_path / "model.safetensors"))
    assert tensors["weight"].dtype == np.float16


def test_convert_state_dict_wrapper(tmp_path: Path) -> None:
    inner = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
    src = _make_pt(
        tmp_path / "wrapped.pt",
        {"state_dict": inner, "epoch": 10},
    )
    dst = tmp_path / "wrapped.safetensors"
    result = convert_to_safetensors(src, dst)

    assert result.tensors_count == 2
    tensors = load_file(str(dst))
    assert "weight" in tensors
    assert "bias" in tensors


def test_convert_missing_torch(tmp_path: Path) -> None:
    dummy = tmp_path / "dummy.pt"
    dummy.write_bytes(b"fake")

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = runner.invoke(app, ["convert", str(dummy)])

    assert result.exit_code == 1
    assert "torch is required for conversion" in result.output
