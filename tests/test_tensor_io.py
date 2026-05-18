"""Tests for tensor I/O and dtype utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from sft.utils.dtypes import cast_tensor, resolve_dtype
from sft.utils.tensor_io import (
    copy_with_transform,
    iter_tensor_names,
    read_tensor,
    read_tensors,
    write_file,
)


class TestReadWrite:
    def test_roundtrip(self, mini_model: Path):
        tensors = read_tensors(mini_model)
        assert len(tensors) == 14
        assert "model.norm.weight" in tensors

    def test_read_single_tensor(self, mini_model: Path):
        t = read_tensor(mini_model, "model.norm.weight")
        assert t.shape == (8,)

    def test_write_and_read_back(self, tmp_path: Path):
        path = tmp_path / "test.safetensors"
        tensors = {"a": np.zeros((4, 4), dtype=np.float32)}
        write_file(path, tensors, metadata={"key": "val"})

        loaded = load_file(str(path))
        assert "a" in loaded
        np.testing.assert_array_equal(loaded["a"], tensors["a"])

    def test_iter_tensor_names(self, mini_model: Path):
        names = iter_tensor_names(mini_model)
        assert len(names) == 14
        assert "lm_head.weight" in names


class TestCopyWithTransform:
    def test_identity_transform(self, mini_model: Path, tmp_path: Path):
        dst = tmp_path / "copy.safetensors"
        copy_with_transform(mini_model, dst, transform=lambda _name, t: t)
        original = read_tensors(mini_model)
        copied = read_tensors(dst)
        assert set(original.keys()) == set(copied.keys())

    def test_filter_transform(self, mini_model: Path, tmp_path: Path):
        dst = tmp_path / "filtered.safetensors"
        copy_with_transform(
            mini_model,
            dst,
            transform=lambda name, t: t if "norm" in name else None,
        )
        result = read_tensors(dst)
        assert all("norm" in k for k in result)

    def test_dtype_transform(self, tmp_path: Path):
        src = tmp_path / "src.safetensors"
        write_file(src, {"w": np.ones((4, 4), dtype=np.float32)})

        dst = tmp_path / "dst.safetensors"
        copy_with_transform(src, dst, transform=lambda _name, t: t.astype(np.float16))
        result = read_tensors(dst)
        assert result["w"].dtype == np.float16


class TestDtypes:
    def test_resolve_fp32(self):
        assert resolve_dtype("fp32") == np.dtype("float32")

    def test_resolve_fp16(self):
        assert resolve_dtype("fp16") == np.dtype("float16")

    def test_resolve_case_insensitive(self):
        assert resolve_dtype("FP32") == np.dtype("float32")

    def test_resolve_invalid(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_dtype("invalid")

    def test_cast_tensor(self):
        t = np.ones((4, 4), dtype=np.float32)
        result = cast_tensor(t, np.dtype("float16"))
        assert result.dtype == np.float16

    def test_cast_noop(self):
        t = np.ones((4, 4), dtype=np.float32)
        result = cast_tensor(t, np.dtype("float32"))
        assert result is t  # same object, no copy
