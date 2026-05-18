"""Tests for the split command — sharding safetensors files by size."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from sft.ops.split import parse_size, split_file

# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("4GB", 4 * 1024**3),
        ("500MB", 500 * 1024**2),
        ("1024B", 1024),
        ("1KB", 1024),
        ("2gb", 2 * 1024**3),
    ],
)
def test_parse_size(input_str: str, expected: int) -> None:
    assert parse_size(input_str) == expected


def test_parse_size_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("notasize")


def test_parse_size_unknown_unit() -> None:
    with pytest.raises(ValueError, match="Unknown size unit"):
        parse_size("100TB")


# ---------------------------------------------------------------------------
# split_file — using mini_model fixture (14 tensors, small)
# ---------------------------------------------------------------------------


def test_split_creates_multiple_shards(mini_model: Path) -> None:
    """A very small --max-size forces multiple shards."""
    result = split_file(mini_model, max_bytes=500)

    assert len(result.shards) > 1
    for shard in result.shards:
        assert shard.path.exists()
        tensors = load_file(str(shard.path))
        assert len(tensors) > 0


def test_split_creates_index_json(mini_model: Path) -> None:
    result = split_file(mini_model, max_bytes=500)

    assert result.index_path is not None
    assert result.index_path.exists()

    data = json.loads(result.index_path.read_text())
    assert "metadata" in data
    assert "weight_map" in data
    assert "total_size" in data["metadata"]
    assert isinstance(data["weight_map"], dict)
    assert len(data["weight_map"]) > 0


def test_split_all_tensors_accounted(mini_model: Path) -> None:
    """Every tensor from the original file appears in exactly one shard."""
    original = load_file(str(mini_model))
    original_names = set(original.keys())

    result = split_file(mini_model, max_bytes=500)

    shard_names: list[str] = []
    for shard in result.shards:
        shard_names.extend(shard.tensor_names)

    assert set(shard_names) == original_names
    assert len(shard_names) == len(original_names)  # no duplicates


def test_split_dry_run(mini_model: Path) -> None:
    """--dry-run doesn't write any files."""
    result = split_file(mini_model, max_bytes=500, dry_run=True)

    assert result.index_path is None
    assert len(result.shards) > 1
    for shard in result.shards:
        assert not shard.path.exists()


def test_split_single_shard(mini_model: Path) -> None:
    """A huge --max-size produces a single shard."""
    result = split_file(mini_model, max_bytes=1024**3)

    assert len(result.shards) == 1
    assert result.shards[0].path.exists()
    tensors = load_file(str(result.shards[0].path))
    original = load_file(str(mini_model))
    assert set(tensors.keys()) == set(original.keys())


# ---------------------------------------------------------------------------
# Controlled-size fixture for deterministic shard boundaries
# ---------------------------------------------------------------------------


@pytest.fixture
def three_tensor_model(tmp_path: Path) -> Path:
    """Three tensors of known sizes for predictable sharding."""
    tensors = {
        "a": np.zeros((10,), dtype=np.float32),  # 40 bytes
        "b": np.zeros((20,), dtype=np.float32),  # 80 bytes
        "c": np.zeros((10,), dtype=np.float32),  # 40 bytes
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path


def test_split_deterministic_sharding(three_tensor_model: Path) -> None:
    """With max_bytes=120, 'a'(40)+b(80)=120 fits, then c(40) starts a new shard."""
    result = split_file(three_tensor_model, max_bytes=120)

    assert len(result.shards) == 2
    assert result.shards[0].tensor_names == ["a", "b"]
    assert result.shards[1].tensor_names == ["c"]


def test_split_oversized_tensor(tmp_path: Path) -> None:
    """A tensor larger than max_bytes gets its own shard."""
    tensors = {
        "small": np.zeros((5,), dtype=np.float32),  # 20 bytes
        "big": np.zeros((100,), dtype=np.float32),  # 400 bytes
        "small2": np.zeros((5,), dtype=np.float32),  # 20 bytes
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    result = split_file(path, max_bytes=50)

    big_shard = [s for s in result.shards if "big" in s.tensor_names]
    assert len(big_shard) == 1
    assert big_shard[0].tensor_names == ["big"]


def test_split_index_weight_map_filenames(three_tensor_model: Path) -> None:
    """weight_map values are shard filenames (not full paths)."""
    result = split_file(three_tensor_model, max_bytes=100)
    assert result.index_path is not None

    data = json.loads(result.index_path.read_text())
    for _tensor_name, filename in data["weight_map"].items():
        assert "/" not in filename
        assert filename.endswith(".safetensors")


def test_split_roundtrip_data(mini_model: Path) -> None:
    """Tensor data is preserved through split."""
    original = load_file(str(mini_model))
    result = split_file(mini_model, max_bytes=500)

    reconstructed: dict[str, np.ndarray] = {}
    for shard in result.shards:
        reconstructed.update(load_file(str(shard.path)))

    for name in original:
        np.testing.assert_array_equal(original[name], reconstructed[name])
