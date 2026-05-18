"""Streaming tensor read/write helpers wrapping safetensors.numpy."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Callable

import numpy as np
from ml_dtypes import bfloat16
from safetensors.numpy import load_file, save_file

from sft.index import TensorIndex

_SAFETENSORS_DTYPE_TO_NUMPY = {
    "BF16": np.dtype(bfloat16),
    "F16": np.dtype("float16"),
    "F32": np.dtype("float32"),
    "F64": np.dtype("float64"),
    "I8": np.dtype("int8"),
    "I16": np.dtype("int16"),
    "I32": np.dtype("int32"),
    "I64": np.dtype("int64"),
    "U8": np.dtype("uint8"),
    "U16": np.dtype("uint16"),
    "U32": np.dtype("uint32"),
    "U64": np.dtype("uint64"),
    "BOOL": np.dtype("bool"),
}


def read_tensors(path: Path) -> dict[str, np.ndarray]:
    """Load all tensors from a safetensors file.

    safetensors>=0.4 with ml_dtypes handles BF16 natively. Falls back to
    manual bit-level conversion if the binding still raises TypeError.
    """
    try:
        return load_file(str(path))
    except TypeError:
        return _load_with_bf16_support(path)


def read_tensor(path: Path, name: str) -> np.ndarray:
    """Load a single tensor by name."""
    tensors = read_tensors(path)
    return tensors[name]


def write_file(
    path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write tensors to a safetensors file."""
    save_file(tensors, str(path), metadata=metadata or {})


def copy_with_transform(
    src: Path,
    dst: Path,
    transform: Callable[[str, np.ndarray], np.ndarray | None],
    include_names: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    """Read src, apply transform per tensor, write to dst.

    The transform function receives (name, tensor) and returns:
    - A transformed tensor (possibly different dtype/shape)
    - None to exclude the tensor from the output

    If metadata is None, copies the source file's metadata.
    """
    index = TensorIndex.from_file(src)
    src_tensors = read_tensors(src)

    if metadata is None:
        metadata = index.metadata

    result: dict[str, np.ndarray] = {}
    names = include_names if include_names is not None else list(src_tensors.keys())

    for name in names:
        if name not in src_tensors:
            continue
        transformed = transform(name, src_tensors[name])
        if transformed is not None:
            result[name] = transformed

    save_file(result, str(dst), metadata=metadata)


def iter_tensor_names(path: Path) -> list[str]:
    """Get tensor names from a file (header-only, no data loaded)."""
    index = TensorIndex.from_file(path)
    return [t.full_name for t in index.tensors]


def _load_with_bf16_support(path: Path) -> dict[str, np.ndarray]:
    """Load tensors when numpy can't handle bf16.

    Tries PyTorch first (native bf16 support), then falls back to
    manual bit-level conversion as a last resort.
    """
    try:
        return _load_via_torch(path)
    except ImportError:
        return _load_with_manual_bf16(path)


def _load_via_torch(path: Path) -> dict[str, np.ndarray]:
    """Load via safetensors.torch, convert to numpy arrays.

    PyTorch handles bfloat16 natively. bf16 tensors are converted to
    float32 for numpy compatibility; all other dtypes pass through.
    """
    import torch
    from safetensors.torch import load_file as torch_load_file

    torch_tensors = torch_load_file(str(path))
    result: dict[str, np.ndarray] = {}
    for name, tensor in torch_tensors.items():
        if str(tensor.dtype) == "torch.bfloat16":
            result[name] = tensor.view(torch.int16).numpy().view(np.dtype(bfloat16))
        else:
            result[name] = tensor.numpy()
    return result


def _load_with_manual_bf16(path: Path) -> dict[str, np.ndarray]:
    """Last-resort loader: parse raw bytes, convert bf16 → fp32 manually."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        data_start = 8 + header_size

    header = json.loads(header_bytes.decode("utf-8"))

    result: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            if name == "__metadata__":
                continue

            dtype_str = info["dtype"]
            shape = tuple(info["shape"])
            begin, end = info["data_offsets"]

            f.seek(data_start + begin)
            raw = f.read(end - begin)

            if dtype_str == "BF16":
                result[name] = np.frombuffer(raw, dtype=np.dtype(bfloat16)).reshape(
                    shape
                )
            else:
                numpy_dtype = _SAFETENSORS_DTYPE_TO_NUMPY.get(dtype_str)
                if numpy_dtype is None:
                    raise ValueError(
                        f"Unsupported dtype '{dtype_str}' for tensor '{name}'"
                    )
                result[name] = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)

    return result
