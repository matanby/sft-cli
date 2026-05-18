"""Dtype mapping between CLI names, numpy dtypes, and safetensors internal names."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

# CLI name → numpy dtype
CLI_TO_NUMPY: dict[str, np.dtype] = {
    "fp32": np.dtype("float32"),
    "float32": np.dtype("float32"),
    "fp16": np.dtype("float16"),
    "float16": np.dtype("float16"),
    "bf16": np.dtype(bfloat16),
    "bfloat16": np.dtype(bfloat16),
    "fp64": np.dtype("float64"),
    "float64": np.dtype("float64"),
}

# Safetensors internal name → numpy dtype
SAFETENSORS_TO_NUMPY: dict[str, np.dtype] = {
    "BF16": np.dtype(bfloat16),
    "F16": np.dtype("float16"),
    "F32": np.dtype("float32"),
    "F64": np.dtype("float64"),
    "I8": np.dtype("int8"),
    "I16": np.dtype("int16"),
    "I32": np.dtype("int32"),
    "I64": np.dtype("int64"),
    "U8": np.dtype("uint8"),
}

VALID_DTYPES = sorted({k for k in CLI_TO_NUMPY if CLI_TO_NUMPY[k] is not None})


def resolve_dtype(name: str) -> np.dtype:
    """Resolve a CLI dtype name to a numpy dtype. Raises ValueError if invalid."""
    dtype = CLI_TO_NUMPY.get(name.lower())
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype: '{name}'. Valid options: {', '.join(VALID_DTYPES)}"
        )
    return dtype


def cast_tensor(tensor: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Cast a tensor to a target dtype."""
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.astype(target_dtype)
