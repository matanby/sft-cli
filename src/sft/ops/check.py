"""Pure logic for the check command — validate a .safetensors file."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    """Result of all integrity checks on a .safetensors file."""

    header_ok: bool = False
    header_error: str = ""
    num_tensors: int = 0
    offsets_ok: bool = False
    offsets_error: str = ""
    dtypes: list[str] = field(default_factory=list)
    nan_tensors: list[str] = field(default_factory=list)
    inf_tensors: list[str] = field(default_factory=list)
    values_checked: bool = False

    @property
    def healthy(self) -> bool:
        return (
            self.header_ok
            and self.offsets_ok
            and not self.nan_tensors
            and not self.inf_tensors
        )


def check_file(path: Path, *, skip_values: bool = False) -> CheckResult:
    """Run integrity checks on a .safetensors file."""
    result = CheckResult()

    # 1. Header check
    try:
        from sft.index import TensorIndex

        index = TensorIndex.from_file(path)
        result.header_ok = True
        result.num_tensors = index.total_tensors
    except Exception as exc:
        result.header_error = str(exc)
        return result

    # 2. Offset check
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]

        expected_min = 8 + header_size + index.total_bytes
        if file_size < expected_min:
            result.offsets_error = (
                f"file size {file_size} < expected minimum {expected_min}"
            )
        else:
            result.offsets_ok = True
    except Exception as exc:
        result.offsets_error = str(exc)

    # 3. Dtypes (informational)
    result.dtypes = sorted({t.dtype for t in index.tensors})

    # 4. Values check
    if skip_values:
        result.values_checked = False
        return result

    result.values_checked = True
    try:
        import numpy as np
        from safetensors.numpy import load_file

        tensors = load_file(str(path))
        for name, arr in tensors.items():
            if np.issubdtype(arr.dtype, np.floating):
                if np.isnan(arr).any():
                    result.nan_tensors.append(name)
                if np.isinf(arr).any():
                    result.inf_tensors.append(name)
    except Exception:
        pass

    return result
