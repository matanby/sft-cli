"""Human-readable formatting utilities."""

from __future__ import annotations

_DTYPE_DISPLAY = {
    "F16": "FP16",
    "F32": "FP32",
    "F64": "FP64",
    "BF16": "BF16",
    "F8_E4M3": "FP8_E4M3",
    "F8_E5M2": "FP8_E5M2",
    "I8": "INT8",
    "I16": "INT16",
    "I32": "INT32",
    "I64": "INT64",
    "U8": "UINT8",
    "U16": "UINT16",
    "U32": "UINT32",
    "U64": "UINT64",
    "BOOL": "BOOL",
}


def format_bytes(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    else:
        return f"{nbytes / 1024 / 1024 / 1024:.2f} GB"


def format_number(n: int) -> str:
    """Format large numbers as human-readable (e.g. 6.7B)."""
    if n < 1_000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    elif n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    else:
        return f"{n / 1_000_000_000:.1f}B"


def format_shape(shape: tuple[int, ...]) -> str:
    """Format tensor shape as string."""
    if len(shape) == 0:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return f"({', '.join(str(d) for d in shape)})"


def format_dtype(dtype: str) -> str:
    """Map internal safetensors dtype names to human-readable form."""
    return _DTYPE_DISPLAY.get(dtype, dtype)
