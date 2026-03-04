"""On-demand tensor loading and SVD computation for LoRA analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.lora import LoraPair

# Dtype string -> (numpy dtype, bytes per element)
_DTYPE_MAP = {
    "F32": ("float32", 4),
    "F16": ("float16", 2),
    "BF16": ("uint16", 2),  # Special handling: read as uint16, convert
    "F64": ("float64", 8),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
    "U8": ("uint8", 1),
    "U16": ("uint16", 2),
    "U32": ("uint32", 4),
    "U64": ("uint64", 8),
}


def load_tensor(
    file_path: Path,
    header_size: int,
    data_offsets: tuple[int, int],
    dtype: str,
    shape: tuple[int, ...],
):
    """Load a single tensor from a safetensors file.

    Returns a numpy ndarray.
    """

    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")

    np_dtype, _bpe = _DTYPE_MAP[dtype]
    # Data starts after 8-byte length prefix + header
    data_start = 8 + header_size + data_offsets[0]
    nbytes = data_offsets[1] - data_offsets[0]

    with open(file_path, "rb") as f:
        f.seek(data_start)
        raw = f.read(nbytes)

    arr = np.frombuffer(raw, dtype=np_dtype)

    # BF16 special handling: convert to float32
    if dtype == "BF16":
        arr = arr.astype(np.uint32) << 16
        arr = arr.view(np.float32)

    return arr.reshape(shape)


def compute_lora_svd(
    file_path: Path,
    pair: LoraPair,
    header_size: int,
    index: TensorIndex,
):
    """Compute SVD of the effective LoRA matrix (B @ A).

    Returns normalized singular values as a numpy array.
    """

    # Find tensor info for A and B
    tensor_map = {t.full_name: t for t in index.tensors}
    a_info = tensor_map[pair.a_tensor_name]
    b_info = tensor_map[pair.b_tensor_name]

    # Load tensors
    a = load_tensor(
        file_path, header_size, a_info.data_offsets, a_info.dtype, a_info.shape
    )
    b = load_tensor(
        file_path, header_size, b_info.data_offsets, b_info.dtype, b_info.shape
    )

    # Reshape to 2D if needed (e.g., conv layers)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if b.ndim > 2:
        b = b.reshape(b.shape[0], -1)

    # B@A is (out, in) which can be huge, but has rank ≤ pair.rank.
    # SVD of the full matrix is O(out * in * min(out,in)) — very slow.
    # Instead, QR-factor both thin matrices and SVD their small (rank, rank) product.
    # σ(B @ A) = σ(Rb @ Qb^T @ Qa @ Ra) = σ(Rb @ M @ Ra) where M = Qb^T @ Qa is (rank, rank).
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    try:
        if b.shape[1] == a.shape[0]:
            # B (out, rank), A (rank, in)
            qa, ra = np.linalg.qr(a.T, mode="reduced")  # qa (in, rank), ra (rank, rank)
            qb, rb = np.linalg.qr(
                b.T, mode="reduced"
            )  # qb (out, rank), rb (rank, rank)
            mid = rb.T @ (qb.T @ qa) @ ra  # (rank, rank)
        elif a.shape[1] == b.shape[0]:
            # A (out, rank), B (rank, in) — swapped convention
            qa, ra = np.linalg.qr(a, mode="reduced")
            qb, rb = np.linalg.qr(b, mode="reduced")
            mid = ra @ (qa.T @ qb) @ rb.T
        else:
            # Fallback: try B @ A^T
            qa, ra = np.linalg.qr(a, mode="reduced")
            qb, rb = np.linalg.qr(b.T, mode="reduced")
            mid = rb.T @ (qb.T @ qa) @ ra
    except (ValueError, IndexError):
        # Last resort: form the full matrix
        mid = (b @ a).astype(np.float32)

    s = np.linalg.svd(mid, compute_uv=False)
    s = s[: pair.rank]

    # Normalize to max
    max_s = s[0] if s[0] > 0 else 1.0
    return s / max_s


def compute_stable_rank(
    file_path: Path,
    pair: LoraPair,
    header_size: int,
    index: TensorIndex,
) -> float:
    """Compute stable rank (||A||_F^2 / ||A||_2^2) for a LoRA pair.

    Returns stable rank as a float. Cheaper than full SVD visualization
    since we only need singular values, not the chart.
    """
    sv = compute_lora_svd(file_path, pair, header_size, index)
    sq = sv * sv
    if sq[0] > 0:
        return float(sq.sum() / sq[0])
    return 0.0


def compute_frobenius_norms(
    file_path: Path,
    pair: LoraPair,
    header_size: int,
    index: TensorIndex,
) -> tuple[float, float]:
    """Compute Frobenius norms of A and B tensors. No SVD needed — very cheap.

    Returns (||A||_F, ||B||_F).
    """

    tensor_map = {t.full_name: t for t in index.tensors}
    a_info = tensor_map[pair.a_tensor_name]
    b_info = tensor_map[pair.b_tensor_name]

    a = load_tensor(
        file_path, header_size, a_info.data_offsets, a_info.dtype, a_info.shape
    )
    b = load_tensor(
        file_path, header_size, b_info.data_offsets, b_info.dtype, b_info.shape
    )

    return float(np.linalg.norm(a)), float(np.linalg.norm(b))
