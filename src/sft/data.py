"""On-demand tensor loading, SVD computation, and safetensors I/O for LoRA analysis."""

from __future__ import annotations

import json
import struct
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

    Returns (normalized_sv, sigma0) where sigma0 is the raw first singular value.
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
    # B = Qb @ Rb, A = Ra.T @ Qa.T  →  B@A = Qb @ (Rb @ Ra.T) @ Qa.T
    # σ(B@A) = σ(Rb @ Ra.T) since Qb, Qa are orthonormal.
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if b.shape[1] == a.shape[0]:
        # B (out, rank), A (rank, in)
        qa, ra = np.linalg.qr(a.T, mode="reduced")  # qa (in, rank), ra (rank, rank)
        qb, rb = np.linalg.qr(b, mode="reduced")    # qb (out, rank), rb (rank, rank)
        mid = rb @ ra.T  # (rank, rank)
    elif a.shape[1] == b.shape[0]:
        # A (out, rank), B (rank, in) — swapped convention
        qa, ra = np.linalg.qr(a, mode="reduced")
        qb, rb = np.linalg.qr(b.T, mode="reduced")
        mid = ra @ rb.T
    else:
        # Fallback: form the full matrix
        mid = (b @ a).astype(np.float32)

    s = np.linalg.svd(mid, compute_uv=False)
    s = s[: pair.rank]

    # Normalize to max
    sigma0 = float(s[0])
    max_s = s[0] if s[0] > 0 else 1.0
    return s / max_s, sigma0


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
    sv, sigma0 = compute_lora_svd(file_path, pair, header_size, index)
    sq = sv * sv
    stable_rank = float(sq.sum() / sq[0]) if sq[0] > 0 else 0.0
    return stable_rank, sigma0


def compute_frobenius_norms(
    file_path: Path,
    pair: LoraPair,
    header_size: int,
    index: TensorIndex,
) -> dict[str, float]:
    """Compute element-level stats for A and B tensors. Very cheap.

    Returns dict with norm, min, max, mean, median for each tensor.
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

    return {
        "norm_a": float(np.linalg.norm(a)),
        "norm_b": float(np.linalg.norm(b)),
        "min_a": float(np.min(a)),
        "max_a": float(np.max(a)),
        "mean_a": float(np.mean(a)),
        "median_a": float(np.median(a)),
        "min_b": float(np.min(b)),
        "max_b": float(np.max(b)),
        "mean_b": float(np.mean(b)),
        "median_b": float(np.median(b)),
    }


def truncate_lora_pair(
    a: np.ndarray,
    b: np.ndarray,
    target_rank: int,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Truncate a LoRA A/B pair to a lower rank via SVD.

    Args:
        a: A tensor, shape (rank, in_features) or higher-dim for conv layers.
        b: B tensor, shape (out_features, rank) or higher-dim for conv layers.
        target_rank: Desired output rank.
        dtype: Original dtype string (e.g. "BF16", "F16", "F32").

    Returns:
        (a_new, b_new, energy_retained) where energy_retained is
        sum(s[:k]^2) / sum(s^2), i.e. fraction of Frobenius energy kept.
    """
    a_orig_shape = a.shape
    b_orig_shape = b.shape

    # Reshape to 2D if needed (conv layers)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if b.ndim > 2:
        b = b.reshape(b.shape[0], -1)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # B (out, rank) @ A (rank, in) — use QR trick for efficient SVD
    # QR(A.T) where A.T is (in, rank): Qa (in, rank), Ra (rank, rank), so A = Ra.T @ Qa.T
    # QR(B)   where B is (out, rank):  Qb (out, rank), Rb (rank, rank), so B = Qb @ Rb
    # B @ A = Qb @ Rb @ Ra.T @ Qa.T = Qb @ (mid) @ Qa.T
    qa, ra = np.linalg.qr(a.T, mode="reduced")  # qa (in, rank), ra (rank, rank)
    qb, rb = np.linalg.qr(b, mode="reduced")    # qb (out, rank), rb (rank, rank)
    mid = rb @ ra.T  # (rank, rank)

    um, s, vtm = np.linalg.svd(mid, full_matrices=False)

    # Energy retained: fraction of squared Frobenius norm kept
    s_sq = s * s
    total_energy = s_sq.sum()
    kept_energy = s_sq[:target_rank].sum()
    energy_retained = float(kept_energy / total_energy) if total_energy > 0 else 1.0

    # Truncate to target rank
    k = target_rank
    s_k = s[:k]
    um_k = um[:, :k]    # (rank, k)
    vtm_k = vtm[:k, :]  # (k, rank)

    sqrt_s = np.sqrt(s_k)

    # B@A = Qb @ Um @ diag(S) @ Vtm @ Qa.T
    # Truncated: A_new = diag(sqrt_s) @ Vtm_k @ Qa.T  -> (k, in)
    #            B_new = Qb @ Um_k @ diag(sqrt_s)      -> (out, k)
    a_new = (np.diag(sqrt_s) @ vtm_k) @ qa.T  # (k, in)
    b_new = qb @ (um_k @ np.diag(sqrt_s))     # (out, k)

    # Reshape back if conv layers
    if len(a_orig_shape) > 2:
        a_new = a_new.reshape(k, *a_orig_shape[1:])
    if len(b_orig_shape) > 2:
        b_new = b_new.reshape(b_orig_shape[0], k, *b_orig_shape[2:])

    # Cast back to original dtype
    a_new = _cast_to_dtype(a_new, dtype)
    b_new = _cast_to_dtype(b_new, dtype)

    return a_new, b_new, energy_retained


def _cast_to_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Cast a float32 array back to the target safetensors dtype."""
    if dtype == "BF16":
        # float32 -> bf16 stored as uint16
        return (arr.view(np.uint32) >> 16).astype(np.uint16)
    elif dtype == "F16":
        return arr.astype(np.float16)
    elif dtype == "F32":
        return arr
    elif dtype == "F64":
        return arr.astype(np.float64)
    else:
        return arr


def rewrite_with_metadata(
    file_path: Path,
    index: TensorIndex,
    metadata: dict[str, str],
) -> None:
    """Rewrite a safetensors file in-place with updated metadata.

    Loads all tensors, then writes back with the new metadata dict.
    """
    all_tensors = load_all_tensors(file_path, index)
    tensor_order = [t.full_name for t in index.tensors]
    write_safetensors(file_path, all_tensors, tensor_order, metadata or None)


def scale_lora_a(
    file_path: Path,
    index: TensorIndex,
    pairs: list,
    alpha: float,
) -> dict[str, tuple[np.ndarray, str]]:
    """Scale all lora_A tensors by alpha, returning modified tensor dict.

    delta_W = B @ (alpha * A) = alpha * (B @ A), so scaling A scales the
    entire LoRA contribution uniformly.

    Returns dict of name -> (ndarray, dtype_string) ready for write_safetensors.
    """
    all_tensors = load_all_tensors(file_path, index)

    for pair in pairs:
        a_name = pair.a_tensor_name
        arr, dtype_str = all_tensors[a_name]

        # Convert to float32 for scaling, then cast back
        if dtype_str == "BF16":
            f32 = arr.astype(np.uint32) << 16
            f32 = f32.view(np.float32)
        elif dtype_str == "F16":
            f32 = arr.astype(np.float32)
        else:
            f32 = arr.astype(np.float32)

        f32 = f32 * alpha
        all_tensors[a_name] = (_cast_to_dtype(f32, dtype_str), dtype_str)

    return all_tensors


def load_all_tensors(
    file_path: Path, index: TensorIndex,
) -> dict[str, tuple[np.ndarray, str]]:
    """Load all tensors from a safetensors file as raw arrays (no BF16 conversion).

    Returns dict of name -> (raw_ndarray, dtype_string).
    """
    result = {}
    for t in index.tensors:
        np_dtype, _bpe = _DTYPE_MAP[t.dtype]
        data_start = 8 + index.header_size + t.data_offsets[0]
        nbytes = t.data_offsets[1] - t.data_offsets[0]

        with open(file_path, "rb") as f:
            f.seek(data_start)
            raw = f.read(nbytes)

        arr = np.frombuffer(raw, dtype=np_dtype).copy()
        arr = arr.reshape(t.shape)
        result[t.full_name] = (arr, t.dtype)
    return result


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[np.ndarray, str]],
    tensor_order: list[str],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a safetensors file.

    Args:
        path: Output file path.
        tensors: Dict of name -> (ndarray, dtype_string). Arrays must already
                 be in the correct numpy dtype for storage (e.g. uint16 for BF16).
        tensor_order: Order in which to write tensors.
        metadata: Optional __metadata__ dict.
    """
    # Build header
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for name in tensor_order:
        arr, dtype_str = tensors[name]
        nbytes = arr.nbytes
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    # Serialize header
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad to 8-byte alignment
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    with open(path, "wb") as f:
        # 8-byte header size
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # Tensor data in order
        for name in tensor_order:
            arr, _ = tensors[name]
            f.write(arr.tobytes())
