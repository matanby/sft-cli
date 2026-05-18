"""Reduce LoRA rank via truncated SVD.

Uses the same QR-accelerated SVD as svd.py — avoids forming the full
B@A product matrix, keeping cost at O(rank^3) instead of O(out*in*rank).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.ops.lora.detect import detect_lora
from sft.ops.lora.svd import _qr_svd
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class ResizeResult:
    original_rank: int
    new_rank: int
    modules_resized: int
    errors: dict[str, float] = field(default_factory=dict)
    output_path: Path | None = None


def resize_lora(
    src: Path,
    dst: Path | None,
    target_rank: int,
    dry_run: bool = False,
) -> ResizeResult:
    """Resize a LoRA adapter to a lower rank using truncated SVD.

    For each A/B pair, decomposes B@A via QR-accelerated SVD and keeps
    only the top `target_rank` singular values/vectors.  The sqrt of each
    kept singular value is split evenly between the new A and B factors.
    """
    info = detect_lora(src)
    if info is None:
        raise ValueError(f"Not a LoRA adapter: {src.name}")

    if target_rank >= info.rank:
        raise ValueError(
            f"Target rank {target_rank} must be less than current rank {info.rank}"
        )
    if target_rank < 1:
        raise ValueError(f"Target rank must be >= 1, got {target_rank}")

    tensors = read_tensors(src)
    new_tensors: dict[str, np.ndarray] = {}
    errors: dict[str, float] = {}

    for name in tensors:
        if not any(name == p.lora_a_name or name == p.lora_b_name for p in info.pairs):
            new_tensors[name] = tensors[name]

    for pair in info.pairs:
        a = tensors[pair.lora_a_name]
        b = tensors[pair.lora_b_name]
        orig_dtype = a.dtype

        u, s, vt = _qr_svd(a, b, compute_uv=True)

        r = target_rank
        sqrt_s = np.sqrt(s[:r])
        a_new = (sqrt_s[:, np.newaxis] * vt[:r, :]).astype(orig_dtype)
        b_new = (u[:, :r] * sqrt_s[np.newaxis, :]).astype(orig_dtype)

        s_sq = s**2
        total_energy = s_sq.sum()
        if total_energy > 0:
            error = float(np.sqrt(1.0 - s_sq[:r].sum() / total_energy))
        else:
            error = 0.0

        errors[pair.target_module] = error
        new_tensors[pair.lora_a_name] = a_new
        new_tensors[pair.lora_b_name] = b_new

    result = ResizeResult(
        original_rank=info.rank,
        new_rank=target_rank,
        modules_resized=len(info.pairs),
        errors=errors,
        output_path=dst,
    )

    if not dry_run and dst is not None:
        metadata = dict(info.metadata)
        metadata["rank"] = str(target_rank)
        write_file(dst, new_tensors, metadata=metadata)

    return result
