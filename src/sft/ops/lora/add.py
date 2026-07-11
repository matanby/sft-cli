"""Weighted linear combination of LoRA adapters (task arithmetic).

Reconstructs each LoRA's delta (B @ A), computes a weighted sum,
then re-decomposes via SVD into new A/B pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.ops.lora.conflict import (
    gram_schmidt_orthogonalize,
    norm_scale_factor,
    validate_mode,
)
from sft.ops.lora.detect import LoRAInfo, detect_lora
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class AddResult:
    combined_modules: int
    output_rank: int
    output_path: Path | None


def _validate_compatibility(infos: list[LoRAInfo]) -> None:
    """Ensure all LoRAs target the same modules."""
    reference_keys = sorted(p.module_key for p in infos[0].pairs)
    for i, info in enumerate(infos[1:], start=1):
        keys = sorted(p.module_key for p in info.pairs)
        if keys != reference_keys:
            raise ValueError(
                f"LoRA {i} has incompatible modules: "
                f"expected {reference_keys}, got {keys}"
            )


def _decompose_svd(delta: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """SVD-decompose a delta matrix into LoRA A and B at the given rank.

    Returns (A, B) where A has shape (rank, in_features)
    and B has shape (out_features, rank).
    """
    u, s, vt = np.linalg.svd(delta, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]
    sqrt_s = np.sqrt(s_r)
    b_new = u_r * sqrt_s[np.newaxis, :]
    a_new = sqrt_s[:, np.newaxis] * vt_r
    return a_new.astype(np.float32), b_new.astype(np.float32)


def add_loras(
    lora_paths: list[Path],
    weights: list[float] | None,
    dst: Path | None = None,
    output_rank: int | None = None,
    dry_run: bool = False,
    mode: str = "none",
) -> AddResult:
    """Combine multiple LoRA adapters via weighted task arithmetic.

    1. Detect and validate each LoRA
    2. Reconstruct per-module deltas (B @ A)
    3. Optionally transform each non-reference delta against the first
       (reference) adapter's delta: ``norm-scaler`` rescales to the reference's
       Frobenius norm, ``gram-schmidt`` removes the reference-aligned component
    4. Compute the weighted sum
    5. SVD re-decompose at target rank
    6. Write combined adapter
    """
    validate_mode(mode)

    if len(lora_paths) < 2:
        raise ValueError("Need at least 2 LoRA files to add")

    infos: list[LoRAInfo] = []
    for p in lora_paths:
        info = detect_lora(p)
        if info is None:
            raise ValueError(f"No LoRA pairs found in {p}")
        infos.append(info)

    _validate_compatibility(infos)

    if weights is None:
        n = len(lora_paths)
        weights = [1.0 / n] * n
    elif len(weights) != len(lora_paths):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of LoRA files ({len(lora_paths)})"
        )

    rank = output_rank if output_rank is not None else infos[0].rank
    output_path = dst if dst is not None else Path("combined.safetensors")

    if dry_run:
        return AddResult(
            combined_modules=len(infos[0].pairs),
            output_rank=rank,
            output_path=None,
        )

    all_tensors = [read_tensors(p) for p in lora_paths]

    combined: dict[str, np.ndarray] = {}
    ref_info = infos[0]

    for pair_idx, ref_pair in enumerate(ref_info.pairs):
        # Reconstruct every adapter's delta for this module first, so the
        # reference (index 0) is available before the others are transformed.
        deltas: list[np.ndarray] = []
        for info, tensors in zip(infos, all_tensors):
            pair = info.pairs[pair_idx]
            lora_a = tensors[pair.lora_a_name].astype(np.float64)
            lora_b = tensors[pair.lora_b_name].astype(np.float64)
            deltas.append(lora_b @ lora_a)

        ref_delta = deltas[0]
        if mode == "norm-scaler":
            for i in range(1, len(deltas)):
                deltas[i] = norm_scale_factor(ref_delta, deltas[i]) * deltas[i]
        elif mode == "gram-schmidt":
            for i in range(1, len(deltas)):
                deltas[i] = gram_schmidt_orthogonalize(ref_delta, deltas[i])

        delta_sum = weights[0] * deltas[0]
        for i in range(1, len(deltas)):
            delta_sum = delta_sum + weights[i] * deltas[i]

        a_new, b_new = _decompose_svd(delta_sum, rank)
        combined[ref_pair.lora_a_name] = a_new
        combined[ref_pair.lora_b_name] = b_new

    metadata = dict(infos[0].metadata)
    metadata["rank"] = str(rank)
    if mode != "none":
        metadata["conflict_mode"] = mode

    write_file(output_path, combined, metadata=metadata)

    return AddResult(
        combined_modules=len(ref_info.pairs),
        output_rank=rank,
        output_path=output_path,
    )
