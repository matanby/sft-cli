"""Lossless weighted combination of two PEFT-format LoRA adapters.

This is the *factor-stacking* merge — distinct from `add_loras` which
re-decomposes to a fixed rank via SVD (lossy).

Per module:

    merged_A = vstack([coeff_a * A_a, coeff_b * A_b])     shape: (r_a + r_b, in)
    merged_B = hstack([B_a, B_b])                         shape: (out, r_a + r_b)

so ``merged_B @ merged_A == coeff_a * (B_a @ A_a) + coeff_b * (B_b @ A_b)``
exactly. Merged rank is ``r_a + r_b`` (or the input rank, when one side is
missing). An optional `target_rank` runs the same QR-accelerated SVD truncate
used by `resize_lora`, applied to each merged pair after stacking.

Union semantics for modules: when a pair is present in only one file it is
kept (and scaled by that file's coefficient). Non-LoRA tensors are copied
through with collision warnings:

  • only in one file              → copy
  • identical in both             → copy once
  • different values, same dtype + shape → warn, keep file A's version
  • different shape/dtype         → warn, keep file A's version

Kohya-form modules are skipped (run `sft lora convert` first).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.ops.lora.conflict import (
    frob_inner_factored,
    validate_mode,
)
from sft.ops.lora.detect import LoRAInfo, detect_lora
from sft.ops.lora.svd import _qr_svd
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class StackResult:
    """Outcome of a `stack_loras` call."""

    n_both: int = 0  # modules present in both files
    n_a_only: int = 0  # modules only in file A
    n_b_only: int = 0  # modules only in file B
    n_skipped_shape: int = 0  # modules whose in/out shapes don't match
    n_passthrough: int = 0  # non-LoRA tensors copied through
    target_rank: int | None = None  # set when post-merge SVD truncation ran
    collisions: list[str] = field(default_factory=list)  # warning messages
    output_path: Path | None = None


def _truncate_pair(
    a: np.ndarray, b: np.ndarray, target_rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """SVD-truncate a single A/B pair to the given rank, sharing svd.py logic."""
    u, s, vt = _qr_svd(a, b, compute_uv=True)
    r = target_rank
    sqrt_s = np.sqrt(s[:r])
    a_new = (sqrt_s[:, np.newaxis] * vt[:r, :]).astype(a.dtype)
    b_new = (u[:, :r] * sqrt_s[np.newaxis, :]).astype(b.dtype)
    return a_new, b_new


def _by_module_key(info: LoRAInfo) -> dict[str, tuple[str, str]]:
    """Map module_key -> (lora_a_name, lora_b_name)."""
    return {p.module_key: (p.lora_a_name, p.lora_b_name) for p in info.pairs}


def _collect_non_lora_names(index: TensorIndex, info: LoRAInfo) -> list[str]:
    """All tensor names that are NOT part of a detected LoRA pair."""
    lora_names = {p.lora_a_name for p in info.pairs} | {
        p.lora_b_name for p in info.pairs
    }
    return [t.full_name for t in index.tensors if t.full_name not in lora_names]


def stack_loras(
    src_a: Path,
    src_b: Path,
    dst: Path | None,
    coeff_a: float = 1.0,
    coeff_b: float = 1.0,
    target_rank: int | None = None,
    dry_run: bool = False,
    mode: str = "none",
) -> StackResult:
    """Combine two PEFT LoRAs via lossless factor stacking.

    Both inputs must be PEFT-format LoRA adapters. Modules present in
    only one file are kept and scaled by that side's coefficient.

    ``mode`` optionally resolves conflict with file A (the reference) before
    stacking, staying lossless because both transforms are linear and fold
    into the concatenated factors as a per-module coefficient tweak:

      • ``norm-scaler``:   B's A-block coefficient becomes ``coeff_b · s`` with
        ``s = ‖ΔW_a‖_F / ‖ΔW_b‖_F`` (matches file B's delta norm to A's).
      • ``gram-schmidt``:  A's A-block coefficient becomes ``coeff_a − coeff_b · c``
        with ``c = ⟨ΔW_a, ΔW_b⟩ / ⟨ΔW_a, ΔW_a⟩`` (removes the A-aligned part of B).

    Scalars are computed in rank space, so the full delta is never formed.

    Raises:
        ValueError: if ``mode`` is invalid, either input is not a PEFT LoRA,
            or shapes within a paired module are incompatible.
    """
    validate_mode(mode)

    info_a = detect_lora(src_a)
    info_b = detect_lora(src_b)
    if info_a is None:
        raise ValueError(f"Not a PEFT LoRA adapter: {src_a.name}")
    if info_b is None:
        raise ValueError(f"Not a PEFT LoRA adapter: {src_b.name}")

    tensors_a = read_tensors(src_a)
    tensors_b = read_tensors(src_b)
    index_a = TensorIndex.from_file(src_a)
    index_b = TensorIndex.from_file(src_b)

    pairs_a = _by_module_key(info_a)
    pairs_b = _by_module_key(info_b)
    all_modules = sorted(set(pairs_a) | set(pairs_b))

    out_tensors: dict[str, np.ndarray] = {}
    result = StackResult(target_rank=target_rank)

    for module_key in all_modules:
        in_a = module_key in pairs_a
        in_b = module_key in pairs_b

        if in_a and in_b:
            a_name_a, b_name_a = pairs_a[module_key]
            a_name_b, b_name_b = pairs_b[module_key]
            A_a = tensors_a[a_name_a]
            B_a = tensors_a[b_name_a]
            A_b = tensors_b[a_name_b]
            B_b = tensors_b[b_name_b]

            # Shape compatibility: A's in_features and B's out_features must match.
            # A has shape (rank, in_features); B has shape (out_features, rank).
            if A_a.shape[1] != A_b.shape[1] or B_a.shape[0] != B_b.shape[0]:
                result.n_skipped_shape += 1
                continue

            # Effective A-block coefficients. The conflict transforms are linear
            # in the deltas, so they fold losslessly into these scalars while
            # merged_B stays a plain hstack (see the docstring).
            eff_a = coeff_a
            eff_b = coeff_b
            if mode == "norm-scaler":
                norm_a_sq = frob_inner_factored(A_a, B_a, A_a, B_a)
                norm_b_sq = frob_inner_factored(A_b, B_b, A_b, B_b)
                if norm_b_sq > 0.0:
                    eff_b = coeff_b * float(np.sqrt(norm_a_sq / norm_b_sq))
            elif mode == "gram-schmidt":
                norm_a_sq = frob_inner_factored(A_a, B_a, A_a, B_a)
                if norm_a_sq > 0.0:
                    inner_ab = frob_inner_factored(A_a, B_a, A_b, B_b)
                    eff_a = coeff_a - coeff_b * (inner_ab / norm_a_sq)

            merged_A = np.concatenate(
                [eff_a * A_a.astype(np.float32), eff_b * A_b.astype(np.float32)],
                axis=0,
            ).astype(A_a.dtype)
            merged_B = np.concatenate(
                [B_a.astype(np.float32), B_b.astype(np.float32)], axis=1
            ).astype(B_a.dtype)

            if target_rank is not None and target_rank < merged_A.shape[0]:
                merged_A, merged_B = _truncate_pair(merged_A, merged_B, target_rank)

            out_tensors[a_name_a] = merged_A
            out_tensors[b_name_a] = merged_B
            result.n_both += 1

        elif in_a:
            a_name, b_name = pairs_a[module_key]
            A = (coeff_a * tensors_a[a_name].astype(np.float32)).astype(
                tensors_a[a_name].dtype
            )
            B = tensors_a[b_name]
            if target_rank is not None and target_rank < A.shape[0]:
                A, B = _truncate_pair(A, B, target_rank)
            out_tensors[a_name] = A
            out_tensors[b_name] = B
            result.n_a_only += 1

        else:  # in_b only
            a_name, b_name = pairs_b[module_key]
            A = (coeff_b * tensors_b[a_name].astype(np.float32)).astype(
                tensors_b[a_name].dtype
            )
            B = tensors_b[b_name]
            if target_rank is not None and target_rank < A.shape[0]:
                A, B = _truncate_pair(A, B, target_rank)
            out_tensors[a_name] = A
            out_tensors[b_name] = B
            result.n_b_only += 1

    # Passthrough of non-LoRA tensors with collision detection.
    non_lora_a = _collect_non_lora_names(index_a, info_a)
    non_lora_b = _collect_non_lora_names(index_b, info_b)
    set_b = set(non_lora_b)

    for name in non_lora_a:
        if name in set_b:
            ta = tensors_a[name]
            tb = tensors_b[name]
            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                result.collisions.append(
                    f"{name}: shape/dtype mismatch "
                    f"({ta.shape} {ta.dtype} vs {tb.shape} {tb.dtype}); "
                    "keeping file A"
                )
            elif not np.array_equal(ta, tb):
                result.collisions.append(
                    f"{name}: present in both files with different values; "
                    "keeping file A"
                )
        out_tensors[name] = tensors_a[name]
        result.n_passthrough += 1

    # Tensors only in file B (not in A)
    for name in non_lora_b:
        if name not in set(non_lora_a):
            out_tensors[name] = tensors_b[name]
            result.n_passthrough += 1

    result.output_path = dst

    if not dry_run and dst is not None:
        metadata = dict(info_a.metadata)
        metadata["merged_from"] = json.dumps(
            {
                "file_a": str(src_a.name),
                "coeff_a": coeff_a,
                "file_b": str(src_b.name),
                "coeff_b": coeff_b,
                "target_rank": target_rank,
                "conflict_mode": mode,
            }
        )
        if mode != "none":
            metadata["conflict_mode"] = mode
        if target_rank is not None:
            metadata["rank"] = str(target_rank)
        write_file(dst, out_tensors, metadata=metadata)

    return result
