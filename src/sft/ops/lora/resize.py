"""Reduce LoRA rank via truncated SVD.

Two modes:

- **Fixed rank**: every pair is truncated to the same target rank.
- **Auto rank**: each pair is truncated to ``ceil(stable_rank) + margin``,
  so modules that already have low effective rank get compressed further
  than modules with high effective rank. Output files have heterogeneous
  ranks across pairs (PEFT loaders handle this fine).

Both modes use the QR-accelerated SVD from svd.py — avoids forming the
full B@A product matrix, keeping cost at O(rank^3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.ops.lora.detect import detect_lora
from sft.ops.lora.svd import _qr_svd
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class ResizeResult:
    """Result of resizing a LoRA adapter.

    `new_rank` is the largest output rank across pairs (for auto mode this
    can differ per pair; see `per_module_ranks`). `errors` and `energies`
    are per-pair: errors is the relative Frobenius reconstruction error,
    energies is the fraction of singular-value squared mass retained.
    """

    original_rank: int
    new_rank: int
    modules_resized: int
    errors: dict[str, float] = field(default_factory=dict)
    energies: dict[str, float] = field(default_factory=dict)
    per_module_ranks: dict[str, int] = field(default_factory=dict)
    auto_margin: int | None = None
    output_path: Path | None = None


def _stable_rank(s: np.ndarray) -> float:
    """Stable rank: sum(s^2) / max(s)^2  (a.k.a. effective rank)."""
    if s.size == 0:
        return 0.0
    s_sq = s.astype(np.float64) ** 2
    smax = s_sq.max()
    return float(s_sq.sum() / smax) if smax > 0 else 0.0


def resize_lora(
    src: Path,
    dst: Path | None,
    target_rank: int | None = None,
    auto_margin: int | None = None,
    dry_run: bool = False,
) -> ResizeResult:
    """Resize a LoRA adapter to a lower rank using truncated SVD.

    Exactly one of `target_rank` (fixed mode) or `auto_margin` (auto mode)
    must be set.

    Auto mode picks per-pair `ceil(stable_rank) + margin`. Pairs whose
    auto-rank is >= their current rank are left at their current rank
    (no expansion).
    """
    if (target_rank is None) == (auto_margin is None):
        raise ValueError("Exactly one of `target_rank` or `auto_margin` must be set")
    if target_rank is not None and target_rank < 1:
        raise ValueError(f"Target rank must be >= 1, got {target_rank}")
    if auto_margin is not None and auto_margin < 0:
        raise ValueError(f"auto_margin must be >= 0, got {auto_margin}")

    info = detect_lora(src)
    if info is None:
        raise ValueError(f"Not a LoRA adapter: {src.name}")

    if target_rank is not None and target_rank >= info.rank:
        raise ValueError(
            f"Target rank {target_rank} must be less than current rank {info.rank}"
        )

    tensors = read_tensors(src)
    new_tensors: dict[str, np.ndarray] = {}
    errors: dict[str, float] = {}
    energies: dict[str, float] = {}
    per_module_ranks: dict[str, int] = {}

    for name in tensors:
        if not any(name == p.lora_a_name or name == p.lora_b_name for p in info.pairs):
            new_tensors[name] = tensors[name]

    for pair in info.pairs:
        a = tensors[pair.lora_a_name]
        b = tensors[pair.lora_b_name]
        orig_dtype = a.dtype

        u, s, vt = _qr_svd(a, b, compute_uv=True)

        # Pick this pair's target rank
        if auto_margin is not None:
            sr = _stable_rank(s)
            r = min(pair.rank, max(1, math.ceil(sr) + auto_margin))
        else:
            r = target_rank  # type: ignore[assignment]

        sqrt_s = np.sqrt(s[:r])
        a_new = (sqrt_s[:, np.newaxis] * vt[:r, :]).astype(orig_dtype)
        b_new = (u[:, :r] * sqrt_s[np.newaxis, :]).astype(orig_dtype)

        s_sq = s**2
        total_energy = float(s_sq.sum())
        kept_energy = float(s_sq[:r].sum())
        if total_energy > 0:
            energy_retained = kept_energy / total_energy
            error = float(np.sqrt(max(0.0, 1.0 - energy_retained)))
        else:
            energy_retained = 1.0
            error = 0.0

        errors[pair.target_module] = error
        energies[pair.target_module] = energy_retained
        per_module_ranks[pair.target_module] = r
        new_tensors[pair.lora_a_name] = a_new
        new_tensors[pair.lora_b_name] = b_new

    new_rank_advertised = max(per_module_ranks.values()) if per_module_ranks else 0
    if target_rank is not None:
        new_rank_advertised = target_rank

    result = ResizeResult(
        original_rank=info.rank,
        new_rank=new_rank_advertised,
        modules_resized=len(info.pairs),
        errors=errors,
        energies=energies,
        per_module_ranks=per_module_ranks,
        auto_margin=auto_margin,
        output_path=dst,
    )

    if not dry_run and dst is not None:
        metadata = dict(info.metadata)
        metadata["rank"] = str(new_rank_advertised)
        if auto_margin is not None:
            metadata["resize_mode"] = f"auto+{auto_margin}"
        write_file(dst, new_tensors, metadata=metadata)

    return result
