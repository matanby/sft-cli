"""Singular value spectrum analysis for LoRA adapters.

Uses a QR-accelerated SVD: instead of forming the full B@A matrix
(potentially thousands x thousands) and running SVD on it, we
QR-factor both thin matrices and SVD their small (rank x rank) product.

    B = Qb @ Rb,  A^T = Qa @ Ra^T  (i.e. A = Ra @ Qa^T)
    B @ A = Qb @ (Rb @ Ra) @ Qa^T
    sigma(B @ A) = sigma(Rb @ Ra)   since Qb, Qa are orthonormal

This reduces the cost from O(out * in * min(out,in)) to O(rank^3),
which is ~1000x faster for typical LoRA ranks (8-64) on large layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.ops.lora.detect import detect_lora, format_lora_module_display
from sft.utils.tensor_io import read_tensors


@dataclass
class ModuleSVDInfo:
    """SVD analysis results for a single LoRA module."""

    module_key: str
    module: str
    rank: int
    singular_values: list[float]
    sv_90: int
    sv_95: int
    sv_99: int
    suggested_rank: int


@dataclass
class SVDAnalysis:
    """Aggregated SVD analysis across all LoRA modules."""

    modules: list[ModuleSVDInfo]
    threshold: float


def _sv_for_variance(cumvar: np.ndarray, fraction: float) -> int:
    """Number of singular values needed to capture `fraction` of total variance."""
    return int(np.searchsorted(cumvar, fraction)) + 1


def _qr_svd(
    a: np.ndarray,
    b: np.ndarray,
    compute_uv: bool = False,
) -> tuple[np.ndarray, ...]:
    """SVD of B@A via QR factorization of the thin factors.

    a: shape (rank, in_features) — or higher-dim (reshaped to 2D).
    b: shape (out_features, rank) — or higher-dim (reshaped to 2D).

    Returns (u, s, vt) if compute_uv else (s,).
    Singular values are in descending order, truncated to the LoRA rank.
    """
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if b.ndim > 2:
        b = b.reshape(b.shape[0], -1)

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    rank = min(a.shape[0], b.shape[1])

    if b.shape[1] == a.shape[0]:
        # Standard layout: B (out, rank) @ A (rank, in)
        qa, ra = np.linalg.qr(a.T, mode="reduced")  # (in, rank), (rank, rank)
        qb, rb = np.linalg.qr(b, mode="reduced")  # (out, rank), (rank, rank)
        mid = rb @ ra.T  # (rank, rank)
    elif a.shape[1] == b.shape[0]:
        # Swapped convention: A (out, rank) @ B (rank, in)
        qa, ra = np.linalg.qr(a, mode="reduced")
        qb, rb = np.linalg.qr(b.T, mode="reduced")
        mid = ra @ rb.T
    else:
        # Dimensions don't match a thin factorization — fall back to full product
        mid = (b @ a).astype(np.float64)

    if compute_uv:
        um, s, vtm = np.linalg.svd(mid, full_matrices=False)
        s = s[:rank]
        # Reconstruct full U and Vt from the QR bases
        u = qb @ um[:, :rank]
        vt = vtm[:rank, :] @ qa.T
        return u, s, vt

    s = np.linalg.svd(mid, compute_uv=False)
    return (s[:rank],)


def analyze_svd(path: Path, threshold: float = 0.95) -> SVDAnalysis:
    """Analyze the singular value spectrum of every LoRA pair in *path*.

    Raises ValueError if the file is not a LoRA adapter.
    """
    info = detect_lora(path)
    if info is None:
        raise ValueError(f"{path.name} is not a LoRA adapter")

    tensors = read_tensors(path)
    modules: list[ModuleSVDInfo] = []

    for pair in info.pairs:
        a = tensors[pair.lora_a_name]
        b = tensors[pair.lora_b_name]

        (s,) = _qr_svd(a, b, compute_uv=False)

        s_sq = s**2
        total = s_sq.sum()
        cumvar = np.ones_like(s_sq) if total == 0 else np.cumsum(s_sq) / total

        sv_90 = _sv_for_variance(cumvar, 0.90)
        sv_95 = _sv_for_variance(cumvar, 0.95)
        sv_99 = _sv_for_variance(cumvar, 0.99)
        suggested = _sv_for_variance(cumvar, threshold)

        modules.append(
            ModuleSVDInfo(
                module_key=pair.module_key,
                module=format_lora_module_display(pair.module_key),
                rank=pair.rank,
                singular_values=s.tolist(),
                sv_90=sv_90,
                sv_95=sv_95,
                sv_99=sv_99,
                suggested_rank=suggested,
            )
        )

    return SVDAnalysis(modules=modules, threshold=threshold)
