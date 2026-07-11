"""Conflict-resolution transforms for combining LoRA adapters.

When two adapters are combined, they can *conflict*: one update can dominate
by magnitude, or the two can overlap in the same directions. Both fixes here
transform a non-reference adapter's delta ``ΔW = B @ A`` relative to a
reference adapter's delta, per module, before the normal combination runs:

  • ``norm-scaler`` — rescale so ``‖ΔW_other‖_F == ‖ΔW_ref‖_F``.
  • ``gram-schmidt`` — subtract the component of ``ΔW_other`` aligned (Frobenius
    projection) with ``ΔW_ref``: ``ΔW_other − c·ΔW_ref`` with
    ``c = ⟨ΔW_ref, ΔW_other⟩ / ⟨ΔW_ref, ΔW_ref⟩``.

Both transforms are linear, which is what lets ``stack`` apply them losslessly
as a per-module coefficient tweak on the concatenated factors (see stack.py).
"""

from __future__ import annotations

import numpy as np

VALID_MODES = ("none", "norm-scaler", "gram-schmidt")


def validate_mode(mode: str, allowed: tuple[str, ...] = VALID_MODES) -> str:
    """Return *mode* if it is in *allowed*, else raise ValueError.

    Guards direct callers of the ops; the Typer enum guards the CLI.
    """
    if mode not in allowed:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {', '.join(allowed)}")
    return mode


def frob_inner_factored(
    a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray
) -> float:
    """Frobenius inner product ⟨B1@A1, B2@A2⟩ computed in rank space (float64).

    ⟨B1 A1, B2 A2⟩_F = trace(A1ᵀ B1ᵀ B2 A2) = Σ((B1ᵀB2) ⊙ (A1 A2ᵀ)),
    where both operands are small (rank-sized), so the full out×in delta is
    never formed. Mirrors the QR-accelerated approach used in svd.py.
    """
    a1 = a1.astype(np.float64)
    b1 = b1.astype(np.float64)
    a2 = a2.astype(np.float64)
    b2 = b2.astype(np.float64)
    return float(np.sum((b1.T @ b2) * (a1 @ a2.T)))


def norm_scale_factor(
    dw_ref: np.ndarray, dw_other: np.ndarray, eps: float = 1e-12
) -> float:
    """Scalar ``s`` with ``‖s·dw_other‖_F == ‖dw_ref‖_F``.

    Returns 1.0 (no-op) when ``dw_other`` is (near) zero, matching the
    reference script's "skip if norm == 0" behaviour.
    """
    norm_ref = float(np.linalg.norm(dw_ref.astype(np.float64)))
    norm_other = float(np.linalg.norm(dw_other.astype(np.float64)))
    if norm_other <= eps:
        return 1.0
    return norm_ref / norm_other


def gram_schmidt_coefficient(
    dw_ref: np.ndarray, dw_other: np.ndarray, eps: float = 1e-12
) -> float:
    """Projection coefficient ``c = ⟨dw_ref, dw_other⟩ / ⟨dw_ref, dw_ref⟩``.

    Returns 0.0 (no-op) when ``dw_ref`` is (near) zero.
    """
    dw_ref = dw_ref.astype(np.float64)
    dw_other = dw_other.astype(np.float64)
    denom = float(np.sum(dw_ref * dw_ref))
    if denom <= eps:
        return 0.0
    return float(np.sum(dw_ref * dw_other)) / denom


def gram_schmidt_orthogonalize(
    dw_ref: np.ndarray, dw_other: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Remove the reference-aligned component: ``dw_other − c·dw_ref`` (float64).

    Returns ``dw_other`` unchanged when ``dw_ref`` is (near) zero.
    """
    dw_ref = dw_ref.astype(np.float64)
    dw_other = dw_other.astype(np.float64)
    c = gram_schmidt_coefficient(dw_ref, dw_other, eps=eps)
    if c == 0.0:
        return dw_other
    return dw_other - c * dw_ref
