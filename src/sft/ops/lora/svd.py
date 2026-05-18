"""Singular value spectrum analysis for LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.ops.lora.detect import detect_lora
from sft.utils.tensor_io import read_tensors


@dataclass
class ModuleSVDInfo:
    """SVD analysis results for a single LoRA module."""

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
        a = tensors[pair.lora_a_name].astype(np.float64)
        b = tensors[pair.lora_b_name].astype(np.float64)

        delta = b @ a
        _u, s, _vt = np.linalg.svd(delta, full_matrices=False)

        s_sq = s**2
        total = s_sq.sum()
        cumvar = np.ones_like(s_sq) if total == 0 else np.cumsum(s_sq) / total

        sv_90 = _sv_for_variance(cumvar, 0.90)
        sv_95 = _sv_for_variance(cumvar, 0.95)
        sv_99 = _sv_for_variance(cumvar, 0.99)
        suggested = _sv_for_variance(cumvar, threshold)

        modules.append(
            ModuleSVDInfo(
                module=pair.target_module,
                rank=pair.rank,
                singular_values=s.tolist(),
                sv_90=sv_90,
                sv_95=sv_95,
                sv_99=sv_99,
                suggested_rank=suggested,
            )
        )

    return SVDAnalysis(modules=modules, threshold=threshold)
