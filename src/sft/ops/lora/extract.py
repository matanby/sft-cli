"""Extract a LoRA adapter from the weight delta between a base and fine-tuned model.

Uses truncated SVD to decompose each weight delta into low-rank A/B pairs,
stored in PEFT-compatible naming format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.utils.glob import filter_tensors
from sft.utils.output import resolve_output
from sft.utils.tensor_io import read_tensors, write_file

DELTA_NORM_EPSILON = 1e-12


@dataclass
class ExtractionResult:
    modules: list[str]
    errors: dict[str, float] = field(default_factory=dict)
    output_path: Path | None = None
    rank: int = 0


def _to_peft_key(tensor_name: str) -> str:
    """Convert a base model tensor name to a PEFT module key.

    'model.layers.0.self_attn.q_proj.weight'
    → 'base_model.model.model.layers.0.self_attn.q_proj'
    """
    if tensor_name.endswith(".weight"):
        tensor_name = tensor_name[: -len(".weight")]
    return f"base_model.model.{tensor_name}"


def _decompose(delta: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD decomposition of a weight delta.

    Returns (lora_A, lora_B) where B @ A ≈ delta.
    lora_A shape: (rank, in_features)
    lora_B shape: (out_features, rank)
    """
    U, S, Vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    r = min(rank, len(S))
    sqrt_S = np.sqrt(S[:r])
    lora_A = sqrt_S[:, np.newaxis] * Vt[:r, :]
    lora_B = U[:, :r] * sqrt_S[np.newaxis, :]
    return lora_A.astype(np.float32), lora_B.astype(np.float32)


def extract_lora(
    base_path: Path,
    finetuned_path: Path,
    dst: Path | None,
    rank: int,
    include: str | None = None,
    exclude: str | None = None,
    alpha: float | None = None,
    dry_run: bool = False,
) -> ExtractionResult:
    """Extract a LoRA adapter from the delta between base and fine-tuned models."""
    if alpha is None:
        alpha = float(rank)

    base_tensors = read_tensors(base_path)
    ft_tensors = read_tensors(finetuned_path)

    common_names = sorted(set(base_tensors) & set(ft_tensors))
    candidates = [
        n for n in common_names if base_tensors[n].ndim == 2 and ft_tensors[n].ndim == 2
    ]
    candidates = filter_tensors(candidates, include=include, exclude=exclude)

    output_path = resolve_output(dst, base_path, f"lora-r{rank}")

    lora_tensors: dict[str, np.ndarray] = {}
    modules: list[str] = []
    errors: dict[str, float] = {}

    for name in candidates:
        delta = ft_tensors[name].astype(np.float64) - base_tensors[name].astype(
            np.float64
        )

        delta_norm = np.linalg.norm(delta, "fro")
        if delta_norm < DELTA_NORM_EPSILON:
            continue

        lora_A, lora_B = _decompose(delta, rank)

        peft_key = _to_peft_key(name)
        lora_tensors[f"{peft_key}.lora_A.weight"] = lora_A
        lora_tensors[f"{peft_key}.lora_B.weight"] = lora_B

        reconstruction = lora_B.astype(np.float64) @ lora_A.astype(np.float64)
        error = float(np.linalg.norm(delta - reconstruction, "fro") / delta_norm)

        module_name = name[: -len(".weight")] if name.endswith(".weight") else name
        modules.append(module_name)
        errors[module_name] = error

    if not dry_run and lora_tensors:
        metadata = {"rank": str(rank), "alpha": str(alpha)}
        write_file(output_path, lora_tensors, metadata=metadata)

    return ExtractionResult(
        modules=modules,
        errors=errors,
        output_path=output_path,
        rank=rank,
    )
