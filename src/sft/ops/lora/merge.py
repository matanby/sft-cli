"""Merge a LoRA adapter into a base model's weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sft.ops.lora.detect import LoRAInfo, detect_lora, get_base_weight_name
from sft.utils.output import resolve_output
from sft.utils.tensor_io import read_tensors, write_file


@dataclass
class MergeResult:
    """Result of a LoRA merge operation."""

    merged_modules: list[str] = field(default_factory=list)
    unchanged_tensors: int = 0
    output_path: Path | None = None
    scale: float = 1.0


def merge_lora(
    base_path: Path,
    lora_path: Path,
    dst: Path | None = None,
    scale: float | None = None,
    dry_run: bool = False,
) -> MergeResult:
    """Merge a LoRA adapter into base model weights.

    For each LoRA pair: W_merged = W_base + scale * (B @ A)
    """
    info: LoRAInfo | None = detect_lora(lora_path)
    if info is None:
        raise ValueError(f"No LoRA pairs found in {lora_path}")

    if scale is None:
        scale = info.effective_scale if info.effective_scale is not None else 1.0

    output_path = resolve_output(dst, base_path, "merged")

    target_base_names: dict[str, int] = {}
    for pair in info.pairs:
        base_name = get_base_weight_name(pair.module_key)
        target_base_names[base_name] = info.pairs.index(pair)

    result = MergeResult(scale=scale, output_path=output_path)

    if dry_run:
        result.merged_modules = [get_base_weight_name(p.module_key) for p in info.pairs]
        return result

    base_tensors = read_tensors(base_path)
    lora_tensors = read_tensors(lora_path)

    merged: dict[str, np.ndarray] = {}

    for name, tensor in base_tensors.items():
        if name in target_base_names:
            pair = info.pairs[target_base_names[name]]
            lora_a = lora_tensors[pair.lora_a_name]
            lora_b = lora_tensors[pair.lora_b_name]
            delta = lora_b @ lora_a
            merged[name] = tensor + scale * delta.astype(tensor.dtype)
            result.merged_modules.append(name)
        else:
            merged[name] = tensor
            result.unchanged_tensors += 1

    from sft.index import TensorIndex

    base_metadata = TensorIndex.from_file(base_path).metadata
    write_file(output_path, merged, metadata=base_metadata)

    return result
