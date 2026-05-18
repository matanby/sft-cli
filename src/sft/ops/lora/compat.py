"""Check compatibility between a LoRA adapter and a base model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sft.index import TensorIndex
from sft.ops.lora.detect import detect_lora, get_base_weight_name


@dataclass
class ShapeMismatch:
    module: str
    base_shape: tuple[int, ...]
    lora_a_shape: tuple[int, ...]
    expected_in_features: int
    actual_in_features: int


@dataclass
class CompatResult:
    compatible: bool
    missing_modules: list[str] = field(default_factory=list)
    shape_mismatches: list[ShapeMismatch] = field(default_factory=list)
    matched_modules: int = 0


def check_compat(base_path: Path, lora_path: Path) -> CompatResult:
    """Check whether a LoRA adapter is compatible with a base model.

    For each LoRA pair, finds the corresponding base weight and verifies
    that dimensions align for merging (in_features and out_features).
    """
    lora_info = detect_lora(lora_path)
    if lora_info is None:
        raise ValueError(f"No LoRA pairs found in {lora_path}")

    base_index = TensorIndex.from_file(base_path)
    base_by_name = {t.full_name: t for t in base_index.tensors}

    missing: list[str] = []
    mismatches: list[ShapeMismatch] = []
    matched = 0

    for pair in lora_info.pairs:
        base_weight_name = get_base_weight_name(pair.module_key)
        base_tensor = base_by_name.get(base_weight_name)

        if base_tensor is None:
            missing.append(pair.target_module)
            continue

        base_in = base_tensor.shape[-1]
        lora_in = pair.lora_a_shape[1]
        base_out = base_tensor.shape[0]
        lora_out = pair.lora_b_shape[0]

        if base_in != lora_in or base_out != lora_out:
            mismatches.append(
                ShapeMismatch(
                    module=pair.target_module,
                    base_shape=base_tensor.shape,
                    lora_a_shape=pair.lora_a_shape,
                    expected_in_features=base_in,
                    actual_in_features=lora_in,
                )
            )
            continue

        matched += 1

    compatible = not missing and not mismatches
    return CompatResult(
        compatible=compatible,
        missing_modules=missing,
        shape_mismatches=mismatches,
        matched_modules=matched,
    )
