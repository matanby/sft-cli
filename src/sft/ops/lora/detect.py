"""LoRA format detection and A/B pair grouping.

Supports PEFT-style naming:
  base_model.model.{module}.lora_A.weight
  base_model.model.{module}.lora_B.weight
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sft.index import TensorIndex

LORA_A_PATTERN = re.compile(r"^(.+)\.lora_A\.(?:weight|default\.weight)$")
LORA_B_PATTERN = re.compile(r"^(.+)\.lora_B\.(?:weight|default\.weight)$")

PEFT_PREFIX = re.compile(r"^base_model\.model\.")


@dataclass
class LoRAPair:
    """A matched lora_A / lora_B pair for a single target module."""

    module_key: str
    target_module: str
    lora_a_name: str
    lora_b_name: str
    lora_a_shape: tuple[int, ...]
    lora_b_shape: tuple[int, ...]
    rank: int
    dtype: str

    @property
    def in_features(self) -> int:
        return self.lora_a_shape[1]

    @property
    def out_features(self) -> int:
        return self.lora_b_shape[0]


@dataclass
class LoRAInfo:
    """Detected LoRA structure from a safetensors file."""

    pairs: list[LoRAPair]
    rank: int
    alpha: float | None
    target_modules: list[str]
    total_params: int
    metadata: dict[str, str]
    non_lora_tensors: list[str] = field(default_factory=list)

    @property
    def effective_scale(self) -> float | None:
        if self.alpha is not None and self.rank > 0:
            return self.alpha / self.rank
        return None

    @property
    def num_layers(self) -> int:
        layer_indices: set[str] = set()
        for pair in self.pairs:
            match = re.search(r"layers\.(\d+)", pair.module_key)
            if match:
                layer_indices.add(match.group(1))
        return len(layer_indices) if layer_indices else 1


def strip_peft_prefix(name: str) -> str:
    """Remove PEFT wrapper prefix from a tensor name."""
    return PEFT_PREFIX.sub("", name)


def extract_target_module(module_key: str) -> str:
    """Extract the target module name from a full module key.

    e.g. 'base_model.model.model.layers.0.self_attn.q_proj' → 'q_proj'
    """
    parts = module_key.split(".")
    return parts[-1] if parts else module_key


def format_lora_module_display(module_key: str, *, tail_parts: int = 6) -> str:
    """Return a concise but unambiguous display name for a LoRA module key.

    Strips the PEFT wrapper prefix and keeps the last *tail_parts* dot-separated
    segments so layer/block context is visible (e.g. ``0.attn.to_k`` rather than
    just ``to_k`` or ``0``).
    """
    name = strip_peft_prefix(module_key)
    parts = name.split(".")
    if len(parts) > tail_parts:
        return ".".join(parts[-tail_parts:])
    return name


def detect_lora(path: Path) -> LoRAInfo | None:
    """Detect LoRA structure in a safetensors file.

    Returns LoRAInfo if the file contains LoRA A/B pairs, None otherwise.
    """
    index = TensorIndex.from_file(path)

    a_tensors: dict[str, tuple[str, tuple[int, ...], str]] = {}
    b_tensors: dict[str, tuple[str, tuple[int, ...], str]] = {}
    non_lora: list[str] = []

    for t in index.tensors:
        a_match = LORA_A_PATTERN.match(t.full_name)
        if a_match:
            module_key = a_match.group(1)
            a_tensors[module_key] = (t.full_name, t.shape, t.dtype)
            continue

        b_match = LORA_B_PATTERN.match(t.full_name)
        if b_match:
            module_key = b_match.group(1)
            b_tensors[module_key] = (t.full_name, t.shape, t.dtype)
            continue

        non_lora.append(t.full_name)

    if not a_tensors and not b_tensors:
        return None

    pairs: list[LoRAPair] = []
    matched_keys = set(a_tensors.keys()) & set(b_tensors.keys())

    for key in sorted(matched_keys):
        a_name, a_shape, a_dtype = a_tensors[key]
        b_name, b_shape, b_dtype = b_tensors[key]

        rank = a_shape[0]
        target = extract_target_module(key)

        pairs.append(
            LoRAPair(
                module_key=key,
                target_module=target,
                lora_a_name=a_name,
                lora_b_name=b_name,
                lora_a_shape=a_shape,
                lora_b_shape=b_shape,
                rank=rank,
                dtype=a_dtype,
            )
        )

    if not pairs:
        return None

    rank = pairs[0].rank
    target_modules = sorted({p.target_module for p in pairs})

    total_params = 0
    for p in pairs:
        a_numel = 1
        for d in p.lora_a_shape:
            a_numel *= d
        b_numel = 1
        for d in p.lora_b_shape:
            b_numel *= d
        total_params += a_numel + b_numel

    alpha = None
    if "alpha" in index.metadata:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            alpha = float(index.metadata["alpha"])

    return LoRAInfo(
        pairs=pairs,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        total_params=total_params,
        metadata=index.metadata,
        non_lora_tensors=non_lora,
    )


def get_base_weight_name(module_key: str) -> str:
    """Derive the base model weight name from a LoRA module key.

    e.g. 'base_model.model.model.layers.0.self_attn.q_proj'
         → 'model.layers.0.self_attn.q_proj.weight'
    """
    stripped = strip_peft_prefix(module_key)
    return f"{stripped}.weight"
