"""LoRA detection and analysis for safetensors files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from sft.index import TensorInfo


class LoraRole(Enum):
    """Role of a tensor in a LoRA pair."""

    A = "A (down)"
    B = "B (up)"


@dataclass
class LoraPair:
    """A paired set of LoRA tensors (A and B)."""

    base_name: str
    a_tensor_name: str
    b_tensor_name: str
    rank: int
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]


@dataclass
class LoraInfo:
    """LoRA information for a single tensor."""

    role: LoraRole
    pair: LoraPair


# Patterns: (a_suffix, b_suffix) — the part that distinguishes A from B
_LORA_PATTERNS = [
    (r"lora_A", r"lora_B"),
    (r"lora_a", r"lora_b"),
    (r"lora_down", r"lora_up"),
    (r"down", r"up"),
]


def _try_match_lora_a(
    name: str,
) -> tuple[str, str, str] | None:
    """Try to match a tensor name as a LoRA A tensor.

    Returns (base_name, a_suffix, b_suffix) or None.
    """
    for a_suffix, b_suffix in _LORA_PATTERNS:
        # Match the suffix as a component (preceded by . or _ or start)
        pattern = rf"^(.*(?:\.|_)){re.escape(a_suffix)}((?:\..*)?)$"
        m = re.match(pattern, name)
        if m:
            prefix = m.group(1)
            trailing = m.group(2)
            base = prefix.rstrip("._") + trailing
            b_name = prefix + b_suffix + trailing
            return base, name, b_name
    return None


def detect_lora_pairs(
    tensors: list[TensorInfo],
) -> tuple[list[LoraPair], dict[str, LoraInfo]]:
    """Detect LoRA tensor pairs from a list of tensors.

    Returns:
        A tuple of (list of LoraPair, dict mapping tensor name to LoraInfo).
    """
    tensor_map = {t.full_name: t for t in tensors}
    pairs: list[LoraPair] = []
    tensor_lora_map: dict[str, LoraInfo] = {}
    seen_a: set[str] = set()

    for tensor in tensors:
        if tensor.full_name in seen_a:
            continue

        match = _try_match_lora_a(tensor.full_name)
        if match is None:
            continue

        base_name, a_name, b_name = match

        if b_name not in tensor_map:
            continue

        seen_a.add(a_name)
        a_tensor = tensor_map[a_name]
        b_tensor = tensor_map[b_name]

        # Determine rank from the shared inner dimension
        # A is typically (rank, in_features) and B is (out_features, rank)
        # But conventions vary — find the shared small dimension
        a_shape = a_tensor.shape
        b_shape = b_tensor.shape

        rank = _extract_rank(a_shape, b_shape)

        pair = LoraPair(
            base_name=base_name,
            a_tensor_name=a_name,
            b_tensor_name=b_name,
            rank=rank,
            a_shape=a_shape,
            b_shape=b_shape,
        )
        pairs.append(pair)

        tensor_lora_map[a_name] = LoraInfo(role=LoraRole.A, pair=pair)
        tensor_lora_map[b_name] = LoraInfo(role=LoraRole.B, pair=pair)

    return pairs, tensor_lora_map


def _extract_rank(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> int:
    """Extract the LoRA rank from paired tensor shapes.

    For 2D tensors: A is (rank, in) or (in, rank), B is (out, rank) or (rank, out).
    The rank is the shared small dimension.
    """
    if len(a_shape) == 2 and len(b_shape) == 2:
        # Find shared dimension values
        a_dims = set(a_shape)
        b_dims = set(b_shape)
        shared = a_dims & b_dims
        if shared:
            return min(shared)
        # Fallback: rank is typically the smaller dimension of A
        return min(a_shape)
    # For other shapes, use smallest dimension of A
    if a_shape:
        return min(a_shape)
    return 0
