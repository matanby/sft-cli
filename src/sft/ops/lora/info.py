"""LoRA adapter analysis logic."""

from __future__ import annotations

from pathlib import Path

from sft.ops.lora.detect import LoRAInfo, detect_lora


def lora_info(path: Path) -> LoRAInfo:
    """Analyze a LoRA adapter file. Raises ValueError if not a LoRA file."""
    info = detect_lora(path)
    if info is None:
        raise ValueError(f"Not a LoRA adapter file: {path.name}")
    return info
