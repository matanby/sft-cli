"""Pure logic for converting PyTorch checkpoint files to safetensors format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.utils.dtypes import resolve_dtype


@dataclass
class ConvertResult:
    tensors_count: int
    output_path: Path
    source_format: str


def convert_to_safetensors(
    src: Path, dst: Path, dtype: str | None = None
) -> ConvertResult:
    """Load a PyTorch checkpoint and save it as safetensors."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required for conversion. Install with: pip install sft-cli[torch]"
        ) from None

    checkpoint = torch.load(src, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    target_np = resolve_dtype(dtype) if dtype else None

    tensors: dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        arr = tensor.detach().cpu().numpy()
        if target_np is not None:
            arr = arr.astype(target_np)
        tensors[key] = arr

    from safetensors.numpy import save_file

    save_file(tensors, str(dst))

    ext = src.suffix.lower()
    source_format = {".pt": "pt", ".pth": "pth", ".bin": "bin"}.get(ext, ext)

    return ConvertResult(
        tensors_count=len(tensors),
        output_path=dst,
        source_format=source_format,
    )
