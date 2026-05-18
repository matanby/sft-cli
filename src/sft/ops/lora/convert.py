"""Bidirectional conversion between Kohya and PEFT LoRA naming conventions.

Kohya format:
    <prefix>.lora_down.weight   (the A factor)
    <prefix>.lora_up.weight     (the B factor)
    <prefix>.alpha              (scalar; effective scale = alpha / rank)

PEFT format:
    <prefix>.lora_A.weight       (the down/A factor)
    <prefix>.lora_B.weight       (the up/B factor)
    alpha and rank stored in __metadata__ (one value per file)

Conversion preserves the effective delta = (alpha/rank) * B @ A exactly:
- Kohya -> PEFT: rename suffixes, drop .alpha tensors, lift alpha/rank to metadata
- PEFT -> Kohya: rename suffixes, emit per-module .alpha tensors from metadata

Only suffixes are renamed; the rest of each tensor name is preserved. Users
who also need dot/underscore remapping can chain `sft rename`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sft.index import TensorIndex
from sft.utils.tensor_io import read_tensors, write_file

KOHYA_DOWN = ".lora_down.weight"
KOHYA_UP = ".lora_up.weight"
KOHYA_ALPHA = ".alpha"
PEFT_A = ".lora_A.weight"
PEFT_B = ".lora_B.weight"


@dataclass
class FormatInfo:
    """Summary of LoRA-format tensors in a file."""

    kohya_modules: int  # prefixes with lora_down/lora_up
    peft_modules: int  # prefixes with lora_A/lora_B
    has_alpha_tensors: int  # number of .alpha tensors found
    non_lora: int  # tensors not matching any LoRA suffix

    @property
    def format(self) -> str | None:
        """Detected format: 'kohya', 'peft', 'mixed', or None."""
        if self.kohya_modules and self.peft_modules:
            return "mixed"
        if self.kohya_modules:
            return "kohya"
        if self.peft_modules:
            return "peft"
        return None


@dataclass
class ConversionResult:
    """Result of a Kohya<->PEFT conversion."""

    source_format: str
    target_format: str
    modules_converted: int
    passthrough: int  # non-LoRA tensors copied unchanged
    output_path: Path
    alpha: float | None = None  # alpha used / written
    rank: int | None = None  # rank of the converted modules


def _group_by_prefix(
    names: list[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Group tensor names by LoRA-module prefix.

    Returns (modules, passthrough). `modules` maps prefix -> {suffix: name}.
    """
    suffixes = (KOHYA_DOWN, KOHYA_UP, KOHYA_ALPHA, PEFT_A, PEFT_B)
    modules: dict[str, dict[str, str]] = {}
    passthrough: list[str] = []
    for name in names:
        for sfx in suffixes:
            if name.endswith(sfx):
                prefix = name[: -len(sfx)]
                modules.setdefault(prefix, {})[sfx] = name
                break
        else:
            passthrough.append(name)
    return modules, passthrough


def detect_format(path: Path) -> FormatInfo:
    """Scan a safetensors file and classify its LoRA naming convention."""
    index = TensorIndex.from_file(path)
    names = [t.full_name for t in index.tensors]
    modules, passthrough = _group_by_prefix(names)

    kohya_n = peft_n = 0
    alpha_n = 0
    for parts in modules.values():
        has_kohya = KOHYA_DOWN in parts or KOHYA_UP in parts
        has_peft = PEFT_A in parts or PEFT_B in parts
        if KOHYA_ALPHA in parts:
            alpha_n += 1
        if has_kohya:
            kohya_n += 1
        elif has_peft:
            peft_n += 1

    return FormatInfo(
        kohya_modules=kohya_n,
        peft_modules=peft_n,
        has_alpha_tensors=alpha_n,
        non_lora=len(passthrough),
    )


def _resolve_target(source: str, target: str | None) -> str:
    """Resolve auto-detected target format ('other' direction)."""
    if target is None:
        return "peft" if source == "kohya" else "kohya"
    if target not in ("kohya", "peft"):
        raise ValueError(f"target must be 'kohya' or 'peft', got {target!r}")
    return target


def _kohya_to_peft(
    src_tensors: dict[str, np.ndarray],
    modules: dict[str, dict[str, str]],
    src_metadata: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, str], int, float | None, int | None]:
    """Convert in-memory tensors from Kohya to PEFT layout."""
    out: dict[str, np.ndarray] = {}
    n_converted = 0
    alphas: list[float] = []
    ranks: list[int] = []

    for prefix, parts in modules.items():
        down = parts.get(KOHYA_DOWN)
        up = parts.get(KOHYA_UP)
        alpha_name = parts.get(KOHYA_ALPHA)
        a_name = parts.get(PEFT_A)
        b_name = parts.get(PEFT_B)

        if down and up:
            a_arr = src_tensors[down]
            b_arr = src_tensors[up]
            rank = a_arr.shape[0]
            ranks.append(rank)
            if alpha_name is not None:
                val = float(np.asarray(src_tensors[alpha_name]).flatten()[0])
                alphas.append(val)
            else:
                alphas.append(float(rank))
            out[f"{prefix}{PEFT_A}"] = a_arr
            out[f"{prefix}{PEFT_B}"] = b_arr
            n_converted += 1
        elif a_name and b_name:
            out[a_name] = src_tensors[a_name]
            out[b_name] = src_tensors[b_name]

    metadata = dict(src_metadata)
    alpha = alphas[0] if alphas else None
    rank = ranks[0] if ranks else None

    if alphas and len({round(a, 6) for a in alphas}) > 1:
        # Heterogeneous alphas — store the first but note it
        metadata["alpha_warning"] = "multiple_alphas_collapsed"
    if alpha is not None:
        metadata["alpha"] = str(int(alpha) if alpha.is_integer() else alpha)
    if rank is not None:
        metadata["rank"] = str(rank)
    metadata["converted_from"] = "kohya"

    return out, metadata, n_converted, alpha, rank


def _peft_to_kohya(
    src_tensors: dict[str, np.ndarray],
    modules: dict[str, dict[str, str]],
    src_metadata: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, str], int, float | None, int | None]:
    """Convert in-memory tensors from PEFT to Kohya layout."""
    metadata_alpha: float | None = None
    if "alpha" in src_metadata:
        try:
            metadata_alpha = float(src_metadata["alpha"])
        except (ValueError, TypeError):
            metadata_alpha = None

    out: dict[str, np.ndarray] = {}
    n_converted = 0
    rank_seen: int | None = None

    for prefix, parts in modules.items():
        a_name = parts.get(PEFT_A)
        b_name = parts.get(PEFT_B)
        down_name = parts.get(KOHYA_DOWN)
        up_name = parts.get(KOHYA_UP)

        if a_name and b_name:
            a_arr = src_tensors[a_name]
            b_arr = src_tensors[b_name]
            rank = a_arr.shape[0]
            rank_seen = rank
            out[f"{prefix}{KOHYA_DOWN}"] = a_arr
            out[f"{prefix}{KOHYA_UP}"] = b_arr
            alpha_val = metadata_alpha if metadata_alpha is not None else float(rank)
            # Kohya alpha is typically stored as fp16 scalar
            out[f"{prefix}{KOHYA_ALPHA}"] = np.array(alpha_val, dtype=np.float16)
            n_converted += 1
        elif down_name and up_name:
            out[down_name] = src_tensors[down_name]
            out[up_name] = src_tensors[up_name]
            if KOHYA_ALPHA in parts:
                out[parts[KOHYA_ALPHA]] = src_tensors[parts[KOHYA_ALPHA]]

    metadata = {k: v for k, v in src_metadata.items() if k not in ("alpha", "rank")}
    metadata["converted_from"] = "peft"

    return out, metadata, n_converted, metadata_alpha, rank_seen


def convert_lora(
    src: Path,
    dst: Path,
    target: str | None = None,
) -> ConversionResult:
    """Convert a LoRA file between Kohya and PEFT naming conventions.

    Args:
        src: Source .safetensors file.
        dst: Output .safetensors path.
        target: Target format ('kohya' or 'peft'). If None, auto-detects the
            source format and converts to the other.

    Returns:
        ConversionResult describing what was done.

    Raises:
        ValueError: if the source format cannot be detected, or is already
            in the target format, or is 'mixed'.
    """
    info = detect_format(src)
    source = info.format
    if source is None:
        raise ValueError(f"No LoRA tensors found in {src.name}; nothing to convert.")
    if source == "mixed":
        raise ValueError(
            f"{src.name} contains both Kohya and PEFT modules; "
            "split or normalize manually before converting."
        )

    target_fmt = _resolve_target(source, target)
    if target_fmt == source:
        raise ValueError(f"{src.name} is already in {source} format.")

    index = TensorIndex.from_file(src)
    src_tensors = read_tensors(src)
    names = [t.full_name for t in index.tensors]
    modules, passthrough = _group_by_prefix(names)

    if target_fmt == "peft":
        out, metadata, n_converted, alpha, rank = _kohya_to_peft(
            src_tensors, modules, index.metadata
        )
    else:
        out, metadata, n_converted, alpha, rank = _peft_to_kohya(
            src_tensors, modules, index.metadata
        )

    for name in passthrough:
        out[name] = src_tensors[name]

    write_file(dst, out, metadata=metadata)

    return ConversionResult(
        source_format=source,
        target_format=target_fmt,
        modules_converted=n_converted,
        passthrough=len(passthrough),
        output_path=dst,
        alpha=alpha,
        rank=rank,
    )
