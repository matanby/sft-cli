"""Tests for LoRA extraction from weight deltas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from sft.ops.lora.extract import extract_lora


def test_extract_basic(lora_base_model: Path, finetuned_model: Path, tmp_path: Path):
    """Output contains lora_A/B pairs for each decomposed module."""
    out = tmp_path / "out.safetensors"
    result = extract_lora(lora_base_model, finetuned_model, dst=out, rank=4)

    assert out.exists()
    assert len(result.modules) > 0

    tensors = load_file(str(out))
    for module in result.modules:
        peft_key = f"base_model.model.{module}"
        a_key = f"{peft_key}.lora_A.weight"
        b_key = f"{peft_key}.lora_B.weight"
        assert a_key in tensors, f"Missing lora_A for {module}"
        assert b_key in tensors, f"Missing lora_B for {module}"


def test_extract_rank(lora_base_model: Path, finetuned_model: Path, tmp_path: Path):
    """lora_A first dimension equals the requested rank."""
    out = tmp_path / "out.safetensors"
    rank = 4
    result = extract_lora(lora_base_model, finetuned_model, dst=out, rank=rank)

    tensors = load_file(str(out))
    for module in result.modules:
        peft_key = f"base_model.model.{module}"
        a = tensors[f"{peft_key}.lora_A.weight"]
        b = tensors[f"{peft_key}.lora_B.weight"]
        assert a.shape[0] == rank
        assert b.shape[1] == rank


def test_extract_reconstruction(
    lora_base_model: Path, finetuned_model: Path, tmp_path: Path
):
    """B @ A approximates the original delta within tolerance."""
    out = tmp_path / "out.safetensors"
    rank = 4
    result = extract_lora(lora_base_model, finetuned_model, dst=out, rank=rank)

    base = load_file(str(lora_base_model))
    ft = load_file(str(finetuned_model))
    lora = load_file(str(out))

    for module in result.modules:
        weight_name = f"{module}.weight"
        delta = ft[weight_name].astype(np.float64) - base[weight_name].astype(
            np.float64
        )

        peft_key = f"base_model.model.{module}"
        A = lora[f"{peft_key}.lora_A.weight"].astype(np.float64)
        B = lora[f"{peft_key}.lora_B.weight"].astype(np.float64)

        reconstructed = B @ A
        error = np.linalg.norm(delta - reconstructed, "fro") / np.linalg.norm(
            delta, "fro"
        )
        assert error < 0.5, f"Reconstruction error too high for {module}: {error:.4f}"


def test_extract_smart_output(lora_base_model: Path, finetuned_model: Path):
    """Default output path uses {base_stem}.lora-r{rank}.safetensors."""
    result = extract_lora(lora_base_model, finetuned_model, dst=None, rank=4)
    assert result.output_path is not None
    assert result.output_path.name == "base_model.lora-r4.safetensors"


def test_extract_metadata(lora_base_model: Path, finetuned_model: Path, tmp_path: Path):
    """Output file contains rank and alpha in metadata."""
    out = tmp_path / "out.safetensors"
    extract_lora(lora_base_model, finetuned_model, dst=out, rank=4, alpha=8.0)

    from sft.index import TensorIndex

    index = TensorIndex.from_file(out)
    assert index.metadata["rank"] == "4"
    assert index.metadata["alpha"] == "8.0"
