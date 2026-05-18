"""Shared test fixtures for sft tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def mini_model(tmp_path: Path) -> Path:
    """A tiny transformer-like model with 2 layers, mixed dtypes, and metadata."""
    tensors = {
        "model.embed_tokens.weight": np.random.randn(16, 8).astype(np.float16),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(
            np.float16
        ),
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(8, 8).astype(
            np.float16
        ),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(16, 8).astype(
            np.float16
        ),
        "model.layers.0.mlp.down_proj.weight": np.random.randn(8, 16).astype(
            np.float16
        ),
        "model.layers.0.input_layernorm.weight": np.random.randn(8).astype(np.float16),
        "model.layers.1.self_attn.q_proj.weight": np.random.randn(8, 8).astype(
            np.float16
        ),
        "model.layers.1.self_attn.k_proj.weight": np.random.randn(8, 8).astype(
            np.float16
        ),
        "model.layers.1.mlp.gate_proj.weight": np.random.randn(16, 8).astype(
            np.float16
        ),
        "model.layers.1.mlp.down_proj.weight": np.random.randn(8, 16).astype(
            np.float16
        ),
        "model.layers.1.input_layernorm.weight": np.random.randn(8).astype(np.float16),
        "model.norm.weight": np.random.randn(8).astype(np.float16),
        "lm_head.weight": np.random.randn(16, 8).astype(np.float16),
        "model.layers.0.rotary_emb.inv_freq": np.random.randn(4).astype(np.float32),
    }
    metadata = {"format": "pt", "model_type": "llama"}
    path = tmp_path / "mini_model.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    return path


@pytest.fixture
def lora_base_model(tmp_path: Path) -> Path:
    """A base model compatible with lora_adapter fixture."""
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((8, 8), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.ones((8, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.ones((16, 8), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        "model.embed_tokens.weight": np.ones((16, 8), dtype=np.float32),
    }
    path = tmp_path / "base_model.safetensors"
    save_file(tensors, str(path))
    return path


@pytest.fixture
def finetuned_model(tmp_path: Path) -> Path:
    """Same structure as lora_base_model but with different values."""
    rng = np.random.RandomState(42)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": (
            np.ones((8, 8), dtype=np.float32) + 0.1 * rng.randn(8, 8).astype(np.float32)
        ),
        "model.layers.0.self_attn.v_proj.weight": (
            np.ones((8, 8), dtype=np.float32) + 0.1 * rng.randn(8, 8).astype(np.float32)
        ),
        "model.layers.0.mlp.gate_proj.weight": (
            np.ones((16, 8), dtype=np.float32)
            + 0.1 * rng.randn(16, 8).astype(np.float32)
        ),
        "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        "model.embed_tokens.weight": np.ones((16, 8), dtype=np.float32),
    }
    path = tmp_path / "finetuned_model.safetensors"
    save_file(tensors, str(path))
    return path


@pytest.fixture
def lora_adapter(tmp_path: Path) -> Path:
    """A LoRA adapter file with rank-4 A/B pairs for 2 modules."""
    rank = 4
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
            np.random.randn(rank, 8).astype(np.float32)
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (
            np.random.randn(8, rank).astype(np.float32)
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": (
            np.random.randn(rank, 8).astype(np.float32)
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": (
            np.random.randn(8, rank).astype(np.float32)
        ),
    }
    metadata = {
        "rank": "4",
        "alpha": "8",
        "target_modules": "q_proj,v_proj",
    }
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    return path


@pytest.fixture
def model_with_nans(tmp_path: Path) -> Path:
    """A model file containing NaN and Inf values."""
    w1 = np.random.randn(4, 4).astype(np.float32)
    w1[0, 0] = np.nan
    w2 = np.random.randn(4, 4).astype(np.float32)
    w2[1, 1] = np.inf
    w3 = np.random.randn(4, 4).astype(np.float32)
    tensors = {
        "has_nan.weight": w1,
        "has_inf.weight": w2,
        "clean.weight": w3,
    }
    path = tmp_path / "model_with_nans.safetensors"
    save_file(tensors, str(path))
    return path
