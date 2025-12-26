#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "safetensors",
# ]
# ///
"""Create a test safetensors file for development."""

import numpy as np
from safetensors.numpy import save_file

# Create a variety of tensors with different shapes and dtypes
tensors = {
    # Simulating a transformer-like structure
    "model.embed_tokens.weight": np.random.randn(32000, 4096).astype(np.float16),
    "model.layers.0.self_attn.q_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.0.self_attn.k_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.0.self_attn.v_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.0.self_attn.o_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.0.mlp.gate_proj.weight": np.random.randn(11008, 4096).astype(
        np.float16
    ),
    "model.layers.0.mlp.up_proj.weight": np.random.randn(11008, 4096).astype(
        np.float16
    ),
    "model.layers.0.mlp.down_proj.weight": np.random.randn(4096, 11008).astype(
        np.float16
    ),
    "model.layers.0.input_layernorm.weight": np.random.randn(4096).astype(np.float16),
    "model.layers.0.post_attention_layernorm.weight": np.random.randn(4096).astype(
        np.float16
    ),
    "model.layers.1.self_attn.q_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.1.self_attn.k_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.1.self_attn.v_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.1.self_attn.o_proj.weight": np.random.randn(4096, 4096).astype(
        np.float16
    ),
    "model.layers.1.mlp.gate_proj.weight": np.random.randn(11008, 4096).astype(
        np.float16
    ),
    "model.layers.1.mlp.up_proj.weight": np.random.randn(11008, 4096).astype(
        np.float16
    ),
    "model.layers.1.mlp.down_proj.weight": np.random.randn(4096, 11008).astype(
        np.float16
    ),
    "model.layers.1.input_layernorm.weight": np.random.randn(4096).astype(np.float16),
    "model.layers.1.post_attention_layernorm.weight": np.random.randn(4096).astype(
        np.float16
    ),
    "model.norm.weight": np.random.randn(4096).astype(np.float16),
    "lm_head.weight": np.random.randn(32000, 4096).astype(np.float16),
}

# Add some float32 tensors for dtype variety
tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = np.random.randn(64).astype(
    np.float32
)
tensors["model.layers.1.self_attn.rotary_emb.inv_freq"] = np.random.randn(64).astype(
    np.float32
)

metadata = {
    "format": "pt",
    "framework": "pytorch",
    "model_type": "llama",
}

save_file(tensors, "test_model.safetensors", metadata=metadata)
print(f"Created test_model.safetensors with {len(tensors)} tensors")
