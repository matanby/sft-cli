"""Generate a set of dummy .safetensors files for local CLI testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_models"


def _make_small_transformer() -> None:
    """A small 4-layer transformer (~1.5M params), mixed dtypes, with metadata."""
    hidden = 256
    intermediate = 512
    vocab = 1024
    n_layers = 4

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.random.randn(vocab, hidden).astype(np.float16),
    }

    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        tensors.update(
            {
                f"{prefix}.self_attn.q_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.self_attn.k_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.self_attn.v_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.self_attn.o_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.mlp.gate_proj.weight": np.random.randn(
                    intermediate, hidden
                ).astype(np.float16),
                f"{prefix}.mlp.up_proj.weight": np.random.randn(
                    intermediate, hidden
                ).astype(np.float16),
                f"{prefix}.mlp.down_proj.weight": np.random.randn(
                    hidden, intermediate
                ).astype(np.float16),
                f"{prefix}.input_layernorm.weight": np.ones(hidden).astype(np.float32),
                f"{prefix}.post_attention_layernorm.weight": np.ones(hidden).astype(
                    np.float32
                ),
            }
        )

    tensors["model.norm.weight"] = np.ones(hidden).astype(np.float32)
    tensors["lm_head.weight"] = np.random.randn(vocab, hidden).astype(np.float16)

    metadata = {"format": "pt", "model_type": "llama", "quantization": "none"}
    path = OUTPUT_DIR / "small_transformer.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    print(
        f"  Created {path.name} ({path.stat().st_size / 1024:.1f} KB, {len(tensors)} tensors)"
    )


def _make_sharded_model() -> None:
    """A model split across 3 shard files, like HuggingFace large models."""
    hidden = 512
    intermediate = 1024
    vocab = 2048
    n_layers = 6

    shard_dir = OUTPUT_DIR / "sharded_model"
    shard_dir.mkdir(exist_ok=True)

    all_tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.random.randn(vocab, hidden).astype(np.float16),
    }
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        all_tensors.update(
            {
                f"{prefix}.self_attn.q_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.self_attn.k_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.self_attn.v_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.self_attn.o_proj.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float16),
                f"{prefix}.mlp.gate_proj.weight": np.random.randn(
                    intermediate, hidden
                ).astype(np.float16),
                f"{prefix}.mlp.up_proj.weight": np.random.randn(
                    intermediate, hidden
                ).astype(np.float16),
                f"{prefix}.mlp.down_proj.weight": np.random.randn(
                    hidden, intermediate
                ).astype(np.float16),
                f"{prefix}.input_layernorm.weight": np.ones(hidden).astype(np.float32),
                f"{prefix}.post_attention_layernorm.weight": np.ones(hidden).astype(
                    np.float32
                ),
            }
        )
    all_tensors["model.norm.weight"] = np.ones(hidden).astype(np.float32)
    all_tensors["lm_head.weight"] = np.random.randn(vocab, hidden).astype(np.float16)

    keys = list(all_tensors.keys())
    n_shards = 3
    chunk_size = len(keys) // n_shards + 1

    for shard_idx in range(n_shards):
        shard_keys = keys[shard_idx * chunk_size : (shard_idx + 1) * chunk_size]
        shard_tensors = {k: all_tensors[k] for k in shard_keys}
        if not shard_tensors:
            continue
        fname = f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.safetensors"
        path = shard_dir / fname
        save_file(shard_tensors, str(path))
        print(
            f"  Created {shard_dir.name}/{fname} ({path.stat().st_size / 1024:.1f} KB, {len(shard_tensors)} tensors)"
        )


def _make_lora_adapter() -> None:
    """A LoRA adapter with rank-16 A/B matrices for attention projections."""
    rank = 16
    hidden = 256
    n_layers = 4

    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = f"base_model.model.model.layers.{i}.self_attn.{proj}"
            tensors[f"{prefix}.lora_A.weight"] = np.random.randn(rank, hidden).astype(
                np.float32
            )
            tensors[f"{prefix}.lora_B.weight"] = np.random.randn(hidden, rank).astype(
                np.float32
            )

    metadata = {
        "rank": "16",
        "alpha": "32",
        "target_modules": "q_proj,k_proj,v_proj,o_proj",
        "peft_type": "LORA",
    }
    path = OUTPUT_DIR / "lora_adapter.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    print(
        f"  Created {path.name} ({path.stat().st_size / 1024:.1f} KB, {len(tensors)} tensors)"
    )


def _make_quantized_model() -> None:
    """Simulates a quantized model with uint8 weights and fp32 scales/zeros."""
    hidden = 256
    intermediate = 512
    n_layers = 4

    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        for proj in ("q_proj", "v_proj", "k_proj", "o_proj"):
            tensors[f"{prefix}.self_attn.{proj}.weight"] = np.random.randint(
                0, 255, (hidden, hidden), dtype=np.uint8
            )
            tensors[f"{prefix}.self_attn.{proj}.weight_scale"] = np.random.randn(
                hidden
            ).astype(np.float32)
            tensors[f"{prefix}.self_attn.{proj}.weight_zero_point"] = np.zeros(
                hidden, dtype=np.float32
            )

        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randint(
            0, 255, (intermediate, hidden), dtype=np.uint8
        )
        tensors[f"{prefix}.mlp.gate_proj.weight_scale"] = np.random.randn(
            intermediate
        ).astype(np.float32)
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hidden).astype(np.float16)

    metadata = {"quantization": "int8", "format": "pt"}
    path = OUTPUT_DIR / "quantized_model.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    print(
        f"  Created {path.name} ({path.stat().st_size / 1024:.1f} KB, {len(tensors)} tensors)"
    )


def _make_embedding_model() -> None:
    """A simple embedding/encoder model (BERT-like structure)."""
    hidden = 384
    vocab = 30522
    n_layers = 6

    tensors: dict[str, np.ndarray] = {
        "embeddings.word_embeddings.weight": np.random.randn(vocab, hidden).astype(
            np.float32
        ),
        "embeddings.position_embeddings.weight": np.random.randn(512, hidden).astype(
            np.float32
        ),
        "embeddings.token_type_embeddings.weight": np.random.randn(2, hidden).astype(
            np.float32
        ),
        "embeddings.LayerNorm.weight": np.ones(hidden).astype(np.float32),
        "embeddings.LayerNorm.bias": np.zeros(hidden).astype(np.float32),
    }

    for i in range(n_layers):
        prefix = f"encoder.layer.{i}"
        tensors.update(
            {
                f"{prefix}.attention.self.query.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float32),
                f"{prefix}.attention.self.query.bias": np.zeros(hidden).astype(
                    np.float32
                ),
                f"{prefix}.attention.self.key.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float32),
                f"{prefix}.attention.self.key.bias": np.zeros(hidden).astype(
                    np.float32
                ),
                f"{prefix}.attention.self.value.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float32),
                f"{prefix}.attention.self.value.bias": np.zeros(hidden).astype(
                    np.float32
                ),
                f"{prefix}.attention.output.dense.weight": np.random.randn(
                    hidden, hidden
                ).astype(np.float32),
                f"{prefix}.attention.output.dense.bias": np.zeros(hidden).astype(
                    np.float32
                ),
                f"{prefix}.attention.output.LayerNorm.weight": np.ones(hidden).astype(
                    np.float32
                ),
                f"{prefix}.attention.output.LayerNorm.bias": np.zeros(hidden).astype(
                    np.float32
                ),
                f"{prefix}.intermediate.dense.weight": np.random.randn(
                    hidden * 4, hidden
                ).astype(np.float32),
                f"{prefix}.intermediate.dense.bias": np.zeros(hidden * 4).astype(
                    np.float32
                ),
                f"{prefix}.output.dense.weight": np.random.randn(
                    hidden, hidden * 4
                ).astype(np.float32),
                f"{prefix}.output.dense.bias": np.zeros(hidden).astype(np.float32),
                f"{prefix}.output.LayerNorm.weight": np.ones(hidden).astype(np.float32),
                f"{prefix}.output.LayerNorm.bias": np.zeros(hidden).astype(np.float32),
            }
        )

    tensors["pooler.dense.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
    tensors["pooler.dense.bias"] = np.zeros(hidden).astype(np.float32)

    metadata = {"model_type": "bert", "format": "pt"}
    path = OUTPUT_DIR / "bert_base.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    print(
        f"  Created {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB, {len(tensors)} tensors)"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Generating test models in {OUTPUT_DIR}/\n")

    print("[1/5] Small transformer (4 layers, fp16+fp32 mixed)...")
    _make_small_transformer()

    print("[2/5] Sharded model (6 layers across 3 files)...")
    _make_sharded_model()

    print("[3/5] LoRA adapter (rank-16, 4 layers)...")
    _make_lora_adapter()

    print("[4/5] Quantized model (int8 weights + fp32 scales)...")
    _make_quantized_model()

    print("[5/5] BERT-like embedding model (6 layers, fp32)...")
    _make_embedding_model()

    print(f"\nDone! All test files are in {OUTPUT_DIR}/")
    print("\nTry these commands:")
    print(f"  sft info {OUTPUT_DIR}/small_transformer.safetensors")
    print(f"  sft info --json {OUTPUT_DIR}/lora_adapter.safetensors")
    print(f"  sft ls {OUTPUT_DIR}/sharded_model/")
    print(f"  sft browse {OUTPUT_DIR}/bert_base.safetensors")
    print(f"  sft check {OUTPUT_DIR}/quantized_model.safetensors")


if __name__ == "__main__":
    main()
