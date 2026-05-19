"""Tests for LoRA detection utilities."""

from __future__ import annotations

from pathlib import Path

from sft.ops.lora.detect import (
    detect_lora,
    extract_target_module,
    format_lora_module_display,
    get_base_weight_name,
    strip_peft_prefix,
)


def test_detect_lora_adapter(lora_adapter: Path):
    info = detect_lora(lora_adapter)
    assert info is not None
    assert info.rank == 4
    assert info.alpha == 8.0
    assert info.effective_scale == 2.0
    assert len(info.pairs) == 2
    assert set(info.target_modules) == {"q_proj", "v_proj"}
    assert info.total_params > 0


def test_detect_lora_non_lora(mini_model: Path):
    info = detect_lora(mini_model)
    assert info is None


def test_detect_lora_pair_shapes(lora_adapter: Path):
    info = detect_lora(lora_adapter)
    assert info is not None
    for pair in info.pairs:
        assert pair.lora_a_shape[0] == 4  # rank
        assert pair.lora_b_shape[1] == 4  # rank


def test_strip_peft_prefix():
    assert (
        strip_peft_prefix("base_model.model.model.layers.0.self_attn.q_proj")
        == "model.layers.0.self_attn.q_proj"
    )
    assert strip_peft_prefix("base_model.model.layers.0.q_proj") == "layers.0.q_proj"
    assert strip_peft_prefix("no_prefix.weight") == "no_prefix.weight"


def test_extract_target_module():
    assert (
        extract_target_module("base_model.model.layers.0.self_attn.q_proj") == "q_proj"
    )
    assert extract_target_module("layers.0.mlp.gate_proj") == "gate_proj"


def test_format_lora_module_display():
    assert (
        format_lora_module_display("base_model.model.model.layers.0.self_attn.q_proj")
        == "model.layers.0.self_attn.q_proj"
    )
    assert (
        format_lora_module_display("transformer.single_transformer_blocks.0.attn.to_k")
        == "transformer.single_transformer_blocks.0.attn.to_k"
    )
    assert (
        format_lora_module_display(
            "diffusion_model.transformer_blocks.0.attn1.to_out.0"
        )
        == "diffusion_model.transformer_blocks.0.attn1.to_out.0"
    )
    assert (
        format_lora_module_display("transformer.single_transformer_blocks.0")
        == "transformer.single_transformer_blocks.0"
    )
    assert (
        format_lora_module_display(
            "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k"
        )
        == "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k"
    )


def test_get_base_weight_name():
    name = get_base_weight_name("base_model.model.model.layers.0.self_attn.q_proj")
    assert name == "model.layers.0.self_attn.q_proj.weight"


def test_detect_lora_num_layers(lora_adapter: Path):
    info = detect_lora(lora_adapter)
    assert info is not None
    assert info.num_layers == 1  # fixture has only layer 0
