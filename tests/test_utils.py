"""Tests for shared utility modules."""

from __future__ import annotations

from pathlib import Path

from sft.utils.formatting import (
    format_bytes,
    format_dtype,
    format_number,
    format_shape,
)
from sft.utils.glob import filter_tensors, tensor_matches
from sft.utils.output import default_output, resolve_output


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(512) == "512 B"

    def test_kilobytes(self):
        assert format_bytes(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_bytes(1_500_000) == "1.4 MB"

    def test_gigabytes(self):
        assert format_bytes(2_500_000_000) == "2.33 GB"

    def test_zero(self):
        assert format_bytes(0) == "0 B"


class TestFormatNumber:
    def test_small(self):
        assert format_number(999) == "999"

    def test_thousands(self):
        assert format_number(1_500) == "1.5K"

    def test_millions(self):
        assert format_number(6_738_415) == "6.7M"

    def test_billions(self):
        assert format_number(6_738_415_616) == "6.7B"


class TestFormatShape:
    def test_scalar(self):
        assert format_shape(()) == "()"

    def test_vector(self):
        assert format_shape((4096,)) == "(4096,)"

    def test_matrix(self):
        assert format_shape((4096, 4096)) == "(4096, 4096)"


class TestFormatDtype:
    def test_float16(self):
        assert format_dtype("F16") == "FP16"

    def test_bfloat16(self):
        assert format_dtype("BF16") == "BF16"

    def test_float32(self):
        assert format_dtype("F32") == "FP32"

    def test_passthrough(self):
        assert format_dtype("I32") == "INT32"

    def test_unknown(self):
        assert format_dtype("WEIRD") == "WEIRD"


class TestTensorGlob:
    def test_wildcard_single_segment(self):
        assert tensor_matches("model.layers.0.weight", "model.layers.*.weight")

    def test_wildcard_no_match(self):
        assert not tensor_matches("model.layers.0.bias", "model.layers.*.weight")

    def test_double_star(self):
        assert tensor_matches("model.layers.0.self_attn.q_proj.weight", "**.weight")

    def test_exact_match(self):
        assert tensor_matches("lm_head.weight", "lm_head.weight")

    def test_star_matches_single_segment(self):
        assert tensor_matches("model.layers.0.weight", "model.layers.*.weight")
        assert not tensor_matches(
            "model.layers.0.extra.weight", "model.layers.*.weight"
        )

    def test_star_at_end(self):
        assert tensor_matches("model.layers.0.weight", "model.layers.0.*")
        assert not tensor_matches("model.layers.0.self_attn.weight", "model.layers.0.*")

    def test_double_star_matches_deep(self):
        assert tensor_matches(
            "model.layers.0.self_attn.q_proj.weight", "model.layers.0.**"
        )

    def test_no_cross_segment(self):
        assert not tensor_matches(
            "model.layers.0.self_attn.q_proj.weight",
            "model.*.weight",
        )


class TestFilterTensors:
    def test_include_only(self):
        names = [
            "model.layers.0.weight",
            "model.layers.0.bias",
            "lm_head.weight",
        ]
        result = filter_tensors(names, include="model.layers.**")
        assert result == ["model.layers.0.weight", "model.layers.0.bias"]

    def test_exclude_only(self):
        names = [
            "model.layers.0.weight",
            "model.layers.0.bias",
            "lm_head.weight",
        ]
        result = filter_tensors(names, exclude="**.bias")
        assert result == ["model.layers.0.weight", "lm_head.weight"]

    def test_include_and_exclude(self):
        names = [
            "model.layers.0.weight",
            "model.layers.0.bias",
            "model.layers.1.weight",
            "lm_head.weight",
        ]
        result = filter_tensors(names, include="model.layers.**", exclude="**.bias")
        assert result == ["model.layers.0.weight", "model.layers.1.weight"]

    def test_no_filters(self):
        names = ["a", "b", "c"]
        assert filter_tensors(names) == names


class TestOutput:
    def test_default_output(self, tmp_path: Path):
        inp = tmp_path / "model.safetensors"
        result = default_output(inp, "bf16")
        assert result == tmp_path / "model.bf16.safetensors"

    def test_resolve_explicit(self, tmp_path: Path):
        inp = tmp_path / "model.safetensors"
        explicit = tmp_path / "custom.safetensors"
        result = resolve_output(explicit, inp, "bf16")
        assert result == explicit

    def test_resolve_default(self, tmp_path: Path):
        inp = tmp_path / "model.safetensors"
        result = resolve_output(None, inp, "merged")
        assert result == tmp_path / "model.merged.safetensors"
