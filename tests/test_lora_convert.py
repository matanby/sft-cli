"""Tests for sft lora convert command (Kohya <-> PEFT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from typer.testing import CliRunner

from sft.cli import app
from sft.index import TensorIndex
from sft.ops.lora.convert import convert_lora, detect_format

runner = CliRunner()


@pytest.fixture
def kohya_adapter(tmp_path: Path) -> Path:
    """A Kohya-style LoRA file with 2 modules and explicit .alpha tensors."""
    rank = 4
    alpha = 8.0
    tensors = {
        "lora_unet_q.lora_down.weight": np.random.randn(rank, 8).astype(np.float32),
        "lora_unet_q.lora_up.weight": np.random.randn(8, rank).astype(np.float32),
        "lora_unet_q.alpha": np.array(alpha, dtype=np.float16),
        "lora_unet_v.lora_down.weight": np.random.randn(rank, 8).astype(np.float32),
        "lora_unet_v.lora_up.weight": np.random.randn(8, rank).astype(np.float32),
        "lora_unet_v.alpha": np.array(alpha, dtype=np.float16),
    }
    path = tmp_path / "kohya_adapter.safetensors"
    save_file(tensors, str(path), metadata={"ss_network_module": "lora"})
    return path


class TestDetectFormat:
    def test_detects_kohya(self, kohya_adapter: Path) -> None:
        info = detect_format(kohya_adapter)
        assert info.format == "kohya"
        assert info.kohya_modules == 2
        assert info.peft_modules == 0
        assert info.has_alpha_tensors == 2

    def test_detects_peft(self, lora_adapter: Path) -> None:
        info = detect_format(lora_adapter)
        assert info.format == "peft"
        assert info.peft_modules == 2
        assert info.kohya_modules == 0

    def test_detects_none(self, mini_model: Path) -> None:
        info = detect_format(mini_model)
        assert info.format is None
        assert info.kohya_modules == 0
        assert info.peft_modules == 0
        assert info.non_lora > 0


class TestKohyaToPeft:
    def test_renames_suffixes(self, kohya_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        convert_lora(kohya_adapter, dst, target="peft")
        out = load_file(str(dst))
        assert "lora_unet_q.lora_A.weight" in out
        assert "lora_unet_q.lora_B.weight" in out
        assert "lora_unet_v.lora_A.weight" in out
        assert "lora_unet_v.lora_B.weight" in out

    def test_drops_alpha_tensors(self, kohya_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        convert_lora(kohya_adapter, dst, target="peft")
        out = load_file(str(dst))
        assert "lora_unet_q.alpha" not in out
        assert "lora_unet_v.alpha" not in out
        assert not any(k.endswith(".lora_down.weight") for k in out)
        assert not any(k.endswith(".lora_up.weight") for k in out)

    def test_lifts_alpha_to_metadata(self, kohya_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        convert_lora(kohya_adapter, dst, target="peft")
        idx = TensorIndex.from_file(dst)
        assert idx.metadata["alpha"] == "8"
        assert idx.metadata["rank"] == "4"
        assert idx.metadata["converted_from"] == "kohya"

    def test_preserves_effective_delta(
        self, kohya_adapter: Path, tmp_path: Path
    ) -> None:
        """delta = B @ A should be identical before and after conversion."""
        dst = tmp_path / "out.safetensors"
        convert_lora(kohya_adapter, dst, target="peft")

        src = load_file(str(kohya_adapter))
        out = load_file(str(dst))
        for prefix in ("lora_unet_q", "lora_unet_v"):
            delta_in = (
                src[f"{prefix}.lora_up.weight"] @ src[f"{prefix}.lora_down.weight"]
            )
            delta_out = out[f"{prefix}.lora_B.weight"] @ out[f"{prefix}.lora_A.weight"]
            np.testing.assert_array_equal(delta_in, delta_out)


class TestPeftToKohya:
    def test_renames_suffixes(self, lora_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        convert_lora(lora_adapter, dst, target="kohya")
        out = load_file(str(dst))
        assert any(k.endswith(".lora_down.weight") for k in out)
        assert any(k.endswith(".lora_up.weight") for k in out)
        assert not any(k.endswith(".lora_A.weight") for k in out)
        assert not any(k.endswith(".lora_B.weight") for k in out)

    def test_emits_alpha_tensors(self, lora_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        convert_lora(lora_adapter, dst, target="kohya")
        out = load_file(str(dst))
        alphas = [k for k in out if k.endswith(".alpha")]
        assert len(alphas) == 2
        # Alpha from fixture metadata = 8
        for k in alphas:
            assert float(np.asarray(out[k]).flatten()[0]) == 8.0

    def test_emits_alpha_from_rank_when_metadata_missing(self, tmp_path: Path) -> None:
        """When PEFT metadata lacks alpha, default to alpha == rank."""
        rank = 4
        tensors = {
            "x.lora_A.weight": np.random.randn(rank, 8).astype(np.float32),
            "x.lora_B.weight": np.random.randn(8, rank).astype(np.float32),
        }
        src_path = tmp_path / "peft_no_meta.safetensors"
        save_file(tensors, str(src_path), metadata={})
        dst = tmp_path / "out.safetensors"
        convert_lora(src_path, dst, target="kohya")
        out = load_file(str(dst))
        assert float(np.asarray(out["x.alpha"]).flatten()[0]) == float(rank)

    def test_preserves_effective_delta(
        self, lora_adapter: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        convert_lora(lora_adapter, dst, target="kohya")
        src = load_file(str(lora_adapter))
        out = load_file(str(dst))

        for prefix_key in (
            "base_model.model.model.layers.0.self_attn.q_proj",
            "base_model.model.model.layers.0.self_attn.v_proj",
        ):
            delta_in = (
                src[f"{prefix_key}.lora_B.weight"] @ src[f"{prefix_key}.lora_A.weight"]
            )
            delta_out = (
                out[f"{prefix_key}.lora_up.weight"]
                @ out[f"{prefix_key}.lora_down.weight"]
            )
            np.testing.assert_array_equal(delta_in, delta_out)


class TestAutoDetect:
    def test_auto_kohya_to_peft(self, kohya_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        result = convert_lora(kohya_adapter, dst, target=None)
        assert result.source_format == "kohya"
        assert result.target_format == "peft"

    def test_auto_peft_to_kohya(self, lora_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        result = convert_lora(lora_adapter, dst, target=None)
        assert result.source_format == "peft"
        assert result.target_format == "kohya"

    def test_same_target_errors(self, lora_adapter: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        with pytest.raises(ValueError, match="already in peft"):
            convert_lora(lora_adapter, dst, target="peft")

    def test_no_lora_errors(self, mini_model: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.safetensors"
        with pytest.raises(ValueError, match="No LoRA tensors"):
            convert_lora(mini_model, dst, target=None)


class TestRoundTrip:
    def test_kohya_peft_kohya_preserves_delta(
        self, kohya_adapter: Path, tmp_path: Path
    ) -> None:
        peft_path = tmp_path / "as_peft.safetensors"
        back_path = tmp_path / "back_to_kohya.safetensors"
        convert_lora(kohya_adapter, peft_path, target="peft")
        convert_lora(peft_path, back_path, target="kohya")

        src = load_file(str(kohya_adapter))
        out = load_file(str(back_path))
        for prefix in ("lora_unet_q", "lora_unet_v"):
            delta_in = (
                src[f"{prefix}.lora_up.weight"] @ src[f"{prefix}.lora_down.weight"]
            )
            delta_out = (
                out[f"{prefix}.lora_up.weight"] @ out[f"{prefix}.lora_down.weight"]
            )
            np.testing.assert_array_equal(delta_in, delta_out)


class TestCli:
    def test_convert_cli_kohya_to_peft(
        self, kohya_adapter: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app,
            ["lora", "convert", str(kohya_adapter), "-o", str(dst)],
        )
        assert result.exit_code == 0, result.output
        assert "kohya" in result.output
        assert "peft" in result.output
        assert dst.exists()

    def test_convert_cli_explicit_target(
        self, kohya_adapter: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app,
            ["lora", "convert", str(kohya_adapter), "--to", "peft", "-o", str(dst)],
        )
        assert result.exit_code == 0, result.output

    def test_convert_cli_rejects_non_lora(
        self, mini_model: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["lora", "convert", str(mini_model), "-o", str(tmp_path / "x.safetensors")],
        )
        assert result.exit_code == 1
        assert "no lora tensors" in result.output.lower()

    def test_convert_cli_rejects_invalid_target(self, kohya_adapter: Path) -> None:
        result = runner.invoke(
            app,
            ["lora", "convert", str(kohya_adapter), "--to", "garbage"],
        )
        assert result.exit_code == 1

    def test_convert_cli_smart_output(self, kohya_adapter: Path) -> None:
        """When -o is omitted, output goes next to source as <stem>.peft.safetensors."""
        result = runner.invoke(app, ["lora", "convert", str(kohya_adapter)])
        assert result.exit_code == 0, result.output
        expected = kohya_adapter.parent / "kohya_adapter.peft.safetensors"
        assert expected.exists()
