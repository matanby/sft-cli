"""Tests for sft lora resize command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from sft.ops.lora.resize import resize_lora


def test_resize_basic(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)

    assert result.original_rank == 4
    assert result.new_rank == 2
    assert result.modules_resized == 2
    assert dst.exists()


def test_resize_shapes(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    resize_lora(lora_adapter, dst, target_rank=2)

    tensors = load_file(str(dst))
    for name, t in tensors.items():
        if "lora_A" in name:
            assert t.shape[0] == 2, f"{name} rank dim should be 2, got {t.shape[0]}"
            assert t.shape[1] == 8, f"{name} in_features should be 8, got {t.shape[1]}"
        elif "lora_B" in name:
            assert t.shape[0] == 8, f"{name} out_features should be 8, got {t.shape[0]}"
            assert t.shape[1] == 2, f"{name} rank dim should be 2, got {t.shape[1]}"


def test_resize_reconstruction(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)

    old = load_file(str(lora_adapter))
    new = load_file(str(dst))

    for pair_key in [
        "base_model.model.model.layers.0.self_attn.q_proj",
        "base_model.model.model.layers.0.self_attn.v_proj",
    ]:
        a_name = f"{pair_key}.lora_A.weight"
        b_name = f"{pair_key}.lora_B.weight"

        old_delta = old[b_name] @ old[a_name]
        new_delta = new[b_name] @ new[a_name]

        error = np.linalg.norm(old_delta - new_delta, "fro") / np.linalg.norm(
            old_delta, "fro"
        )
        assert error < 1.0, f"Reconstruction error too large: {error}"

    for err in result.errors.values():
        assert err < 1.0


def test_resize_smart_output(lora_adapter: Path):
    from sft.utils.output import resolve_output

    dst = resolve_output(None, lora_adapter, "r2")
    assert dst.name == "adapter.r2.safetensors"


def test_resize_error_if_rank_too_large(lora_adapter: Path):
    import pytest

    with pytest.raises(ValueError, match="must be less than current rank"):
        resize_lora(lora_adapter, None, target_rank=8)

    with pytest.raises(ValueError, match="must be less than current rank"):
        resize_lora(lora_adapter, None, target_rank=4)


def test_resize_metadata_updated(lora_adapter: Path, tmp_path: Path):
    dst = tmp_path / "out.safetensors"
    resize_lora(lora_adapter, dst, target_rank=2)

    from sft.index import TensorIndex

    index = TensorIndex.from_file(dst)
    assert index.metadata["rank"] == "2"
    assert index.metadata["alpha"] == "8"


def test_resize_error_matches_energy(lora_adapter: Path, tmp_path: Path):
    """Reported error is consistent with the SVD energy not retained."""
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)

    old = load_file(str(lora_adapter))
    new = load_file(str(dst))

    for pair_key in [
        "base_model.model.model.layers.0.self_attn.q_proj",
        "base_model.model.model.layers.0.self_attn.v_proj",
    ]:
        a_name = f"{pair_key}.lora_A.weight"
        b_name = f"{pair_key}.lora_B.weight"
        target = pair_key.split(".")[-1]

        old_delta = old[b_name].astype(np.float64) @ old[a_name].astype(np.float64)
        new_delta = new[b_name].astype(np.float64) @ new[a_name].astype(np.float64)

        actual_error = np.linalg.norm(old_delta - new_delta, "fro") / np.linalg.norm(
            old_delta, "fro"
        )
        reported_error = result.errors[target]

        np.testing.assert_allclose(actual_error, reported_error, rtol=0.05)


def test_resize_energies_reported(lora_adapter: Path, tmp_path: Path):
    """`ResizeResult.energies` reports per-pair fraction of energy retained."""
    dst = tmp_path / "out.safetensors"
    result = resize_lora(lora_adapter, dst, target_rank=2)
    assert result.energies
    for module, energy in result.energies.items():
        assert 0.0 <= energy <= 1.0
        # error and energy must be consistent: error = sqrt(1 - energy)
        expected_error = float(np.sqrt(max(0.0, 1.0 - energy)))
        np.testing.assert_allclose(result.errors[module], expected_error, atol=1e-9)


def test_resize_requires_exactly_one_mode(lora_adapter: Path):
    """Must specify exactly one of target_rank or auto_margin."""
    import pytest

    with pytest.raises(ValueError, match="Exactly one"):
        resize_lora(lora_adapter, None, target_rank=None, auto_margin=None)
    with pytest.raises(ValueError, match="Exactly one"):
        resize_lora(lora_adapter, None, target_rank=2, auto_margin=1)


# --- Auto rank mode ---


class TestAutoRank:
    def test_auto_basic(self, lora_adapter: Path, tmp_path: Path):
        """auto_margin=1 produces a per-pair rank dict, each <= original rank."""
        dst = tmp_path / "out.safetensors"
        result = resize_lora(lora_adapter, dst, auto_margin=1)

        assert result.auto_margin == 1
        assert result.per_module_ranks
        for rank in result.per_module_ranks.values():
            assert 1 <= rank <= 4  # original rank in fixture is 4

    def test_auto_writes_heterogeneous_shapes(self, lora_adapter: Path, tmp_path: Path):
        """Output file has each pair sized to its own per-module rank."""
        dst = tmp_path / "out.safetensors"
        result = resize_lora(lora_adapter, dst, auto_margin=1)
        tensors = load_file(str(dst))
        for module, rank in result.per_module_ranks.items():
            # Find the A/B pair for this module in the output
            a_name = next(k for k in tensors if k.endswith(f"{module}.lora_A.weight"))
            b_name = next(k for k in tensors if k.endswith(f"{module}.lora_B.weight"))
            assert tensors[a_name].shape[0] == rank
            assert tensors[b_name].shape[1] == rank

    def test_auto_margin_increases_rank(self, lora_adapter: Path, tmp_path: Path):
        """Larger margin produces at-least-as-large per-pair ranks."""
        dst1 = tmp_path / "out1.safetensors"
        dst2 = tmp_path / "out2.safetensors"
        r1 = resize_lora(lora_adapter, dst1, auto_margin=1)
        r2 = resize_lora(lora_adapter, dst2, auto_margin=2)
        for module in r1.per_module_ranks:
            assert r2.per_module_ranks[module] >= r1.per_module_ranks[module]

    def test_auto_caps_at_original_rank(self, lora_adapter: Path, tmp_path: Path):
        """Per-pair rank can never exceed the input pair's rank."""
        dst = tmp_path / "out.safetensors"
        result = resize_lora(lora_adapter, dst, auto_margin=100)
        for rank in result.per_module_ranks.values():
            assert rank <= 4  # original

    def test_auto_negative_margin_rejected(self, lora_adapter: Path, tmp_path: Path):
        import pytest

        with pytest.raises(ValueError, match="auto_margin"):
            resize_lora(lora_adapter, tmp_path / "x.safetensors", auto_margin=-1)


# --- CLI rank parsing ---


class TestCliRankSpec:
    def test_parse_integer(self):
        from sft.commands.lora import _parse_rank_spec

        assert _parse_rank_spec("8") == (8, None)
        assert _parse_rank_spec("  16  ") == (16, None)

    def test_parse_auto(self):
        from sft.commands.lora import _parse_rank_spec

        assert _parse_rank_spec("auto") == (None, 1)
        assert _parse_rank_spec("AUTO") == (None, 1)

    def test_parse_auto_plus(self):
        from sft.commands.lora import _parse_rank_spec

        assert _parse_rank_spec("auto+0") == (None, 0)
        assert _parse_rank_spec("auto+3") == (None, 3)

    def test_parse_invalid(self):
        import pytest

        from sft.commands.lora import _parse_rank_spec

        with pytest.raises(ValueError):
            _parse_rank_spec("auto+")
        with pytest.raises(ValueError):
            _parse_rank_spec("auto+xyz")
        with pytest.raises(ValueError):
            _parse_rank_spec("0")
        with pytest.raises(ValueError):
            _parse_rank_spec("not-a-rank")
        with pytest.raises(ValueError):
            _parse_rank_spec("auto+-1")

    def test_cli_auto_writes_file(self, lora_adapter: Path, tmp_path: Path):
        from typer.testing import CliRunner

        from sft.cli import app

        runner = CliRunner()
        dst = tmp_path / "out.safetensors"
        result = runner.invoke(
            app,
            ["lora", "resize", str(lora_adapter), "--rank", "auto", "-o", str(dst)],
        )
        assert result.exit_code == 0, result.output
        assert dst.exists()
        assert "auto" in result.output.lower()
        assert "energy retained" in result.output.lower()

    def test_cli_auto_smart_output(self, lora_adapter: Path):
        from typer.testing import CliRunner

        from sft.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["lora", "resize", str(lora_adapter), "--rank", "auto"]
        )
        assert result.exit_code == 0, result.output
        expected = lora_adapter.parent / "adapter.rauto.safetensors"
        assert expected.exists()
