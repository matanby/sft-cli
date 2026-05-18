"""Smoke tests for TUI browser features."""

from __future__ import annotations

from pathlib import Path


def test_app_creates(mini_model: Path) -> None:
    from sft.browser import SftApp

    app = SftApp(mini_model)
    assert app.file_path == mini_model


def test_app_creates_lora(lora_adapter: Path) -> None:
    from sft.browser import SftApp

    app = SftApp(lora_adapter)
    assert app.file_path == lora_adapter


def test_stats_screen_instantiates(mini_model: Path) -> None:
    from sft.browser import TensorStatsScreen
    from sft.index import TensorIndex

    index = TensorIndex.from_file(mini_model)
    tensor = index.tensors[0]
    screen = TensorStatsScreen(tensor, mini_model)
    assert screen.tensor is tensor
    assert screen.file_path == mini_model


def test_lora_header_text(lora_adapter: Path) -> None:
    from sft.browser import SftApp
    from sft.ops.lora.detect import detect_lora

    app = SftApp(lora_adapter)
    app.lora_info = detect_lora(lora_adapter)
    text = app._lora_header_text()
    assert "LoRA Adapter" in text
    assert "Rank: 4" in text
    assert "Alpha: 8" in text
    assert "q_proj" in text
    assert "v_proj" in text


def test_lora_header_not_shown_for_regular_model(mini_model: Path) -> None:
    from sft.browser import SftApp
    from sft.ops.lora.detect import detect_lora

    info = detect_lora(mini_model)
    assert info is None

    app = SftApp(mini_model)
    assert app.lora_info is None


def test_stats_binding_present() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "S" in binding_keys


def test_cast_screen_creates() -> None:
    from sft.browser import CastScreen

    screen = CastScreen(Path("/tmp/test.safetensors"))
    assert screen.file_path == Path("/tmp/test.safetensors")


def test_cast_binding_exists() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "c" in binding_keys


def test_cast_screen_dtypes() -> None:
    from sft.browser import CastScreen

    assert "fp16" in CastScreen.DTYPES
    assert "fp32" in CastScreen.DTYPES
    assert "bf16" in CastScreen.DTYPES


def test_command_palette_provider() -> None:
    """Verify the command provider can be instantiated."""
    import pytest

    from sft.browser import _HAS_COMMAND_PALETTE

    if not _HAS_COMMAND_PALETTE:
        pytest.skip("Command palette not available in this Textual version")

    from sft.browser import SftCommands

    assert SftCommands is not None


def test_colon_binding() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "colon" in binding_keys


def test_commands_registered() -> None:
    from sft.browser import _HAS_COMMAND_PALETTE, SftApp

    if _HAS_COMMAND_PALETTE:
        from sft.browser import SftCommands

        assert SftCommands in SftApp.COMMANDS
    else:
        assert not SftApp.COMMANDS


def test_do_check_method(mini_model: Path) -> None:
    from sft.browser import SftApp

    app = SftApp(mini_model)
    assert hasattr(app, "_do_check_file")


def test_do_cast_methods(mini_model: Path) -> None:
    from sft.browser import SftApp

    app = SftApp(mini_model)
    for method in ("_do_cast_fp16", "_do_cast_fp32", "_do_cast_bf16"):
        assert hasattr(app, method)


def test_do_show_info_method(mini_model: Path) -> None:
    from sft.browser import SftApp

    app = SftApp(mini_model)
    assert hasattr(app, "_do_show_info")
