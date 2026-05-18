"""Smoke tests for TUI browser features."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


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


# --- New TUI feature tests ---


def test_lora_binding_exists() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "l" in binding_keys


def test_kohya_binding_exists() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "k" in binding_keys


def test_diff_binding_exists() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "D" in binding_keys


def test_tensor_detail_screen_with_lora_info(lora_adapter: Path) -> None:
    """LoRA pair info should be passable to TensorDetailScreen."""
    from sft.browser import TensorDetailScreen
    from sft.index import TensorIndex
    from sft.ops.lora.detect import detect_lora

    index = TensorIndex.from_file(lora_adapter)
    info = detect_lora(lora_adapter)
    assert info is not None
    pair = info.pairs[0]
    a_tensor = next(t for t in index.tensors if t.full_name == pair.lora_a_name)
    screen = TensorDetailScreen(a_tensor, pair, "A")
    assert screen.lora_pair is pair
    assert screen.lora_role == "A"


def test_lora_pair_map_built_on_compose(lora_adapter: Path) -> None:
    """The app should populate _lora_pair_map after composing."""
    from sft.browser import SftApp

    app = SftApp(lora_adapter)

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.lora_info is not None
            assert app._lora_pair_map
            for pair in app.lora_info.pairs:
                assert pair.lora_a_name in app._lora_pair_map
                assert pair.lora_b_name in app._lora_pair_map

    asyncio.run(_run())


def test_lora_analysis_screen_opens(lora_adapter: Path) -> None:
    """Pressing `l` on a LoRA file opens the LoRA Analysis screen."""
    from sft.browser import LoraAnalysisScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("l")
            await pilot.pause()
            assert isinstance(app.screen, LoraAnalysisScreen)
            # Background worker should fill stats for all pairs
            for _ in range(30):
                await pilot.pause()
                if len(app.screen._stats) == len(app.screen.lora_info.pairs):
                    break
            assert len(app.screen._stats) == len(app.screen.lora_info.pairs)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, LoraAnalysisScreen)

    asyncio.run(_run())


def test_lora_analysis_screen_skipped_for_non_lora(mini_model: Path) -> None:
    """Pressing `l` on a non-LoRA file shows a notification, does not open the screen."""
    from sft.browser import LoraAnalysisScreen, SftApp

    async def _run() -> None:
        app = SftApp(mini_model)
        async with app.run_test() as pilot:
            await pilot.press("l")
            await pilot.pause()
            assert not isinstance(app.screen, LoraAnalysisScreen)

    asyncio.run(_run())


def test_svd_spectrum_drill_down(lora_adapter: Path) -> None:
    """Pressing Enter on a pair in LoraAnalysisScreen opens the spectrum view."""
    from sft.browser import LoraAnalysisScreen, SftApp, SvdSpectrumScreen

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("l")
            for _ in range(30):
                await pilot.pause()
                if isinstance(app.screen, LoraAnalysisScreen) and len(
                    app.screen._stats
                ) == len(app.screen.lora_info.pairs):
                    break
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SvdSpectrumScreen)

    asyncio.run(_run())


def test_kohya_screen_detects_kohya(tmp_path: Path) -> None:
    """Pressing `k` on a Kohya file opens the convert dialog with detected direction."""
    from sft.browser import KohyaConvertScreen, SftApp

    rank = 4
    p = tmp_path / "kohya.safetensors"
    tensors = {
        "lora_unet_q.lora_down.weight": np.random.randn(rank, 8).astype(np.float32),
        "lora_unet_q.lora_up.weight": np.random.randn(8, rank).astype(np.float32),
        "lora_unet_q.alpha": np.array(8.0, dtype=np.float16),
    }
    save_file(tensors, str(p))

    async def _run() -> None:
        app = SftApp(p)
        async with app.run_test() as pilot:
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, KohyaConvertScreen)
            assert app.screen.source == "kohya"
            assert app.screen.target == "peft"

    asyncio.run(_run())


def test_kohya_screen_writes_file(tmp_path: Path) -> None:
    """Confirming Kohya conversion produces the expected output file."""
    from sft.browser import SftApp

    rank = 4
    p = tmp_path / "kohya.safetensors"
    tensors = {
        "lora_unet_q.lora_down.weight": np.random.randn(rank, 8).astype(np.float32),
        "lora_unet_q.lora_up.weight": np.random.randn(8, rank).astype(np.float32),
        "lora_unet_q.alpha": np.array(8.0, dtype=np.float16),
    }
    save_file(tensors, str(p))

    async def _run() -> None:
        app = SftApp(p)
        async with app.run_test() as pilot:
            await pilot.press("k")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            out = p.parent / "kohya.peft.safetensors"
            assert out.exists()

    asyncio.run(_run())


def test_diff_picker_opens(lora_adapter: Path) -> None:
    """Pressing D opens the file picker."""
    from sft.browser import DiffFilePickerScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("D")
            await pilot.pause()
            assert isinstance(app.screen, DiffFilePickerScreen)
            assert app.screen.source_file == lora_adapter

    asyncio.run(_run())


def test_diff_full_flow(tmp_path: Path) -> None:
    """Selecting a file in the picker leads to a DiffResultScreen that computes the diff."""
    from sft.browser import DiffResultScreen, SftApp

    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    save_file(
        {
            "x": np.ones((4, 4), dtype=np.float32),
            "y": np.zeros((2, 2), dtype=np.float32),
        },
        str(p1),
    )
    save_file(
        {"x": np.ones((4, 4), dtype=np.float32), "z": np.eye(3, dtype=np.float32)},
        str(p2),
    )

    async def _run() -> None:
        app = SftApp(p1)
        async with app.run_test() as pilot:
            await pilot.press("D")
            await pilot.pause()
            # Simulate the picker dismissing with our target file
            app.screen.dismiss(p2)
            for _ in range(50):
                await pilot.pause()
                if isinstance(app.screen, DiffResultScreen) and app.screen._loaded:
                    break
            assert isinstance(app.screen, DiffResultScreen)
            assert app.screen._loaded

    asyncio.run(_run())


def test_diff_rejects_same_file(lora_adapter: Path) -> None:
    """The picker should refuse selecting the same source file."""
    from sft.browser import DiffFilePickerScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("D")
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, DiffFilePickerScreen)

            class _Evt:
                path = lora_adapter

            picker.on_directory_tree_file_selected(_Evt())  # type: ignore[arg-type]
            await pilot.pause()
            # Still on picker — same file was rejected
            assert isinstance(app.screen, DiffFilePickerScreen)

    asyncio.run(_run())


def test_lora_resize_prompt_validates_input() -> None:
    """LoRA Resize prompt stores the current rank for display."""
    from sft.browser import LoraResizePromptScreen

    screen = LoraResizePromptScreen(current_rank=4)
    assert screen.current_rank == 4


@pytest.mark.parametrize("key", ["l", "k", "D"])
def test_new_bindings_in_footer(key: str) -> None:
    """New keybindings have show=True so users see them in the footer."""
    from sft.browser import SftApp

    matches = [b for b in SftApp.BINDINGS if b.key == key]
    assert matches, f"binding {key!r} not found"
    assert matches[0].show, f"binding {key!r} should be visible in footer"
