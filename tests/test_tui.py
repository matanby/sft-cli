"""Smoke tests for TUI browser features."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

# --- Fixtures used across tests ---


@pytest.fixture
def kohya_adapter(tmp_path: Path) -> Path:
    rank = 4
    p = tmp_path / "kohya.safetensors"
    tensors = {
        "lora_unet_q.lora_down.weight": np.random.randn(rank, 8).astype(np.float32),
        "lora_unet_q.lora_up.weight": np.random.randn(8, rank).astype(np.float32),
        "lora_unet_q.alpha": np.array(8.0, dtype=np.float16),
    }
    save_file(tensors, str(p))
    return p


# --- Basic instantiation tests ---


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


# --- Main browser bindings: LoRA-agnostic ---


def test_main_browser_has_no_lora_specific_bindings() -> None:
    """The main browser must not expose LoRA-specific bindings (`l`, `k`)."""
    from sft.browser import SftApp

    keys = [b.key for b in SftApp.BINDINGS]
    assert "l" not in keys, "main browser should not have a lowercase `l` binding"
    assert "k" not in keys, "Kohya conversion belongs in LoRA Mode, not main"


def test_main_browser_has_lora_mode_binding() -> None:
    """The single entry point into LoRA Mode is the `L` key."""
    from sft.browser import SftApp

    keys = [b.key for b in SftApp.BINDINGS]
    assert "L" in keys


def test_stats_binding_is_enter() -> None:
    """Stats is opened with Enter (not `S`) on the main browser."""
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "S" not in binding_keys, "the legacy `S` shortcut should be gone"
    enter = [b for b in SftApp.BINDINGS if b.key == "enter"]
    assert enter and enter[0].action == "show_stats"
    assert enter[0].show, "Enter -> Stats should be advertised in the footer"


def test_cast_binding_exists() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "c" in binding_keys


def test_diff_binding_exists() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "D" in binding_keys


def test_colon_binding() -> None:
    from sft.browser import SftApp

    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "colon" in binding_keys


@pytest.mark.parametrize("key", ["L", "D"])
def test_lora_mode_and_diff_in_footer(key: str) -> None:
    """L and D have show=True so the user sees them in the footer."""
    from sft.browser import SftApp

    matches = [b for b in SftApp.BINDINGS if b.key == key]
    assert matches
    assert matches[0].show


# --- Format detection on file open ---


def test_format_peft_detected(lora_adapter: Path) -> None:
    from sft.browser import SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.lora_format == "peft"
            assert app.lora_info is not None

    asyncio.run(_run())


def test_format_kohya_detected(kohya_adapter: Path) -> None:
    from sft.browser import SftApp

    async def _run() -> None:
        app = SftApp(kohya_adapter)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.lora_format == "kohya"

    asyncio.run(_run())


def test_format_none_on_plain_model(mini_model: Path) -> None:
    from sft.browser import SftApp

    async def _run() -> None:
        app = SftApp(mini_model)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.lora_format is None
            assert app.lora_info is None

    asyncio.run(_run())


# --- Cast screen (general-purpose, stays in main) ---


def test_cast_screen_creates() -> None:
    from sft.browser import CastScreen

    screen = CastScreen(Path("/tmp/test.safetensors"))
    assert screen.file_path == Path("/tmp/test.safetensors")


def test_cast_screen_dtypes() -> None:
    from sft.browser import CastScreen

    assert "fp16" in CastScreen.DTYPES
    assert "fp32" in CastScreen.DTYPES
    assert "bf16" in CastScreen.DTYPES


# --- Command palette ---


def test_command_palette_provider() -> None:
    """Verify the command provider can be instantiated."""
    from sft.browser import _HAS_COMMAND_PALETTE

    if not _HAS_COMMAND_PALETTE:
        pytest.skip("Command palette not available in this Textual version")

    from sft.browser import SftCommands

    assert SftCommands is not None


def test_commands_registered() -> None:
    from sft.browser import _HAS_COMMAND_PALETTE, SftApp

    if _HAS_COMMAND_PALETTE:
        from sft.browser import SftCommands

        assert SftCommands in SftApp.COMMANDS
    else:
        assert not SftApp.COMMANDS


# --- Internal helper methods exist ---


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


# --- Tensor details popup intentionally removed ---
#
# The legacy `space` "Details" popup was a strict subset of the Stats
# popup for non-LoRA files, and the LoRA-pair info it surfaced is now
# only reachable via LoRA Mode (`L`). The shortcut and screen are gone.


def test_no_details_popup_or_space_binding() -> None:
    """`space` no longer opens a details popup; the screen is removed."""
    import sft.browser as browser
    from sft.browser import SftApp

    assert not hasattr(browser, "TensorDetailScreen"), (
        "TensorDetailScreen should be removed"
    )
    assert not hasattr(SftApp, "action_show_details"), (
        "action_show_details should be removed"
    )
    binding_keys = [b.key for b in SftApp.BINDINGS]
    assert "space" not in binding_keys, (
        "the legacy `space` Details shortcut should be gone"
    )


# --- LoRA Mode: entry / exit ---


def test_lora_mode_opens_for_peft(lora_adapter: Path) -> None:
    """Pressing `L` on a PEFT file enters LoRA Mode."""
    from sft.browser import LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            assert app.screen.lora_format == "peft"
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, LoraModeScreen)

    asyncio.run(_run())


def test_lora_mode_opens_for_kohya(kohya_adapter: Path) -> None:
    """Kohya files also open LoRA Mode (with conversion notice)."""
    from sft.browser import LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(kohya_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            assert app.screen.lora_format == "kohya"

    asyncio.run(_run())


def test_lora_mode_blocked_for_plain_model(mini_model: Path) -> None:
    """Pressing `L` on a non-LoRA file is a no-op (notifies the user)."""
    from sft.browser import LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(mini_model)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert not isinstance(app.screen, LoraModeScreen)

    asyncio.run(_run())


# --- LoRA Mode: sub-commands ---


def test_lora_mode_stats_computed_in_background(lora_adapter: Path) -> None:
    """Per-pair stats fill in via the background worker."""
    from sft.browser import LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            for _ in range(30):
                await pilot.pause()
                if len(app.screen._stats) == len(app.screen.lora_info.pairs):
                    break
            assert len(app.screen._stats) == len(app.screen.lora_info.pairs)
            for s in app.screen._stats.values():
                assert "norm_a" in s
                assert "eff_rank" in s

    asyncio.run(_run())


def test_lora_mode_sort_by_sv95(lora_adapter: Path) -> None:
    """SV95 sort puts loaded pairs in numeric order; pending pairs last."""
    from sft.browser import LORA_SORT_ORDER, LoraModeScreen, LoraSortMode, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            screen = app.screen
            assert isinstance(app.screen, LoraModeScreen)

            keys = [p.module_key for p in screen.lora_info.pairs]
            screen._stats = {
                keys[0]: {"sv95": 24},
                keys[1]: {"sv95": 8},
            }

            screen._set_sort_mode(LoraSortMode.SV95_DESC)
            desc = screen._sorted_pairs()
            assert [p.module_key for p in desc] == [keys[0], keys[1]]

            screen._set_sort_mode(LoraSortMode.SV95_ASC)
            asc = screen._sorted_pairs()
            assert [p.module_key for p in asc] == [keys[1], keys[0]]

            screen._stats = {keys[0]: {"sv95": 12}}
            screen._set_sort_mode(LoraSortMode.SV95_ASC)
            pending_last = screen._sorted_pairs()
            assert pending_last[0].module_key == keys[0]
            assert pending_last[-1].module_key == keys[1]

            assert LoraSortMode.SV95_DESC in LORA_SORT_ORDER
            assert LoraSortMode.SV95_ASC in LORA_SORT_ORDER

    asyncio.run(_run())


def test_lora_mode_header_click_toggles_sv95_sort(lora_adapter: Path) -> None:
    from textual.widgets import DataTable

    from sft.browser import LORA_SORT_ORDER, LoraModeScreen, LoraSortMode, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LoraModeScreen)
            table = screen.query_one("#lora-table", DataTable)

            screen.on_data_table_header_selected(
                DataTable.HeaderSelected(
                    table,
                    table.columns["sv95"].key,
                    3,
                    table.columns["sv95"].label,
                )
            )
            assert LORA_SORT_ORDER[screen._sort_idx] == LoraSortMode.SV95_DESC

            screen.on_data_table_header_selected(
                DataTable.HeaderSelected(
                    table,
                    table.columns["sv95"].key,
                    3,
                    table.columns["sv95"].label,
                )
            )
            assert LORA_SORT_ORDER[screen._sort_idx] == LoraSortMode.SV95_ASC

    asyncio.run(_run())


def test_lora_mode_sort_by_eff_rank(lora_adapter: Path) -> None:
    """Eff. rank sort puts loaded pairs in numeric order; pending pairs last."""
    from sft.browser import LORA_SORT_ORDER, LoraModeScreen, LoraSortMode, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            screen = app.screen
            assert isinstance(app.screen, LoraModeScreen)

            keys = [p.module_key for p in screen.lora_info.pairs]
            screen._stats = {
                keys[0]: {"eff_rank": 3.0},
                keys[1]: {"eff_rank": 1.0},
            }

            screen._set_sort_mode(LoraSortMode.EFF_RANK_DESC)
            desc = screen._sorted_pairs()
            assert [p.module_key for p in desc] == [keys[0], keys[1]]

            screen._set_sort_mode(LoraSortMode.EFF_RANK_ASC)
            asc = screen._sorted_pairs()
            assert [p.module_key for p in asc] == [keys[1], keys[0]]

            screen._stats = {keys[0]: {"eff_rank": 2.0}}
            screen._set_sort_mode(LoraSortMode.EFF_RANK_ASC)
            pending_last = screen._sorted_pairs()
            assert pending_last[0].module_key == keys[0]
            assert pending_last[-1].module_key == keys[1]

            assert LoraSortMode.EFF_RANK_DESC in LORA_SORT_ORDER
            assert LoraSortMode.EFF_RANK_ASC in LORA_SORT_ORDER

    asyncio.run(_run())


def test_lora_mode_header_click_toggles_eff_rank_sort(lora_adapter: Path) -> None:
    from textual.widgets import DataTable

    from sft.browser import LORA_SORT_ORDER, LoraModeScreen, LoraSortMode, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LoraModeScreen)
            table = screen.query_one("#lora-table", DataTable)

            screen.on_data_table_header_selected(
                DataTable.HeaderSelected(
                    table,
                    table.columns["eff_rank"].key,
                    2,
                    table.columns["eff_rank"].label,
                )
            )
            assert LORA_SORT_ORDER[screen._sort_idx] == LoraSortMode.EFF_RANK_DESC

            screen.on_data_table_header_selected(
                DataTable.HeaderSelected(
                    table,
                    table.columns["eff_rank"].key,
                    2,
                    table.columns["eff_rank"].label,
                )
            )
            assert LORA_SORT_ORDER[screen._sort_idx] == LoraSortMode.EFF_RANK_ASC

    asyncio.run(_run())


def test_lora_mode_spectrum_drill_down(lora_adapter: Path) -> None:
    """Enter on a row inside LoRA Mode opens the spectrum modal."""
    from sft.browser import LoraModeScreen, SftApp, SvdSpectrumScreen

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            for _ in range(30):
                await pilot.pause()
                if isinstance(app.screen, LoraModeScreen) and len(
                    app.screen._stats
                ) == len(app.screen.lora_info.pairs):
                    break
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SvdSpectrumScreen)

    asyncio.run(_run())


def test_lora_mode_compress_opens_prompt(lora_adapter: Path) -> None:
    """`c` inside LoRA Mode opens the compress (resize) prompt."""
    from sft.browser import LoraModeScreen, LoraResizePromptScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            await pilot.press("c")
            await pilot.pause()
            assert isinstance(app.screen, LoraResizePromptScreen)

    asyncio.run(_run())


def test_lora_mode_convert_opens_dialog(lora_adapter: Path) -> None:
    """`k` inside LoRA Mode opens the Kohya<->PEFT convert dialog."""
    from sft.browser import KohyaConvertScreen, LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, KohyaConvertScreen)

    asyncio.run(_run())


def test_lora_mode_info_opens_modal(lora_adapter: Path) -> None:
    """`i` inside LoRA Mode opens the detailed info modal."""
    from sft.browser import LoraInfoScreen, LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            await pilot.press("i")
            await pilot.pause()
            assert isinstance(app.screen, LoraInfoScreen)

    asyncio.run(_run())


def test_kohya_convert_writes_file(kohya_adapter: Path) -> None:
    """Confirming Kohya conversion writes the converted file."""
    from sft.browser import SftApp

    async def _run() -> None:
        app = SftApp(kohya_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            await pilot.press("k")
            await pilot.pause()
            await pilot.press("enter")  # confirm
            await pilot.pause()
            out = kohya_adapter.parent / "kohya.peft.safetensors"
            assert out.exists()

    asyncio.run(_run())


def test_lora_mode_compress_blocked_for_kohya(kohya_adapter: Path) -> None:
    """Compress (`c`) on Kohya files notifies and doesn't open the prompt."""
    from sft.browser import LoraModeScreen, LoraResizePromptScreen, SftApp

    async def _run() -> None:
        app = SftApp(kohya_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert not isinstance(app.screen, LoraResizePromptScreen)
            assert isinstance(app.screen, LoraModeScreen)

    asyncio.run(_run())


def test_lora_resize_prompt_stores_current_rank() -> None:
    """LoRA Resize prompt stores the current rank for display."""
    from sft.browser import LoraResizePromptScreen

    screen = LoraResizePromptScreen(current_rank=4)
    assert screen.current_rank == 4


@pytest.mark.parametrize(
    "typed_value,expected_suffix",
    [
        ("2", ".r2.safetensors"),
        ("auto", ".rauto.safetensors"),
        ("auto+2", ".rauto+2.safetensors"),
    ],
    ids=["integer", "auto", "auto+N"],
)
def test_lora_mode_compress_writes_named_output(
    lora_adapter: Path, typed_value: str, expected_suffix: str
) -> None:
    """Compress accepts integer / auto / auto+N and names the output accordingly."""
    from textual.widgets import Input

    from sft.browser import LoraModeScreen, LoraResizePromptScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, LoraModeScreen)
            await pilot.press("c")
            await pilot.pause()
            assert isinstance(app.screen, LoraResizePromptScreen)

            app.screen.query_one("#rank-input", Input).value = typed_value
            await pilot.press("enter")
            await pilot.pause()
            assert (lora_adapter.parent / f"adapter{expected_suffix}").exists()

    asyncio.run(_run())


def test_lora_mode_footer_hides_main_browser_bindings(lora_adapter: Path) -> None:
    """The LoRA Mode footer must not surface main-browser-only commands.

    Showing both `space` (Details) and `m` (Metadata) alongside `i` (Info)
    is confusing — only LoRA-relevant commands should appear.
    """
    from sft.browser import SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test() as pilot:
            await pilot.press("L")
            await pilot.pause()
            visible = {
                b.binding.key
                for b in app.screen.active_bindings.values()
                if b.binding.show
            }
            for k in ("m", "enter", "D", "slash", "tab", "g", "G"):
                assert k not in visible, f"{k!r} should be hidden from LoRA Mode footer"
            for k in ("escape", "s", "c", "k", "i"):
                assert k in visible, f"{k!r} should be visible in LoRA Mode footer"

    asyncio.run(_run())


# --- Diff (stays in main browser, general-purpose) ---


def test_diff_picker_opens(lora_adapter: Path) -> None:
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
            app.screen.dismiss(p2)
            for _ in range(50):
                await pilot.pause()
                if isinstance(app.screen, DiffResultScreen) and app.screen._loaded:
                    break
            assert isinstance(app.screen, DiffResultScreen)
            assert app.screen._loaded

    asyncio.run(_run())


def test_diff_screen_categorizes_with_value_metrics(tmp_path: Path) -> None:
    """The diff screen classifies tensors into equal/close/differ/incompatible/missing."""
    from sft.browser import DiffResultScreen, SftApp

    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    save_file(
        {
            "identical": np.ones((4, 4), dtype=np.float32),
            "different": np.zeros((4, 4), dtype=np.float32),
            "only_in_a": np.eye(3, dtype=np.float32),
            "shape_mismatch": np.zeros((4, 4), dtype=np.float32),
        },
        str(p1),
    )
    save_file(
        {
            "identical": np.ones((4, 4), dtype=np.float32),
            "different": np.ones((4, 4), dtype=np.float32),
            "only_in_b": np.eye(2, dtype=np.float32),
            "shape_mismatch": np.zeros((8, 8), dtype=np.float32),
        },
        str(p2),
    )

    async def _run() -> None:
        app = SftApp(p1)
        async with app.run_test() as pilot:
            await pilot.press("D")
            await pilot.pause()
            app.screen.dismiss(p2)
            for _ in range(60):
                await pilot.pause()
                if isinstance(app.screen, DiffResultScreen) and app.screen._loaded:
                    break
            assert isinstance(app.screen, DiffResultScreen)
            c = app.screen._counts
            assert c["equal"] == 1  # "identical"
            assert c["differ"] == 1  # "different"
            assert c["incompatible"] == 1  # "shape_mismatch"
            assert c["missing"] == 2  # only_in_a + only_in_b

    asyncio.run(_run())


def test_diff_screen_filter_keys(tmp_path: Path) -> None:
    """Pressing a/d/e/m/i toggles which categories show in the diff table."""
    from sft.browser import DiffResultScreen, SftApp

    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    save_file(
        {
            "same": np.ones((4, 4), dtype=np.float32),
            "diff": np.zeros((4, 4), dtype=np.float32),
            "gone": np.eye(3, dtype=np.float32),
        },
        str(p1),
    )
    save_file(
        {
            "same": np.ones((4, 4), dtype=np.float32),
            "diff": np.ones((4, 4), dtype=np.float32),
            "added": np.eye(2, dtype=np.float32),
        },
        str(p2),
    )

    async def _run() -> None:
        app = SftApp(p1)
        async with app.run_test() as pilot:
            await pilot.press("D")
            await pilot.pause()
            app.screen.dismiss(p2)
            for _ in range(60):
                await pilot.pause()
                if isinstance(app.screen, DiffResultScreen) and app.screen._loaded:
                    break
            screen = app.screen
            assert isinstance(screen, DiffResultScreen)

            # Default visibility excludes "equal"
            assert "equal" not in screen._visible

            await pilot.press("a")  # show all
            await pilot.pause()
            assert screen._visible == {
                "equal",
                "close",
                "differ",
                "incompatible",
                "missing",
            }

            await pilot.press("d")  # only differ
            await pilot.pause()
            assert screen._visible == {"differ"}

            await pilot.press("e")  # equal + close
            await pilot.pause()
            assert screen._visible == {"equal", "close"}

            await pilot.press("m")  # only missing
            await pilot.pause()
            assert screen._visible == {"missing"}

            await pilot.press("i")  # only incompatible
            await pilot.pause()
            assert screen._visible == {"incompatible"}

    asyncio.run(_run())


def test_diff_rejects_same_file(lora_adapter: Path) -> None:
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
            assert isinstance(app.screen, DiffFilePickerScreen)

    asyncio.run(_run())


# --- Visual sizing regression tests ---
#
# The main browser uses a 2-column grid layout on `Screen`. Without an
# explicit override, that rule bleeds into every ModalScreen pushed on
# top of the app, crushing each modal into ~1/3 of the viewport width.
# These tests pin the modal widths to a sensible fraction of the screen
# so the regression can't sneak back in.


def _modal_container_fills(
    file_path: Path,
    key: str,
    container_selector: str,
    min_fraction: float,
    screen_size: tuple[int, int] = (100, 30),
) -> int:
    """Open a modal via `key` and return the actual width of its container.

    Asserts the container width is at least `min_fraction` of the screen
    width — i.e. the App-level grid layout is not leaking into modals.
    """
    from sft.browser import SftApp

    width_holder: list[int] = []

    async def _run() -> None:
        app = SftApp(file_path)
        async with app.run_test(size=screen_size) as pilot:
            await pilot.press(key)
            for _ in range(15):
                await pilot.pause()
                try:
                    container = app.screen.query_one(container_selector)
                    if container.size.width > 0:
                        break
                except Exception:
                    continue
            container = app.screen.query_one(container_selector)
            width_holder.append(container.size.width)

    asyncio.run(_run())
    width = width_holder[0]
    assert width >= int(screen_size[0] * min_fraction), (
        f"{container_selector} only {width}/{screen_size[0]} cols wide — "
        f"app-level grid is probably leaking into modals"
    )
    return width


def test_metadata_modal_fills_screen(lora_adapter: Path) -> None:
    _modal_container_fills(lora_adapter, "m", "#metadata-container", 0.7)


def test_lora_info_modal_fills_screen(lora_adapter: Path) -> None:
    """LoRA Info modal (via L then i) must use most of the screen width."""
    from sft.browser import LoraInfoScreen, LoraModeScreen, SftApp

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("L")
            for _ in range(15):
                await pilot.pause()
                if isinstance(app.screen, LoraModeScreen):
                    break
            await pilot.press("i")
            for _ in range(15):
                await pilot.pause()
                if isinstance(app.screen, LoraInfoScreen):
                    break
            container = app.screen.query_one("#lora-info-container")
            assert container.size.width >= 80, (
                f"lora-info-container only {container.size.width} cols wide"
            )

    asyncio.run(_run())


def test_svd_spectrum_chart_uses_available_width(lora_adapter: Path) -> None:
    """The SVD bar chart should adapt to the modal width, not stay at 40."""
    from sft.browser import LoraModeScreen, SftApp, SvdSpectrumScreen

    async def _run() -> None:
        app = SftApp(lora_adapter)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.press("L")
            for _ in range(30):
                await pilot.pause()
                if isinstance(app.screen, LoraModeScreen) and len(
                    app.screen._stats
                ) == len(app.screen.lora_info.pairs):
                    break
            await pilot.press("enter")
            for _ in range(15):
                await pilot.pause()
                if isinstance(app.screen, SvdSpectrumScreen):
                    break
            for _ in range(5):
                await pilot.pause()
            container = app.screen.query_one("#svd-container")
            assert container.size.width >= 100, (
                f"svd-container only {container.size.width} cols wide"
            )

    asyncio.run(_run())
