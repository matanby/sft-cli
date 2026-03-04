"""Textual TUI application for browsing safetensors files."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Input, Label, Static, Tree
from textual.widgets.tree import TreeNode

from sft.index import (
    PrefixTree,
    PrefixTreeNode,
    TensorIndex,
    TensorInfo,
    natural_sort_key,
)
from sft.lora import LoraInfo, LoraPair, detect_lora_pairs


def format_bytes(nbytes: int) -> str:
    """Format bytes as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    else:
        return f"{nbytes / 1024 / 1024 / 1024:.2f} GB"


def format_shape(shape: tuple[int, ...]) -> str:
    """Format tensor shape as string."""
    if len(shape) == 0:
        return "()"
    return f"({', '.join(str(d) for d in shape)})"


class SortMode(Enum):
    """Sort modes for tensor table."""

    NAME_ASC = "name ↑"
    NAME_DESC = "name ↓"
    SIZE_ASC = "size ↑"
    SIZE_DESC = "size ↓"
    RANK_ASC = "rank ↑"
    RANK_DESC = "rank ↓"


SORT_ORDER = [
    SortMode.NAME_ASC,
    SortMode.NAME_DESC,
    SortMode.SIZE_DESC,
    SortMode.SIZE_ASC,
    SortMode.RANK_DESC,
    SortMode.RANK_ASC,
]


class TensorDetailScreen(ModalScreen):
    """Modal screen showing tensor details."""

    CSS = """
    TensorDetailScreen {
        align: center middle;
    }

    #detail-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #detail-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    .detail-row {
        margin: 0;
    }

    .detail-label {
        color: $text-muted;
    }

    .detail-value {
        color: $text;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("space", "dismiss", "Close"),
    ]

    def __init__(self, tensor: TensorInfo, lora_info: LoraInfo | None = None) -> None:
        super().__init__()
        self.tensor = tensor
        self.lora_info = lora_info

    def compose(self) -> ComposeResult:
        t = self.tensor
        with Container(id="detail-container"):
            yield Label("Tensor Details", id="detail-title")
            yield Static(f"[dim]Name:[/dim]  {t.full_name}", classes="detail-row")
            yield Static(
                f"[dim]Shape:[/dim] {format_shape(t.shape)}", classes="detail-row"
            )
            yield Static(f"[dim]Rank:[/dim]  {t.rank}", classes="detail-row")
            yield Static(f"[dim]Dtype:[/dim] {t.dtype}", classes="detail-row")
            yield Static(
                f"[dim]Size:[/dim]  {format_bytes(t.nbytes)} ({t.nbytes:,} bytes)",
                classes="detail-row",
            )
            yield Static(f"[dim]Numel:[/dim] {t.numel:,}", classes="detail-row")

            if self.lora_info:
                li = self.lora_info
                yield Static("", classes="detail-row")
                yield Static("[bold cyan]LoRA Info[/bold cyan]", classes="detail-row")
                yield Static(f"[dim]Role:[/dim]  {li.role.value}", classes="detail-row")
                yield Static(f"[dim]Rank:[/dim]  {li.pair.rank}", classes="detail-row")
                yield Static(
                    f"[dim]Base:[/dim]  {li.pair.base_name}", classes="detail-row"
                )
                paired = (
                    li.pair.b_tensor_name
                    if li.role.name == "A"
                    else li.pair.a_tensor_name
                )
                yield Static(f"[dim]Pair:[/dim]  {paired}", classes="detail-row")

            yield Static(
                "\n[dim]Press ESC or SPACE to close[/dim]", classes="detail-row"
            )


class MetadataScreen(ModalScreen):
    """Modal screen showing file metadata."""

    CSS = """
    MetadataScreen {
        align: center middle;
    }

    #metadata-container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $secondary;
        padding: 1 2;
    }

    #metadata-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #metadata-content {
        height: auto;
        max-height: 20;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("m", "dismiss", "Close"),
    ]

    def __init__(self, metadata: dict, file_path: Path) -> None:
        super().__init__()
        self.metadata = metadata
        self.file_path = file_path

    def compose(self) -> ComposeResult:
        with Container(id="metadata-container"):
            yield Label("File Metadata", id="metadata-title")
            yield Static(f"[dim]File:[/dim] {self.file_path.name}")

            if self.metadata:
                formatted = json.dumps(self.metadata, indent=2)
                yield Static(f"\n{formatted}", id="metadata-content")
            else:
                yield Static("\n[dim]No metadata found in file[/dim]")

            yield Static("\n[dim]Press ESC or M to close[/dim]")


class FilterScreen(ModalScreen):
    """Modal screen for filtering tensors."""

    CSS = """
    FilterScreen {
        align: center middle;
    }

    #filter-container {
        width: 50;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }

    #filter-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    .filter-section {
        margin: 1 0;
    }

    .filter-label {
        color: $text-muted;
        margin-bottom: 0;
    }

    .filter-options {
        margin-left: 2;
    }

    .dtype-option {
        margin: 0;
    }

    .dtype-option.selected {
        color: $success;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("f", "dismiss", "Close"),
        Binding("c", "clear_filters", "Clear All"),
        Binding("1", "toggle_dtype_0", "Toggle", show=False),
        Binding("2", "toggle_dtype_1", "Toggle", show=False),
        Binding("3", "toggle_dtype_2", "Toggle", show=False),
        Binding("4", "toggle_dtype_3", "Toggle", show=False),
        Binding("5", "toggle_dtype_4", "Toggle", show=False),
    ]

    COMMON_DTYPES = ["F16", "F32", "BF16", "I8", "I32"]

    def __init__(self, current_filters: dict, available_dtypes: set[str]) -> None:
        super().__init__()
        self.current_filters = current_filters.copy()
        self.available_dtypes = sorted(available_dtypes)
        self.selected_dtypes: set[str] = set(current_filters.get("dtypes", []))

    def compose(self) -> ComposeResult:
        with Container(id="filter-container"):
            yield Label("Filter Tensors", id="filter-title")

            # Dtype filter
            yield Static("[bold]Dtype Filter[/bold]", classes="filter-section")
            for i, dtype in enumerate(self.available_dtypes[:5]):
                selected = "✓" if dtype in self.selected_dtypes else " "
                css_class = (
                    "dtype-option selected"
                    if dtype in self.selected_dtypes
                    else "dtype-option"
                )
                yield Static(
                    f"  [{i + 1}] {selected} {dtype}",
                    classes=css_class,
                    id=f"dtype-{i}",
                )

            yield Static("\n[dim]Keys:[/dim]", classes="filter-section")
            yield Static("  [1-5] Toggle dtype")
            yield Static("  [c] Clear all filters")
            yield Static("  [ESC/f] Close")

    def _toggle_dtype(self, index: int) -> None:
        """Toggle a dtype filter."""
        if index >= len(self.available_dtypes):
            return

        dtype = self.available_dtypes[index]
        if dtype in self.selected_dtypes:
            self.selected_dtypes.discard(dtype)
        else:
            self.selected_dtypes.add(dtype)

        # Update display
        selected = "✓" if dtype in self.selected_dtypes else " "
        widget = self.query_one(f"#dtype-{index}", Static)
        widget.update(f"  [{index + 1}] {selected} {dtype}")
        if dtype in self.selected_dtypes:
            widget.add_class("selected")
        else:
            widget.remove_class("selected")

    def action_toggle_dtype_0(self) -> None:
        self._toggle_dtype(0)

    def action_toggle_dtype_1(self) -> None:
        self._toggle_dtype(1)

    def action_toggle_dtype_2(self) -> None:
        self._toggle_dtype(2)

    def action_toggle_dtype_3(self) -> None:
        self._toggle_dtype(3)

    def action_toggle_dtype_4(self) -> None:
        self._toggle_dtype(4)

    def action_clear_filters(self) -> None:
        """Clear all filters."""
        self.selected_dtypes.clear()
        for i in range(min(5, len(self.available_dtypes))):
            widget = self.query_one(f"#dtype-{i}", Static)
            dtype = self.available_dtypes[i]
            widget.update(f"  [{i + 1}]   {dtype}")
            widget.remove_class("selected")

    def action_dismiss(self) -> None:
        """Dismiss and return filters."""
        filters = {}
        if self.selected_dtypes:
            filters["dtypes"] = list(self.selected_dtypes)
        self.dismiss(filters)


class SvdScreen(ModalScreen):
    """Modal screen showing SVD vertical histogram for a single LoRA pair."""

    CSS = """
    SvdScreen {
        layout: vertical;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }

    #svd-title {
        text-align: center;
        text-style: bold;
        height: auto;
        margin-bottom: 1;
    }

    #svd-info {
        height: auto;
        margin-bottom: 1;
    }

    #svd-scroll {
        height: 1fr;
    }

    #svd-help {
        height: auto;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(
        self,
        pair: LoraPair,
        file_path: Path,
        index: TensorIndex,
    ) -> None:
        super().__init__()
        self.pair = pair
        self.file_path = file_path
        self.index = index

    def compose(self) -> ComposeResult:
        pair = self.pair
        yield Label("SVD Spectrum", id="svd-title")
        yield Static(
            f"[bold]{pair.base_name}[/bold]  "
            f"[dim]rank={pair.rank}  "
            f"A {format_shape(pair.a_shape)}  B {format_shape(pair.b_shape)}[/dim]",
            id="svd-info",
        )
        with VerticalScroll(id="svd-scroll"):
            yield Static("[dim]Computing...[/dim]", id="svd-chart")
        yield Static("[dim]ESC: close[/dim]", id="svd-help")

    def on_mount(self) -> None:
        """Compute SVD after layout so we know the terminal width."""
        self.call_after_refresh(self._run_svd)

    def _run_svd(self) -> None:
        """Compute SVD and update the chart."""
        chart = self.query_one("#svd-chart", Static)
        pair = self.pair
        try:
            from sft.data import compute_lora_svd

            sv, sigma0 = compute_lora_svd(
                self.file_path,
                pair,
                self.index.header_size,
                self.index,
            )

            avail_w = self.app.size.width - 8  # terminal width minus borders/padding
            chart.update(self._build_vertical_histogram(sv, avail_w, sigma0))
        except ImportError as e:
            chart.update(f"[red]{e}[/red]")
        except Exception as e:
            chart.update(f"[red]Error: {e}[/red]")

    @staticmethod
    def _build_vertical_histogram(sv, avail_width: int = 120, raw_sigma0: float | None = None) -> str:
        """Build a vertical bar histogram with sqrt scale to show all values."""
        import math

        n = len(sv)
        if n == 0:
            return "[dim]No singular values[/dim]"

        values = [float(sv[i]) for i in range(n)]
        max_val = max(values) if values else 1.0

        # Use sqrt scale so smaller values are visible
        scaled = [math.sqrt(v / max_val) for v in values]

        # Layout: prefix "1.000 │" = 8 chars
        prefix_w = 8
        chart_w = avail_width - prefix_w
        if chart_w < n:
            chart_w = n

        # Bar width: fill available space evenly
        bar_w = max(1, chart_w // n)
        gap = 1 if bar_w > 1 else 0
        # Recalc bar_w accounting for gaps
        if gap and n > 1:
            bar_w = max(1, (chart_w - (n - 1) * gap) // n)
        total_w = n * bar_w + max(0, n - 1) * gap

        chart_h = 18
        heights = [max(1, int(s * chart_h)) if s > 0.01 else 0 for s in scaled]

        lines = []
        for row in range(chart_h, 0, -1):
            cells = []
            for i, h in enumerate(heights):
                block = "█" * bar_w if h >= row else " " * bar_w
                if i > 0 and gap:
                    cells.append(" " * gap)
                cells.append(block)
            # Y-axis labels (show actual values, not sqrt)
            if row == chart_h:
                label = f"{max_val:.3f}│"
            elif row == 1:
                label = "  0.00│"
            else:
                label = "      │"
            lines.append(f"{label:>8}{''.join(cells)}")

        # X axis
        lines.append(f"{'─' * 7}┼{'─' * total_w}")

        # Index labels below axis
        idx_line = list(" " * total_w)
        step = bar_w + gap
        for idx in (0, n // 4, n // 2, 3 * n // 4, n - 1):
            if idx >= n:
                continue
            pos = idx * step
            s = f"σ{idx}"
            for ci, ch in enumerate(s):
                p = pos + ci
                if p < total_w:
                    idx_line[p] = ch
        lines.append(f"{'':>8}{''.join(idx_line)}")

        # Effective rank statistics
        total = sum(values)
        cumsum = 0.0
        eff_90 = n
        eff_99 = n
        for i, v in enumerate(values):
            cumsum += v
            if cumsum >= total * 0.90 and eff_90 == n:
                eff_90 = i + 1
            if cumsum >= total * 0.99 and eff_99 == n:
                eff_99 = i + 1
                break

        # Stable rank: ||A||_F^2 / ||A||_2^2 = sum(σ^2) / σ_0^2
        sq_sum = sum(v * v for v in values)
        stable_rank = sq_sum / (values[0] * values[0]) if values[0] > 0 else 0

        lines.append("")
        sigma0_str = f"  raw σ₀: [cyan]{raw_sigma0:.4f}[/cyan]" if raw_sigma0 is not None else ""
        lines.append(
            f"[bold]Effective rank:[/bold]  "
            f"90% energy: [cyan]{eff_90}[/cyan]/{n}  "
            f"99% energy: [cyan]{eff_99}[/cyan]/{n}  "
            f"stable rank: [cyan]{stable_rank:.1f}[/cyan]{sigma0_str}  "
            f"[dim](norm σ₀={values[0]:.4f}, σ{n - 1}={values[-1]:.4f})  √ scale[/dim]"
        )

        return "\n".join(lines)


class LoraHelpScreen(ModalScreen):
    """Help screen explaining LoRA analysis metrics."""

    CSS = """
    LoraHelpScreen {
        layout: vertical;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        overflow-y: auto;
    }

    #lora-help-title {
        text-align: center;
        text-style: bold;
        height: auto;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Label("LoRA Analysis — Help", id="lora-help-title")
        yield Static(
            "[bold]Frobenius Norm  ||A||, ||B||[/bold]\n"
            "The Frobenius norm is the square root of the sum of all\n"
            "squared elements: ||M|| = √(Σ mᵢⱼ²). It measures the\n"
            "overall magnitude of a matrix.\n"
            "\n"
            "For LoRA, ||A|| and ||B|| tell you how large each factor\n"
            "is. Layers with large norms are being modified more\n"
            "aggressively. Comparing norms across layers reveals where\n"
            "the fine-tuning concentrates its effort. Unusually small\n"
            "norms may indicate dead or underutilized layers.\n"
        )
        yield Static(
            "[bold]Effective Rank  (Stable Rank)[/bold]\n"
            "Stable rank = Σσᵢ² / σ₀², where σᵢ are the singular\n"
            "values of the effective matrix B@A.\n"
            "\n"
            'It answers: "how many equal singular values would produce\n'
            'the same energy distribution?" Unlike hard thresholds,\n'
            "stable rank is continuous — a small perturbation won't\n"
            "cause a jump.\n"
            "\n"
            "  • Stable rank ≈ 1  →  rank-1 in disguise. The LoRA\n"
            "    update is dominated by a single direction. The\n"
            "    allocated rank is mostly wasted.\n"
            "  • Stable rank ≈ rank  →  full utilization. All\n"
            "    dimensions contribute equally.\n"
            "  • Stable rank << rank  →  the LoRA could be retrained\n"
            "    at a lower rank for similar quality.\n"
            "\n"
            "Related to the Herfindahl-Hirschman Index (HHI) from\n"
            "economics: if you treat σᵢ²/Σσⱼ² as market shares,\n"
            '1/HHI gives the "effective number of players" —\n'
            "a closely analogous measure.\n"
        )
        yield Static(
            "[bold]SVD Spectrum[/bold]\n"
            "The singular value decomposition of B@A shows how the\n"
            "LoRA's contribution decomposes into independent rank-1\n"
            "updates, ordered by importance (σ₀ ≥ σ₁ ≥ ... ≥ σᵣ).\n"
            "\n"
            "The histogram uses √ scale so smaller values remain\n"
            "visible alongside dominant ones.\n"
        )
        yield Static("[dim]Press ESC or ? to close[/dim]")


class CompactifyScreen(ModalScreen):
    """Modal screen for compactifying LoRA pairs to a lower rank."""

    CSS = """
    CompactifyScreen {
        align: center middle;
    }

    #compactify-box {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }

    #compactify-title {
        text-align: center;
        text-style: bold;
        height: auto;
        margin-bottom: 1;
    }

    #compactify-info {
        height: auto;
        margin-bottom: 1;
    }

    #compactify-input {
        margin-bottom: 1;
    }

    #compactify-status {
        height: auto;
    }

    #compactify-help {
        height: auto;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(
        self,
        pairs: list[LoraPair],
        file_path: Path,
        index: TensorIndex,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.file_path = file_path
        self.index = index

    def compose(self) -> ComposeResult:
        ranks = sorted({p.rank for p in self.pairs})
        min_rank = min(ranks)
        with Container(id="compactify-box"):
            yield Label("Compactify LoRA", id="compactify-title")
            yield Static(
                f"[bold]{len(self.pairs)}[/bold] pairs  |  "
                f"ranks: {', '.join(str(r) for r in ranks)}\n"
                f"Target rank must be < {min_rank}",
                id="compactify-info",
            )
            yield Input(
                placeholder=f"Rank (1–{min_rank - 1}), 'auto', or 'auto+N'",
                id="compactify-input",
            )
            yield Static("", id="compactify-status")
            yield Static(
                "[dim]Enter: run  |  ESC: cancel[/dim]",
                id="compactify-help",
            )

    def on_mount(self) -> None:
        self.query_one("#compactify-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        status = self.query_one("#compactify-status", Static)
        raw = event.value.strip().lower()

        import re
        auto_match = re.match(r"^auto(?:\+(\d+))?$", raw)
        if auto_match:
            margin = int(auto_match.group(1)) if auto_match.group(1) else 1
            status.update(f"[yellow]Computing per-pair effective ranks (margin +{margin})...[/yellow]")
            self.query_one("#compactify-input", Input).disabled = True
            self.run_worker(
                lambda: self._do_compactify_auto(margin), thread=True,
            )
            return

        try:
            target_rank = int(raw)
        except ValueError:
            status.update("[red]Enter an integer, 'auto', or 'auto+N'[/red]")
            return

        if target_rank < 1:
            status.update("[red]Rank must be >= 1[/red]")
            return

        truncatable = [p for p in self.pairs if p.rank > target_rank]
        if not truncatable:
            status.update(f"[red]No pairs with rank > {target_rank}[/red]")
            return

        status.update(f"[yellow]Truncating {len(truncatable)} pairs to rank {target_rank}...[/yellow]")
        self.query_one("#compactify-input", Input).disabled = True
        self.run_worker(
            lambda: self._do_compactify(truncatable, target_rank), thread=True,
        )

    def _do_compactify(
        self, truncatable: list[LoraPair], target_rank: int,
    ) -> None:
        import traceback

        from sft.data import (
            load_all_tensors,
            load_tensor,
            truncate_lora_pair,
            write_safetensors,
        )

        try:
            all_tensors = load_all_tensors(self.file_path, self.index)
            tensor_map = {t.full_name: t for t in self.index.tensors}

            energies = []
            for i, pair in enumerate(truncatable):
                a_info = tensor_map[pair.a_tensor_name]
                b_info = tensor_map[pair.b_tensor_name]
                a = load_tensor(
                    self.file_path, self.index.header_size,
                    a_info.data_offsets, a_info.dtype, a_info.shape,
                )
                b = load_tensor(
                    self.file_path, self.index.header_size,
                    b_info.data_offsets, b_info.dtype, b_info.shape,
                )
                a_new, b_new, energy = truncate_lora_pair(a, b, target_rank, a_info.dtype)
                all_tensors[pair.a_tensor_name] = (a_new, a_info.dtype)
                all_tensors[pair.b_tensor_name] = (b_new, b_info.dtype)
                energies.append(energy)

                self.app.call_from_thread(
                    self.query_one("#compactify-status", Static).update,
                    f"[yellow]Truncated {i + 1}/{len(truncatable)} pairs...[/yellow]",
                )

            tensor_order = [t.full_name for t in self.index.tensors]
            out_path = self.file_path.parent / f"{self.file_path.stem}_r{target_rank}.safetensors"

            self.app.call_from_thread(
                self.query_one("#compactify-status", Static).update,
                "[yellow]Writing file...[/yellow]",
            )
            write_safetensors(
                out_path,
                all_tensors,
                tensor_order,
                self.index.metadata if self.index.metadata else None,
            )

            orig_size = self.file_path.stat().st_size
            new_size = out_path.stat().st_size
            reduction = (1 - new_size / orig_size) * 100

            avg_energy = sum(energies) / len(energies) * 100
            min_energy = min(energies) * 100
            summary = (
                f"[green]Saved {out_path.name} ({reduction:.0f}% smaller)\n"
                f"Energy retained: avg {avg_energy:.1f}%, min {min_energy:.1f}% "
                f"(across {len(truncatable)} pairs)[/green]"
            )

            self.app.call_from_thread(
                self.query_one("#compactify-status", Static).update,
                summary,
            )
            self.app.call_from_thread(
                self.app.notify,
                f"Saved {out_path.name} — avg energy retained: {avg_energy:.1f}%",
            )
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#compactify-status", Static).update,
                f"[red]{traceback.format_exc()}[/red]",
            )
    

    def _do_compactify_auto(self, margin: int = 1) -> None:
        import math
        import traceback

        from sft.data import (
            compute_lora_svd,
            load_all_tensors,
            load_tensor,
            truncate_lora_pair,
            write_safetensors,
        )

        try:
            # First pass: compute effective rank for each pair
            pair_ranks = {}
            for i, pair in enumerate(self.pairs):
                sv, _sigma0 = compute_lora_svd(
                    self.file_path, pair, self.index.header_size, self.index,
                )
                sq = sv * sv
                stable_rank = float(sq.sum() / sq[0]) if sq[0] > 0 else pair.rank
                auto_rank = min(math.ceil(stable_rank) + margin, pair.rank)
                pair_ranks[pair.base_name] = auto_rank

                self.app.call_from_thread(
                    self.query_one("#compactify-status", Static).update,
                    f"[yellow]Computing ranks: {i + 1}/{len(self.pairs)} "
                    f"(eff={stable_rank:.1f} → r{auto_rank})...[/yellow]",
                )

            # Filter to pairs that actually get truncated
            truncatable = [p for p in self.pairs if pair_ranks[p.base_name] < p.rank]
            if not truncatable:
                self.app.call_from_thread(
                    self.query_one("#compactify-status", Static).update,
                    "[yellow]All pairs already at or below effective rank. Nothing to do.[/yellow]",
                )
                return

            # Second pass: load and truncate
            all_tensors = load_all_tensors(self.file_path, self.index)
            tensor_map = {t.full_name: t for t in self.index.tensors}

            energies = []
            for i, pair in enumerate(truncatable):
                target_rank = pair_ranks[pair.base_name]
                a_info = tensor_map[pair.a_tensor_name]
                b_info = tensor_map[pair.b_tensor_name]
                a = load_tensor(
                    self.file_path, self.index.header_size,
                    a_info.data_offsets, a_info.dtype, a_info.shape,
                )
                b = load_tensor(
                    self.file_path, self.index.header_size,
                    b_info.data_offsets, b_info.dtype, b_info.shape,
                )
                a_new, b_new, energy = truncate_lora_pair(a, b, target_rank, a_info.dtype)
                all_tensors[pair.a_tensor_name] = (a_new, a_info.dtype)
                all_tensors[pair.b_tensor_name] = (b_new, b_info.dtype)
                energies.append(energy)

                self.app.call_from_thread(
                    self.query_one("#compactify-status", Static).update,
                    f"[yellow]Truncated {i + 1}/{len(truncatable)} pairs...[/yellow]",
                )

            tensor_order = [t.full_name for t in self.index.tensors]
            suffix = "auto" if margin == 1 else f"auto+{margin}"
            out_path = self.file_path.parent / f"{self.file_path.stem}_r{suffix}.safetensors"

            self.app.call_from_thread(
                self.query_one("#compactify-status", Static).update,
                "[yellow]Writing file...[/yellow]",
            )
            write_safetensors(
                out_path,
                all_tensors,
                tensor_order,
                self.index.metadata if self.index.metadata else None,
            )

            orig_size = self.file_path.stat().st_size
            new_size = out_path.stat().st_size
            reduction = (1 - new_size / orig_size) * 100

            auto_ranks = [pair_ranks[p.base_name] for p in truncatable]
            avg_energy = sum(energies) / len(energies) * 100
            min_energy = min(energies) * 100
            summary = (
                f"[green]Saved {out_path.name} ({reduction:.0f}% smaller)\n"
                f"Per-pair rank: min {min(auto_ranks)}, max {max(auto_ranks)}, "
                f"avg {sum(auto_ranks)/len(auto_ranks):.0f}\n"
                f"Energy retained: avg {avg_energy:.1f}%, min {min_energy:.1f}% "
                f"({len(truncatable)} pairs truncated, "
                f"{len(self.pairs) - len(truncatable)} unchanged)[/green]"
            )

            self.app.call_from_thread(
                self.query_one("#compactify-status", Static).update,
                summary,
            )
            self.app.call_from_thread(
                self.app.notify,
                f"Saved {out_path.name} — avg energy retained: {avg_energy:.1f}%",
            )
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#compactify-status", Static).update,
                f"[red]{traceback.format_exc()}[/red]",
            )


class LoraScreen(ModalScreen):
    """Modal screen showing LoRA pair analysis with a DataTable."""

    CSS = """
    LoraScreen {
        layout: vertical;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }

    #lora-title {
        text-align: center;
        text-style: bold;
        height: auto;
        margin-bottom: 1;
    }

    #lora-summary {
        height: auto;
        margin-bottom: 1;
    }

    #lora-pair-table {
        height: 1fr;
    }

    #lora-help {
        height: auto;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("l", "dismiss", "Close"),
        Binding("enter", "open_svd", "SVD", priority=True),
        Binding("s", "cycle_sort", "Sort"),
        Binding("c", "compactify", "Compactify"),
        Binding("question_mark", "show_help", "Help"),
        Binding("e", "export_json", "Export"),
    ]

    _SORT_MODES = ["name", "rank", "eff_rank", "sigma0", "norm_a", "norm_b"]
    _SORT_LABELS = {
        "name": "Name",
        "rank": "Rank",
        "eff_rank": "Eff. Rank ↓",
        "sigma0": "σ₀ ↓",
        "norm_a": "||A|| ↓",
        "norm_b": "||B|| ↓",
    }

    def __init__(
        self,
        pairs: list[LoraPair],
        file_path: Path,
        index: TensorIndex,
        stats_cache: dict[str, dict],
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.file_path = file_path
        self.index = index
        self.stats_cache = stats_cache
        self._pool = None
        self._sort_index = 0  # index into _SORT_MODES

    def compose(self) -> ComposeResult:
        ranks = [p.rank for p in self.pairs]
        unique_ranks = sorted(set(ranks))
        rank_dist = ", ".join(f"r{r}:×{ranks.count(r)}" for r in unique_ranks)

        yield Label("LoRA Analysis", id="lora-title")
        yield Static(
            f"[bold]{len(self.pairs)}[/bold] pairs  |  ranks: {rank_dist}",
            id="lora-summary",
        )
        table = DataTable(id="lora-pair-table", cursor_type="row", zebra_stripes=True)
        yield table
        yield Static(
            "[dim]↑/↓ select  |  Enter: SVD  |  s: sort  |  c: compactify  |  e: export  |  ?: help  |  ESC/L: close[/dim]",
            id="lora-help",
        )

    def on_mount(self) -> None:
        """Populate the pair table from cache, then compute missing stats."""
        table = self.query_one("#lora-pair-table", DataTable)
        table.add_column("Base Name", key="base_name")
        table.add_column("Rank", key="rank")
        table.add_column("Eff. Rank", key="eff_rank")
        table.add_column("σ₀", key="sigma0")
        table.add_column("||A||", key="norm_a")
        table.add_column("||B||", key="norm_b")
        table.add_column("A Shape", key="a_shape")
        table.add_column("B Shape", key="b_shape")
        for i, pair in enumerate(self.pairs):
            cached = self.stats_cache.get(pair.base_name, {})
            table.add_row(
                pair.base_name,
                str(pair.rank),
                f"{cached['eff_rank']:.1f}" if "eff_rank" in cached else "...",
                f"{cached['sigma0']:.4f}" if "sigma0" in cached else "...",
                f"{cached['norm_a']:.2f}" if "norm_a" in cached else "...",
                f"{cached['norm_b']:.2f}" if "norm_b" in cached else "...",
                format_shape(pair.a_shape),
                format_shape(pair.b_shape),
                key=str(i),
            )
        table.focus()

        # Only compute stats for pairs not yet cached
        missing_norms = [
            (i, p)
            for i, p in enumerate(self.pairs)
            if "norm_a" not in self.stats_cache.get(p.base_name, {})
        ]
        missing_ranks = [
            (i, p)
            for i, p in enumerate(self.pairs)
            if "eff_rank" not in self.stats_cache.get(p.base_name, {})
        ]
        if missing_norms or missing_ranks:
            self.run_worker(
                self._compute_missing(missing_norms, missing_ranks), thread=True
            )

    async def _compute_missing(
        self,
        missing_norms: list[tuple[int, LoraPair]],
        missing_ranks: list[tuple[int, LoraPair]],
    ) -> None:
        """Compute only uncached stats using a thread pool."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from sft.data import compute_frobenius_norms, compute_stable_rank

        loop = asyncio.get_event_loop()

        def _calc_norms(idx: int, pair: LoraPair) -> tuple[int, str, dict]:
            stats = compute_frobenius_norms(
                self.file_path, pair, self.index.header_size, self.index
            )
            return idx, pair.base_name, stats

        def _calc_rank(idx: int, pair: LoraPair) -> tuple[int, str, float, float]:
            sr, sigma0 = compute_stable_rank(
                self.file_path, pair, self.index.header_size, self.index
            )
            return idx, pair.base_name, sr, sigma0

        self._pool = ThreadPoolExecutor(max_workers=4)
        pool = self._pool

        # Frobenius norms
        if missing_norms:
            futures = [
                loop.run_in_executor(pool, _calc_norms, i, p) for i, p in missing_norms
            ]
            for fut in asyncio.as_completed(futures):
                try:
                    idx, name, stats = await fut
                    self.stats_cache.setdefault(name, {}).update(stats)
                    table = self.query_one("#lora-pair-table", DataTable)
                    table.update_cell(str(idx), "norm_a", f"{stats['norm_a']:.2f}")
                    table.update_cell(str(idx), "norm_b", f"{stats['norm_b']:.2f}")
                except Exception:
                    pass

        # Stable rank
        if missing_ranks:
            futures = [
                loop.run_in_executor(pool, _calc_rank, i, p) for i, p in missing_ranks
            ]
            for fut in asyncio.as_completed(futures):
                try:
                    idx, name, sr, sigma0 = await fut
                    self.stats_cache.setdefault(name, {}).update(
                        eff_rank=sr, sigma0=sigma0,
                    )
                    table = self.query_one("#lora-pair-table", DataTable)
                    table.update_cell(str(idx), "eff_rank", f"{sr:.1f}")
                    table.update_cell(str(idx), "sigma0", f"{sigma0:.4f}")
                except Exception:
                    pass

        pool.shutdown(wait=False)
        self._pool = None

    def _shutdown_pool(self) -> None:
        """Shut down the thread pool if running."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def on_unmount(self) -> None:
        """Clean up when screen is removed."""
        self._shutdown_pool()

    def action_cycle_sort(self) -> None:
        """Cycle through sort modes and re-sort the table."""
        self._sort_index = (self._sort_index + 1) % len(self._SORT_MODES)
        mode = self._SORT_MODES[self._sort_index]
        label = self._SORT_LABELS[mode]

        table = self.query_one("#lora-pair-table", DataTable)

        if mode == "name":
            # Sort by original pair order (name)
            order = list(range(len(self.pairs)))
        elif mode == "rank":
            order = sorted(range(len(self.pairs)),
                           key=lambda i: self.pairs[i].rank, reverse=True)
        else:
            # Sort by cached numeric stat (eff_rank, norm_a, norm_b)
            def _stat(i: int) -> float:
                cached = self.stats_cache.get(self.pairs[i].base_name, {})
                return cached.get(mode, -1.0)
            order = sorted(range(len(self.pairs)), key=_stat, reverse=True)

        # Rebuild table rows in new order
        table.clear()
        # Remap pairs list to match new order
        reordered = [self.pairs[i] for i in order]
        self.pairs = reordered
        for i, pair in enumerate(self.pairs):
            cached = self.stats_cache.get(pair.base_name, {})
            table.add_row(
                pair.base_name,
                str(pair.rank),
                f"{cached['eff_rank']:.1f}" if "eff_rank" in cached else "...",
                f"{cached['sigma0']:.4f}" if "sigma0" in cached else "...",
                f"{cached['norm_a']:.2f}" if "norm_a" in cached else "...",
                f"{cached['norm_b']:.2f}" if "norm_b" in cached else "...",
                format_shape(pair.a_shape),
                format_shape(pair.b_shape),
                key=str(i),
            )

        summary = self.query_one("#lora-summary", Static)
        ranks = [p.rank for p in self.pairs]
        unique_ranks = sorted(set(ranks))
        rank_dist = ", ".join(f"r{r}:×{ranks.count(r)}" for r in unique_ranks)
        summary.update(
            f"[bold]{len(self.pairs)}[/bold] pairs  |  ranks: {rank_dist}  |  sort: [cyan]{label}[/cyan]"
        )

    def action_open_svd(self) -> None:
        """Open SVD screen for the selected pair."""
        table = self.query_one("#lora-pair-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.pairs):
            pair = self.pairs[table.cursor_row]
            self.app.push_screen(SvdScreen(pair, self.file_path, self.index))

    def action_compactify(self) -> None:
        """Open compactify screen."""
        self.app.push_screen(
            CompactifyScreen(self.pairs, self.file_path, self.index)
        )

    def action_show_help(self) -> None:
        """Show LoRA analysis help screen."""
        self.app.push_screen(LoraHelpScreen())

    def action_export_json(self) -> None:
        """Export LoRA analysis to JSON file."""
        export = []
        for pair in self.pairs:
            cached = self.stats_cache.get(pair.base_name, {})
            entry = {
                "base_name": pair.base_name,
                "rank": pair.rank,
                "a_tensor": pair.a_tensor_name,
                "b_tensor": pair.b_tensor_name,
                "a_shape": list(pair.a_shape),
                "b_shape": list(pair.b_shape),
            }
            for key in ("norm_a", "norm_b", "min_a", "max_a", "mean_a", "median_a",
                        "min_b", "max_b", "mean_b", "median_b", "eff_rank", "sigma0"):
                if key in cached:
                    entry[key] = round(cached[key], 4)
            export.append(entry)

        out_path = self.file_path.with_suffix(".lora_analysis.json")
        with open(out_path, "w") as f:
            json.dump(export, f, indent=2)
        self.app.notify(f"Exported to {out_path.name}", severity="information")


class FilteredPrefixTree:
    """A filtered view of a PrefixTree containing only matching tensors."""

    def __init__(
        self, original_tree: PrefixTree, matching_tensors: list[TensorInfo]
    ) -> None:
        """Build a filtered tree from matching tensors."""
        self.original_tree = original_tree
        self.index = original_tree.index
        self.delimiter = original_tree.delimiter
        self.matching_tensor_names = {t.full_name for t in matching_tensors}

        # Build filtered tree structure
        self.root = self._build_filtered_node(original_tree.root, "")

    def _build_filtered_node(
        self, original_node: PrefixTreeNode, prefix: str
    ) -> PrefixTreeNode | None:
        """Recursively build a filtered node, returning None if no matches."""
        # Check direct tensors
        matching_direct = [
            tid
            for tid in original_node.tensor_ids
            if self.index.tensors[tid].full_name in self.matching_tensor_names
        ]

        # Recursively filter children
        filtered_children: dict[str, PrefixTreeNode] = {}
        for child_name, child_node in original_node.children.items():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            filtered_child = self._build_filtered_node(child_node, child_prefix)
            if filtered_child is not None:
                filtered_children[child_name] = filtered_child

        # If no matches in this subtree, return None
        if not matching_direct and not filtered_children:
            return None

        # Create filtered node
        node = PrefixTreeNode(name=original_node.name)
        node.tensor_ids = matching_direct
        node.children = filtered_children

        # Compute aggregates
        direct_count = len(matching_direct)
        direct_bytes = sum(self.index.tensors[tid].nbytes for tid in matching_direct)
        child_count = sum(c.aggregate_count for c in filtered_children.values())
        child_bytes = sum(c.aggregate_bytes for c in filtered_children.values())

        node.aggregate_count = direct_count + child_count
        node.aggregate_bytes = direct_bytes + child_bytes

        return node

    def get_tensors_under(self, prefix: str) -> list[TensorInfo]:
        """Get all matching tensors under a given prefix."""
        if self.root is None:
            return []

        if not prefix:
            return [
                t
                for t in self.index.tensors
                if t.full_name in self.matching_tensor_names
            ]

        # Navigate to the prefix node
        parts = prefix.split(self.delimiter)
        node = self.root

        for part in parts:
            if part in node.children:
                node = node.children[part]
            else:
                return []

        # Collect all tensor IDs under this node
        tensor_ids = self._collect_tensor_ids(node)
        return [self.index.tensors[tid] for tid in tensor_ids]

    def _collect_tensor_ids(self, node: PrefixTreeNode) -> list[int]:
        """Recursively collect all tensor IDs under a node."""
        ids = list(node.tensor_ids)
        for child in node.children.values():
            ids.extend(self._collect_tensor_ids(child))
        return ids


class HierarchyTree(Tree):
    """Tree widget for navigating tensor namespaces."""

    BINDINGS = [
        Binding("left", "collapse_node", "Collapse", show=False),
        Binding("right", "expand_node", "Expand", show=False),
        Binding("enter", "toggle_node", "Toggle", show=False),
    ]

    class NodeSelected(Message):
        """Message sent when a tree node is selected or highlighted."""

        def __init__(self, prefix: str, node: PrefixTreeNode) -> None:
            self.prefix = prefix
            self.node = node
            super().__init__()

    def __init__(self, prefix_tree: PrefixTree) -> None:
        super().__init__("root")
        self.prefix_tree = prefix_tree
        self.filtered_tree: FilteredPrefixTree | None = None
        self._node_prefixes: dict[TreeNode, str] = {}

    @property
    def active_tree(self) -> PrefixTree | FilteredPrefixTree:
        """Return the currently active tree (filtered or original)."""
        return self.filtered_tree if self.filtered_tree else self.prefix_tree

    def on_mount(self) -> None:
        """Build the tree when mounted."""
        self._rebuild_tree_view()

    def _rebuild_tree_view(self) -> None:
        """Rebuild the tree view from the active tree."""
        # Clear existing tree
        self.root.remove_children()
        self._node_prefixes.clear()

        active = self.active_tree
        if active.root is None:
            # No matches - show empty state
            self.root.set_label(
                self._make_label(
                    self.prefix_tree.index.file_path.name + " (no matches)",
                    0,
                    0,
                )
            )
            self._node_prefixes[self.root] = ""
            return

        self.root.expand()
        self._build_tree(self.root, active.root, "")

        # Update root label
        self.root.set_label(
            self._make_label(
                self.prefix_tree.index.file_path.name,
                active.root.aggregate_count,
                active.root.aggregate_bytes,
            )
        )
        self._node_prefixes[self.root] = ""

    def apply_filter(
        self, matching_tensors: list[TensorInfo] | None, query: str = ""
    ) -> None:
        """Apply a filter to the tree, showing only matching tensors."""
        if matching_tensors is None:
            # Clear filter
            self.filtered_tree = None
            self.border_subtitle = ""
        else:
            # Create filtered tree
            self.filtered_tree = FilteredPrefixTree(self.prefix_tree, matching_tensors)
            # Show search query in border subtitle
            if query:
                self.border_subtitle = f"search: {query}"

        self._rebuild_tree_view()

    def _make_label(self, name: str, count: int, nbytes: int) -> Text:
        """Create a formatted label for a tree node."""
        label = Text()
        label.append(name, style="bold")
        label.append(f"  ({count}, {format_bytes(nbytes)})", style="dim")
        return label

    def _build_tree(self, parent: TreeNode, node: PrefixTreeNode, prefix: str) -> None:
        """Recursively build tree nodes."""
        for child_name, child_node in sorted(
            node.children.items(), key=lambda x: natural_sort_key(x[0])
        ):
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name

            # Use add_leaf for nodes without children (no expand icon)
            # Use add for nodes with children (expandable)
            if child_node.children:
                tree_node = parent.add(
                    self._make_label(
                        child_name,
                        child_node.aggregate_count,
                        child_node.aggregate_bytes,
                    ),
                    expand=False,
                )
                self._node_prefixes[tree_node] = child_prefix
                self._build_tree(tree_node, child_node, child_prefix)
            else:
                tree_node = parent.add_leaf(
                    self._make_label(
                        child_name,
                        child_node.aggregate_count,
                        child_node.aggregate_bytes,
                    ),
                )
                self._node_prefixes[tree_node] = child_prefix

    def _get_prefix_tree_node(self, tree_node: TreeNode) -> tuple[str, PrefixTreeNode]:
        """Get the prefix and PrefixTreeNode for a given tree node."""
        prefix = self._node_prefixes.get(tree_node, "")

        # Navigate to find the actual PrefixTreeNode in the active tree
        active = self.active_tree
        node = active.root
        if node is None:
            # Return a dummy empty node
            return prefix, PrefixTreeNode(name="")

        if prefix:
            for part in prefix.split(active.delimiter):
                if part in node.children:
                    node = node.children[part]
                else:
                    break

        return prefix, node

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Handle node highlight (cursor movement) - update right panel."""
        prefix, node = self._get_prefix_tree_node(event.node)
        self.post_message(self.NodeSelected(prefix, node))

    def action_toggle_node(self) -> None:
        """Toggle expand/collapse of the currently highlighted node."""
        if self.cursor_node and self.cursor_node.children:
            self.cursor_node.toggle()

    def action_collapse_node(self) -> None:
        """Collapse the currently highlighted node."""
        if self.cursor_node and self.cursor_node.is_expanded:
            self.cursor_node.collapse()
        elif self.cursor_node and self.cursor_node.parent:
            # If already collapsed, go to parent
            self.select_node(self.cursor_node.parent)

    def action_expand_node(self) -> None:
        """Expand the currently highlighted node."""
        if self.cursor_node and not self.cursor_node.is_expanded:
            self.cursor_node.expand()


class TensorTable(DataTable):
    """Table widget for displaying tensor information."""

    def __init__(self) -> None:
        super().__init__()
        self.cursor_type = "row"
        self.zebra_stripes = True
        self._tensors: list[TensorInfo] = []
        self._current_prefix: str = ""
        self._sort_mode: SortMode | None = None
        self._columns_initialized: bool = False

    def on_mount(self) -> None:
        """Set up table columns."""
        self._setup_columns()

    def _setup_columns(self) -> None:
        """Set up table columns (only once)."""
        if self._columns_initialized:
            return
        self.add_column("Name", key="name")
        self.add_column("Shape", key="shape")
        self.add_column("Dtype", key="dtype")
        self.add_column("Size", key="size")
        self._columns_initialized = True

    def _get_sort_indicator(self) -> str:
        """Get a string indicating the current sort mode."""
        if self._sort_mode is None:
            return ""
        return f" [{self._sort_mode.value}]"

    def update_tensors(self, tensors: list[TensorInfo], prefix: str = "") -> None:
        """Update the table with a list of tensors."""
        self._tensors = tensors
        self._current_prefix = prefix
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table contents."""
        self.clear()

        for tensor in self._tensors:
            # Add sort indicator to the first row's name if sorting is active
            self.add_row(
                tensor.full_name,
                format_shape(tensor.shape),
                tensor.dtype,
                format_bytes(tensor.nbytes),
                key=tensor.full_name,
            )

        # Update border subtitle to show sort mode
        if self._sort_mode:
            self.border_subtitle = f"sort: {self._sort_mode.value}"
        else:
            self.border_subtitle = ""

    def get_selected_tensor(self) -> TensorInfo | None:
        """Get the currently selected tensor."""
        if self.cursor_row is None or self.cursor_row >= len(self._tensors):
            return None
        return self._tensors[self.cursor_row]

    def sort_by(self, mode: SortMode) -> None:
        """Sort tensors by the given mode."""
        self._sort_mode = mode

        if mode == SortMode.NAME_ASC:
            self._tensors.sort(key=lambda t: natural_sort_key(t.full_name))
        elif mode == SortMode.NAME_DESC:
            self._tensors.sort(
                key=lambda t: natural_sort_key(t.full_name), reverse=True
            )
        elif mode == SortMode.SIZE_ASC:
            self._tensors.sort(key=lambda t: t.nbytes)
        elif mode == SortMode.SIZE_DESC:
            self._tensors.sort(key=lambda t: t.nbytes, reverse=True)
        elif mode == SortMode.RANK_ASC:
            self._tensors.sort(key=lambda t: (t.rank, natural_sort_key(t.full_name)))
        elif mode == SortMode.RANK_DESC:
            self._tensors.sort(key=lambda t: (-t.rank, natural_sort_key(t.full_name)))

        self._refresh_table()


class SearchInput(Input):
    """Search input widget."""

    DEFAULT_CSS = """
    SearchInput {
        display: none;
        height: 3;
        border: solid $accent;
        background: $surface;
    }

    SearchInput.visible {
        display: block;
    }
    """

    def __init__(self) -> None:
        super().__init__(placeholder="Type to search...")
        self.border_title = "Search (ESC to cancel)"


class SftApp(App):
    """Interactive browser for .safetensors files."""

    TITLE = "sft"

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr 2fr;
    }

    HierarchyTree {
        height: 100%;
        border: solid $primary;
        scrollbar-gutter: stable;
    }

    TensorTable {
        height: 100%;
        border: solid $secondary;
    }

    SearchInput {
        column-span: 2;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "toggle_panel", "Switch Panel", show=True),
        Binding("slash", "start_search", "Search", show=True),
        Binding("escape", "cancel_search", "Cancel", show=False),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("f", "show_filters", "Filter", show=True),
        Binding("space", "show_details", "Details", show=True),
        Binding("m", "show_metadata", "Metadata", show=True),
        Binding("l", "show_lora", "LoRA", show=True),
        Binding("g", "goto_top", "Top", show=False),
        Binding("G", "goto_bottom", "Bottom", show=False),
    ]

    def __init__(self, file_path: Path) -> None:
        """Initialize the app with a safetensors file path."""
        super().__init__()
        self.file_path = file_path
        self.index: TensorIndex | None = None
        self.prefix_tree: PrefixTree | None = None
        self._current_prefix: str = ""
        self._all_tensors: list[TensorInfo] = []
        self._base_tensors: list[TensorInfo] = []  # Before any filtering
        self._sort_mode_index: int = 0
        self._search_active: bool = False
        self._current_filters: dict = {}
        self._lora_pairs: list[LoraPair] = []
        self._tensor_lora_map: dict[str, LoraInfo] = {}
        self._lora_stats_cache: dict[
            str, dict
        ] = {}  # base_name → {norm_a, norm_b, eff_rank}

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Footer()

        # Parse the file
        try:
            self.index = TensorIndex.from_file(self.file_path)
            self.prefix_tree = PrefixTree(self.index)
            self._all_tensors = self.index.tensors.copy()
            self._base_tensors = self.index.tensors.copy()
            self._lora_pairs, self._tensor_lora_map = detect_lora_pairs(
                self.index.tensors
            )
            self._load_lora_stats_cache()
        except Exception as e:
            yield Static(f"Error loading file: {e}", id="error")
            return

        yield HierarchyTree(self.prefix_tree)
        yield TensorTable()
        yield SearchInput()

    def _load_lora_stats_cache(self) -> None:
        """Load previously exported LoRA analysis JSON if it exists."""
        json_path = self.file_path.with_suffix(".lora_analysis.json")
        if not json_path.exists():
            return
        try:
            with open(json_path) as f:
                data = json.load(f)
            for entry in data:
                name = entry.get("base_name", "")
                if not name:
                    continue
                cached: dict = {}
                if "norm_a" in entry:
                    cached["norm_a"] = float(entry["norm_a"])
                if "norm_b" in entry:
                    cached["norm_b"] = float(entry["norm_b"])
                if "eff_rank" in entry:
                    cached["eff_rank"] = float(entry["eff_rank"])
                if cached:
                    self._lora_stats_cache[name] = cached
        except Exception:
            pass  # Silently ignore corrupt/unreadable files

    def on_mount(self) -> None:
        """Initialize the view after mounting."""
        if self.index is None:
            return

        # Show all tensors initially
        table = self.query_one(TensorTable)
        table.update_tensors(self.index.tensors)

        # Focus the tree
        tree = self.query_one(HierarchyTree)
        tree.focus()

    def action_quit(self) -> None:
        """Shut down any running computation threads and quit."""
        for screen in self.screen_stack:
            if isinstance(screen, LoraScreen):
                screen._shutdown_pool()
        self.workers.cancel_all()
        self.exit()

    def on_hierarchy_tree_node_selected(
        self, event: HierarchyTree.NodeSelected
    ) -> None:
        """Handle tree node selection."""
        self._current_prefix = event.prefix

        # Get tensors under this prefix from the active tree
        tree = self.query_one(HierarchyTree)
        tensors = tree.active_tree.get_tensors_under(event.prefix)
        self._base_tensors = tensors.copy()

        # Apply any active dtype filters
        if self._current_filters:
            self._apply_filters()
        else:
            self._all_tensors = tensors.copy()

            # Update tensor table
            table = self.query_one(TensorTable)
            table.update_tensors(tensors, event.prefix)

            # Apply current sort
            if self._sort_mode_index > 0:
                table.sort_by(SORT_ORDER[self._sort_mode_index])

    def action_toggle_panel(self) -> None:
        """Toggle focus between tree and table panels."""
        tree = self.query_one(HierarchyTree)
        table = self.query_one(TensorTable)

        if tree.has_focus:
            table.focus()
        else:
            tree.focus()

    def action_start_search(self) -> None:
        """Start search mode."""
        search_input = self.query_one(SearchInput)
        search_input.add_class("visible")
        search_input.focus()
        self._search_active = True

    def action_cancel_search(self) -> None:
        """Cancel search and restore full list."""
        search_input = self.query_one(SearchInput)
        search_input.remove_class("visible")
        search_input.value = ""
        self._search_active = False

        # Clear tree filter
        tree = self.query_one(HierarchyTree)
        tree.apply_filter(None)

        # Reset to show all tensors
        self._current_prefix = ""
        self._base_tensors = self.index.tensors.copy()
        self._all_tensors = self.index.tensors.copy()

        # Restore full tensor list
        table = self.query_one(TensorTable)
        table.update_tensors(self._all_tensors, self._current_prefix)

        # Apply current sort
        if self._sort_mode_index > 0:
            table.sort_by(SORT_ORDER[self._sort_mode_index])

        # Focus tree
        tree.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if not self._search_active:
            return

        query = event.value.lower()
        tree = self.query_one(HierarchyTree)
        table = self.query_one(TensorTable)

        if query:
            # Filter tensors from the full index (not current selection)
            filtered = [t for t in self.index.tensors if query in t.full_name.lower()]

            # Update tree with filter (pass query for display)
            tree.apply_filter(filtered, query)

            # Update table with filtered tensors
            self._current_prefix = ""
            self._base_tensors = filtered
            self._all_tensors = filtered
            table.update_tensors(filtered, "")

            # Apply current sort
            if self._sort_mode_index > 0:
                table.sort_by(SORT_ORDER[self._sort_mode_index])
        else:
            # Clear filter
            tree.apply_filter(None)
            self._current_prefix = ""
            self._base_tensors = self.index.tensors.copy()
            self._all_tensors = self.index.tensors.copy()
            table.update_tensors(self.index.tensors, "")

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Handle search input submission."""
        # Keep the search active, just focus the table
        table = self.query_one(TensorTable)
        table.focus()

    def action_cycle_sort(self) -> None:
        """Cycle through sort modes."""
        self._sort_mode_index = (self._sort_mode_index + 1) % len(SORT_ORDER)
        mode = SORT_ORDER[self._sort_mode_index]

        table = self.query_one(TensorTable)
        table.sort_by(mode)

    def action_show_details(self) -> None:
        """Show tensor details popup."""
        table = self.query_one(TensorTable)
        tensor = table.get_selected_tensor()

        if tensor:
            lora_info = self._tensor_lora_map.get(tensor.full_name)
            self.push_screen(TensorDetailScreen(tensor, lora_info))

    def action_show_metadata(self) -> None:
        """Show file metadata popup."""
        if self.index:
            self.push_screen(MetadataScreen(self.index.metadata, self.file_path))

    def action_show_lora(self) -> None:
        """Show LoRA analysis, filtered to visible tensors. Single match → SVD directly."""
        if not self._lora_pairs:
            self.notify("No LoRA pairs detected in this file", severity="warning")
            return

        # Filter pairs to those with at least one tensor in the current view
        visible_names = {t.full_name for t in self._all_tensors}
        filtered_pairs = [
            p
            for p in self._lora_pairs
            if p.a_tensor_name in visible_names or p.b_tensor_name in visible_names
        ]

        if not filtered_pairs:
            # Fall back to all pairs
            filtered_pairs = self._lora_pairs

        if len(filtered_pairs) == 1:
            # Single pair — go straight to SVD
            self.push_screen(SvdScreen(filtered_pairs[0], self.file_path, self.index))
        else:
            self.push_screen(
                LoraScreen(
                    filtered_pairs,
                    self.file_path,
                    self.index,
                    self._lora_stats_cache,
                )
            )

    def action_show_filters(self) -> None:
        """Show filter palette."""
        if self.index is None:
            return

        # Get available dtypes
        available_dtypes = {t.dtype for t in self.index.tensors}

        def on_filter_result(filters: dict) -> None:
            """Handle filter result."""
            self._current_filters = filters
            self._apply_filters()

        self.push_screen(
            FilterScreen(self._current_filters, available_dtypes),
            on_filter_result,
        )

    def _apply_filters(self) -> None:
        """Apply current filters to the tensor list."""
        # Start from base tensors (all tensors under current prefix)
        tensors = self._base_tensors.copy()

        # Apply dtype filter
        if "dtypes" in self._current_filters and self._current_filters["dtypes"]:
            allowed = set(self._current_filters["dtypes"])
            tensors = [t for t in tensors if t.dtype in allowed]

        self._all_tensors = tensors

        # Update table
        table = self.query_one(TensorTable)
        table.update_tensors(tensors, self._current_prefix)

        # Apply current sort
        if self._sort_mode_index > 0:
            table.sort_by(SORT_ORDER[self._sort_mode_index])

    def action_goto_top(self) -> None:
        """Go to top of current focused widget."""
        focused = self.focused
        if isinstance(focused, DataTable):
            focused.move_cursor(row=0)
        elif isinstance(focused, Tree):
            focused.select_node(focused.root)

    def action_goto_bottom(self) -> None:
        """Go to bottom of current focused widget."""
        focused = self.focused
        if isinstance(focused, DataTable):
            focused.move_cursor(row=focused.row_count - 1)
