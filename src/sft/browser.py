"""Textual TUI application for browsing safetensors files."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from rich.json import JSON
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    DataTable,
    DirectoryTree,
    Footer,
    Input,
    Label,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

try:
    from textual.command import DiscoveryHit, Hit, Hits, Provider

    _HAS_COMMAND_PALETTE = True
except ImportError:  # pragma: no cover
    _HAS_COMMAND_PALETTE = False

from sft.index import (
    PrefixTree,
    PrefixTreeNode,
    TensorIndex,
    TensorInfo,
    natural_sort_key,
)
from sft.ops.lora.detect import LoRAInfo, LoRAPair
from sft.utils.formatting import format_bytes, format_dtype, format_number, format_shape


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


class MetadataScreen(ModalScreen):
    """Modal screen showing file metadata."""

    CSS = """
    MetadataScreen {
        align: center middle;
    }

    #metadata-container {
        width: 80%;
        max-width: 120;
        min-width: 60;
        height: 80%;
        background: $surface;
        border: thick $secondary;
        padding: 1 2;
    }

    #metadata-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #metadata-scroll {
        height: 1fr;
        margin-top: 1;
    }

    #metadata-scroll:focus {
        border-left: tall $secondary;
    }

    #metadata-content {
        height: auto;
        width: auto;
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

    @staticmethod
    def _expand(metadata: dict) -> dict:
        """Expand values that are themselves JSON-encoded strings so they render
        as nested structures rather than one long escaped blob."""
        expanded: dict = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                stripped = value.strip()
                if stripped[:1] in "{[" or stripped in ("true", "false", "null"):
                    try:
                        expanded[key] = json.loads(stripped)
                        continue
                    except (json.JSONDecodeError, ValueError):
                        pass
            expanded[key] = value
        return expanded

    def compose(self) -> ComposeResult:
        with Container(id="metadata-container"):
            yield Label("File Metadata", id="metadata-title")
            yield Static(f"[dim]File:[/dim] {self.file_path.name}")

            if self.metadata:
                formatted = json.dumps(self._expand(self.metadata), indent=2)
                scroll = VerticalScroll(
                    Static(JSON(formatted), id="metadata-content"),
                    id="metadata-scroll",
                )
                scroll.can_focus = True
                yield scroll
            else:
                yield Static("\n[dim]No metadata found in file[/dim]")

            yield Static("\n[dim]Press ESC or M to close · ↑↓/PgUp/PgDn to scroll[/dim]")

    def on_mount(self) -> None:
        # Focus the scroll region so arrow / page keys scroll it immediately.
        scroll = self.query("#metadata-scroll")
        if scroll:
            scroll.first().focus()


class TensorStatsScreen(ModalScreen):
    """Modal screen showing computed tensor statistics."""

    CSS = """
    TensorStatsScreen {
        align: center middle;
    }

    #stats-container {
        width: 70%;
        max-width: 120;
        min-width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }

    #stats-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    .stats-row {
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("enter", "dismiss", "Close"),
    ]

    def __init__(self, tensor: TensorInfo, file_path: Path) -> None:
        super().__init__()
        self.tensor = tensor
        self.file_path = file_path

    def compose(self) -> ComposeResult:
        import numpy as np

        from sft.utils.tensor_io import read_tensor

        t = self.tensor

        with Container(id="stats-container"):
            yield Label("Tensor Statistics", id="stats-title")
            yield Static(f"[dim]Name:[/dim]  {t.full_name}", classes="stats-row")
            yield Static(
                f"[dim]Shape:[/dim] {format_shape(t.shape)}", classes="stats-row"
            )
            yield Static(
                f"[dim]Dtype:[/dim] {format_dtype(t.dtype)}", classes="stats-row"
            )
            yield Static(
                f"[dim]Size:[/dim]  {format_bytes(t.nbytes)}", classes="stats-row"
            )
            yield Static(f"[dim]Numel:[/dim] {t.numel:,}", classes="stats-row")

            yield Static("", classes="stats-row")

            try:
                data = read_tensor(self.file_path, t.full_name)
                float_data = data.astype(np.float64)

                mean_val = float(np.nanmean(float_data))
                std_val = float(np.nanstd(float_data))
                min_val = float(np.nanmin(float_data))
                max_val = float(np.nanmax(float_data))
                nan_count = int(np.isnan(float_data).sum())
                inf_count = int(np.isinf(float_data).sum())
                zero_count = int((data == 0).sum())
                sparsity = 100.0 * zero_count / max(data.size, 1)

                yield Static(
                    f"[dim]Mean:[/dim]     {mean_val:.6f}", classes="stats-row"
                )
                yield Static(f"[dim]Std:[/dim]      {std_val:.6f}", classes="stats-row")
                yield Static(f"[dim]Min:[/dim]      {min_val:.6f}", classes="stats-row")
                yield Static(f"[dim]Max:[/dim]      {max_val:.6f}", classes="stats-row")
                yield Static(
                    f"[dim]Sparsity:[/dim] {sparsity:.2f}% ({zero_count:,} zeros)",
                    classes="stats-row",
                )
                yield Static(f"[dim]NaN:[/dim]      {nan_count:,}", classes="stats-row")
                yield Static(f"[dim]Inf:[/dim]      {inf_count:,}", classes="stats-row")
            except Exception as e:
                yield Static(
                    f"[red]Error computing stats: {e}[/red]", classes="stats-row"
                )

            yield Static("\n[dim]Press ESC or S to close[/dim]", classes="stats-row")


class CastScreen(ModalScreen):
    """Modal screen for casting a file to a different dtype."""

    DTYPES = ["fp16", "fp32", "bf16"]

    CSS = """
    CastScreen {
        align: center middle;
    }

    #cast-container {
        width: 50;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $success;
        padding: 1 2;
    }

    #cast-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("1", "select_0", show=False),
        Binding("2", "select_1", show=False),
        Binding("3", "select_2", show=False),
    ]

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path = file_path

    def compose(self) -> ComposeResult:
        with Container(id="cast-container"):
            yield Label(f"Cast {self.file_path.name} to:", id="cast-title")
            for i, dtype in enumerate(self.DTYPES):
                yield Static(f"  [{i + 1}] {dtype}")
            yield Static("\n[dim]Press number to cast, ESC to cancel[/dim]")

    def _select(self, idx: int) -> None:
        if idx < len(self.DTYPES):
            self.dismiss(self.DTYPES[idx])

    def action_select_0(self) -> None:
        self._select(0)

    def action_select_1(self) -> None:
        self._select(1)

    def action_select_2(self) -> None:
        self._select(2)

    def action_cancel(self) -> None:
        self.dismiss(None)


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
        self._sort_mode: SortMode = SortMode.NAME_ASC
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

    # Map sort modes to (column_key, suffix_with_arrow)
    _SORT_COLUMN_MAP: dict[SortMode, tuple[str, str]] = {
        SortMode.NAME_ASC: ("name", " ↑"),
        SortMode.NAME_DESC: ("name", " ↓"),
        SortMode.SIZE_ASC: ("size", " ↑"),
        SortMode.SIZE_DESC: ("size", " ↓"),
        SortMode.RANK_ASC: ("shape", " (rank ↑)"),
        SortMode.RANK_DESC: ("shape", " (rank ↓)"),
    }

    _BASE_LABELS: dict[str, str] = {
        "name": "Name",
        "shape": "Shape",
        "dtype": "Dtype",
        "size": "Size",
    }

    def update_tensors(self, tensors: list[TensorInfo], prefix: str = "") -> None:
        """Update the table with a list of tensors."""
        self._tensors = tensors
        self._current_prefix = prefix
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table contents."""
        self.clear()

        for tensor in self._tensors:
            self.add_row(
                tensor.full_name,
                format_shape(tensor.shape),
                format_dtype(tensor.dtype),
                format_bytes(tensor.nbytes),
                key=tensor.full_name,
            )

        self._update_column_labels()

    def _update_column_labels(self) -> None:
        """Update column headers with sort arrow indicators."""
        if not self._columns_initialized:
            return

        sorted_col, arrow = self._SORT_COLUMN_MAP[self._sort_mode]

        for key, base_label in self._BASE_LABELS.items():
            label_text = base_label
            if key == sorted_col:
                label_text += arrow
            col = self.columns.get(key)
            if col is not None:
                col.label = Text.from_markup(label_text)

        self.refresh()

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


if _HAS_COMMAND_PALETTE:

    class SftCommands(Provider):
        """Command palette provider for sft operations."""

        async def search(self, query: str) -> Hits:
            app = self.app
            assert isinstance(app, SftApp)

            commands: list[tuple[str, str, str]] = [
                ("File info", "Show file summary", "show_info"),
                ("Show metadata", "View file metadata", "show_metadata"),
                (
                    "Tensor details",
                    "Show details for the selected tensor",
                    "show_details",
                ),
                ("Show stats", "Compute statistics for selected tensor", "show_stats"),
                ("Cast to fp16", "Cast all tensors to float16", "cast_fp16"),
                ("Cast to fp32", "Cast all tensors to float32", "cast_fp32"),
                ("Cast to bf16", "Cast all tensors to bfloat16", "cast_bf16"),
                ("Check file", "Validate file integrity", "check_file"),
                ("Diff file", "Compare against another safetensors file", "diff_file"),
                ("LoRA mode", "Enter LoRA analysis mode", "show_lora_mode"),
                ("Search tensors", "Filter tensors by name", "start_search"),
                ("Cycle sort", "Cycle through sort modes", "cycle_sort"),
            ]

            matcher = self.matcher(query)
            for name, help_text, action_name in commands:
                score = matcher.match(name)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(name),
                        getattr(app, f"_do_{action_name}"),
                        help=help_text,
                    )

        async def discover(self) -> Hits:
            app = self.app
            assert isinstance(app, SftApp)

            commands: list[tuple[str, str, str]] = [
                ("File info", "Show file summary", "show_info"),
                ("Show metadata", "View file metadata", "show_metadata"),
                (
                    "Tensor details",
                    "Show details for the selected tensor",
                    "show_details",
                ),
                ("Show stats", "Compute statistics for selected tensor", "show_stats"),
                ("Cast to fp16", "Cast all tensors to float16", "cast_fp16"),
                ("Cast to fp32", "Cast all tensors to float32", "cast_fp32"),
                ("Cast to bf16", "Cast all tensors to bfloat16", "cast_bf16"),
                ("Check file", "Validate file integrity", "check_file"),
                ("Diff file", "Compare against another safetensors file", "diff_file"),
                ("LoRA mode", "Enter LoRA analysis mode", "show_lora_mode"),
                ("Search tensors", "Filter tensors by name", "start_search"),
                ("Cycle sort", "Cycle through sort modes", "cycle_sort"),
            ]

            for name, help_text, action_name in commands:
                yield DiscoveryHit(
                    name,
                    getattr(app, f"_do_{action_name}"),
                    help=help_text,
                )


class LoraSortMode(Enum):
    """Sort modes for the LoRA Analysis table."""

    MODULE_ASC = "module ↑"
    MODULE_DESC = "module ↓"
    RANK_DESC = "rank ↓"
    RANK_ASC = "rank ↑"
    EFF_RANK_DESC = "eff. rank ↓"
    EFF_RANK_ASC = "eff. rank ↑"
    SV95_DESC = "sv95 ↓"
    SV95_ASC = "sv95 ↑"
    NORM_A_DESC = "‖A‖ ↓"
    NORM_A_ASC = "‖A‖ ↑"
    NORM_B_DESC = "‖B‖ ↓"
    NORM_B_ASC = "‖B‖ ↑"


LORA_SORT_ORDER = [
    LoraSortMode.MODULE_ASC,
    LoraSortMode.MODULE_DESC,
    LoraSortMode.RANK_DESC,
    LoraSortMode.RANK_ASC,
    LoraSortMode.EFF_RANK_DESC,
    LoraSortMode.EFF_RANK_ASC,
    LoraSortMode.SV95_DESC,
    LoraSortMode.SV95_ASC,
    LoraSortMode.NORM_A_DESC,
    LoraSortMode.NORM_A_ASC,
    LoraSortMode.NORM_B_DESC,
    LoraSortMode.NORM_B_ASC,
]


class KohyaConvertScreen(ModalScreen):
    """Confirmation modal for Kohya<->PEFT auto-detect conversion."""

    CSS = """
    KohyaConvertScreen {
        align: center middle;
    }

    #kohya-container {
        width: 75%;
        max-width: 120;
        min-width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $success;
        padding: 1 2;
    }

    #kohya-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    .kohya-row {
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Convert"),
    ]

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path = file_path
        self.source: str | None = None
        self.target: str | None = None
        self.modules_kohya = 0
        self.modules_peft = 0

    def compose(self) -> ComposeResult:
        from sft.ops.lora.convert import detect_format
        from sft.utils.output import resolve_output

        info = detect_format(self.file_path)
        self.source = info.format
        with Container(id="kohya-container"):
            yield Label("Kohya \u2194 PEFT Conversion", id="kohya-title")
            yield Static(f"[dim]File:[/dim] {self.file_path.name}", classes="kohya-row")
            yield Static(
                f"[dim]Kohya modules:[/dim] {info.kohya_modules}    "
                f"[dim]PEFT modules:[/dim] {info.peft_modules}    "
                f"[dim]Other tensors:[/dim] {info.non_lora}",
                classes="kohya-row",
            )
            yield Static("", classes="kohya-row")

            if self.source is None:
                yield Static(
                    "[red]No LoRA tensors detected. Nothing to convert.[/red]",
                    classes="kohya-row",
                )
                yield Static("\n[dim]Press ESC to close.[/dim]", classes="kohya-row")
                return
            if self.source == "mixed":
                yield Static(
                    "[red]File contains both Kohya and PEFT modules; "
                    "normalize manually first.[/red]",
                    classes="kohya-row",
                )
                yield Static("\n[dim]Press ESC to close.[/dim]", classes="kohya-row")
                return

            self.target = "peft" if self.source == "kohya" else "kohya"
            output = resolve_output(None, self.file_path, self.target)
            yield Static(
                f"[bold]Detected:[/bold] [cyan]{self.source}[/cyan]    "
                f"[bold]Will write:[/bold] [cyan]{self.target}[/cyan]",
                classes="kohya-row",
            )
            yield Static(f"[dim]Output:[/dim] {output.name}", classes="kohya-row")
            yield Static(
                "\n[dim]ENTER: convert    ESC: cancel[/dim]",
                classes="kohya-row",
            )

    def action_confirm(self) -> None:
        if self.source in (None, "mixed"):
            self.dismiss(None)
            return
        self.dismiss(self.target)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SafetensorsDirectoryTree(DirectoryTree):
    """DirectoryTree that hides everything except directories and .safetensors files."""

    def filter_paths(self, paths):
        return [
            p
            for p in paths
            if p.is_dir() or (p.is_file() and p.suffix.lower() == ".safetensors")
        ]


class DiffFilePickerScreen(ModalScreen):
    """File picker modal for choosing a second .safetensors file to diff against."""

    CSS = """
    DiffFilePickerScreen {
        align: center middle;
    }

    #picker-container {
        width: 90%;
        height: 85%;
        min-width: 50;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }

    #picker-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #picker-path {
        height: 1;
        color: $text-muted;
        margin-top: 1;
    }

    #picker-tree {
        height: 1fr;
    }

    #picker-footer {
        dock: bottom;
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, source_file: Path) -> None:
        super().__init__()
        self.source_file = source_file
        # Start at parent so the user sees the source file's siblings
        self.start_dir = source_file.parent

    def compose(self) -> ComposeResult:
        with Container(id="picker-container"):
            yield Label("Pick a second .safetensors file", id="picker-title")
            yield Static(f"[dim]Comparing against:[/dim] {self.source_file.name}")
            yield SafetensorsDirectoryTree(str(self.start_dir), id="picker-tree")
            yield Static("", id="picker-path")
            yield Static(
                "[dim]\u2191/\u2193:[/dim] navigate  "
                "[dim]enter:[/dim] expand / select  "
                "[dim]esc:[/dim] cancel",
                id="picker-footer",
            )

    def on_mount(self) -> None:
        self.query_one(SafetensorsDirectoryTree).focus()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """User confirmed a file selection."""
        selected = Path(event.path)
        if selected.resolve() == self.source_file.resolve():
            self.notify(
                "That's the same file — pick a different one.",
                severity="warning",
            )
            return
        self.dismiss(selected)

    def on_directory_tree_file_highlighted(
        self, event: DirectoryTree.FileHighlighted
    ) -> None:
        try:
            label = self.query_one("#picker-path", Static)
        except Exception:
            return
        label.update(f"[dim]Selected:[/dim] {event.path}")

    def action_cancel(self) -> None:
        self.dismiss(None)


class DiffResultScreen(ModalScreen):
    """Show the result of comparing two safetensors files as a table.

    Rows are classified into five buckets:

      • equal / close / differ — comparable tensors (same name+shape+dtype),
        with per-tensor metrics: max_abs, rel_L2, cosine. The boundary
        between close/differ is rtol/atol-based (numpy.allclose).
      • incompatible — same name but shape or dtype differs.
      • missing — present in only one file.

    Filter keys (a/d/e/m/i) toggle which categories are visible.
    """

    CSS = """
    DiffResultScreen {
        align: center middle;
    }

    #diff-container {
        width: 95%;
        height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #diff-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #diff-summary {
        color: $text-muted;
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }

    #diff-table {
        height: 1fr;
        width: 100%;
    }

    #diff-footer {
        dock: bottom;
        height: 1;
        color: $text-muted;
    }
    """

    STATUS_STYLES = {
        "equal": "green",
        "close": "cyan",
        "differ": "red",
        "incompatible": "magenta",
        "missing": "yellow",
    }

    # Default visibility: hide "equal" (which is the bulk of any diff) so the
    # user sees changes first. They can press `a` to show everything.
    _DEFAULT_VISIBLE: frozenset[str] = frozenset(
        {"close", "differ", "incompatible", "missing"}
    )

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("a", "filter_all", "All", show=True),
        Binding("d", "filter_differ", "Differ", show=True),
        Binding("e", "filter_equal", "Equal", show=True),
        Binding("m", "filter_missing", "Missing", show=True),
        Binding("i", "filter_incompatible", "Incompatible", show=True),
    ]

    def __init__(self, file_a: Path, file_b: Path) -> None:
        super().__init__()
        self.file_a = file_a
        self.file_b = file_b
        self._loaded = False
        # Internal row model — populated by the worker, then filtered for display.
        self._rows: list[tuple[str, str, str, str, str, str]] = []
        # Active filter (which statuses to show)
        self._visible: set[str] = set(self._DEFAULT_VISIBLE)

    def compose(self) -> ComposeResult:
        with Container(id="diff-container"):
            yield Label(
                f"Diff: {self.file_a.name} \u2194 {self.file_b.name}",
                id="diff-title",
            )
            yield Static("[dim]Computing diff...[/dim]", id="diff-summary")
            table: DataTable = DataTable(id="diff-table", zebra_stripes=True)
            table.cursor_type = "row"
            yield table
            yield Static(
                "[dim]a/d/e/m/i: filter \u2022 esc: close[/dim]",
                id="diff-footer",
            )

    def on_mount(self) -> None:
        self._run_diff()

    @work(thread=True, exclusive=True)
    def _run_diff(self) -> None:
        from sft.ops.diff import diff_files

        try:
            result = diff_files(self.file_a, self.file_b, compute_delta=True)
        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Diff failed: {e}", severity="error"
            )
            self.app.call_from_thread(self.dismiss, None)
            return
        self.app.call_from_thread(self._populate_rows, result)

    def _populate_rows(self, result) -> None:
        """Build the in-memory row model from a diff result and render."""
        self._rows = []

        # Comparable tensors — each has a precise equal/close/differ status
        if result.value_diffs is not None:
            for name, vd in result.value_diffs.items():
                self._rows.append(
                    (
                        vd.status,
                        name,
                        f"max_abs={vd.max_abs:.3e}",
                        f"{vd.rel_l2:.3e}",
                        f"{vd.cosine_sim:.4f}",
                        "",
                    )
                )

        # Incompatible — shape or dtype mismatch on same name
        for name, (sa, sb) in result.shape_changed.items():
            self._rows.append(
                (
                    "incompatible",
                    name,
                    f"shape {sa}\u2192{sb}",
                    "-",
                    "-",
                    "",
                )
            )
        for name, (da, db) in result.dtype_changed.items():
            self._rows.append(
                (
                    "incompatible",
                    name,
                    f"dtype {da}\u2192{db}",
                    "-",
                    "-",
                    "",
                )
            )

        # Missing — present in only one file
        for name in result.added:
            self._rows.append(("missing", name, "only in B", "-", "-", ""))
        for name in result.removed:
            self._rows.append(("missing", name, "only in A", "-", "-", ""))

        # Counts for the summary line
        self._counts = {
            "equal": sum(1 for r in self._rows if r[0] == "equal"),
            "close": sum(1 for r in self._rows if r[0] == "close"),
            "differ": sum(1 for r in self._rows if r[0] == "differ"),
            "incompatible": sum(1 for r in self._rows if r[0] == "incompatible"),
            "missing": sum(1 for r in self._rows if r[0] == "missing"),
        }

        # Set up table columns once
        table = self.query_one("#diff-table", DataTable)
        if not table.columns:
            table.add_columns("Status", "Tensor", "Detail", "rel_L2", "cosine", "")

        self._loaded = True
        self._refresh_view()
        table.focus()

    def _refresh_view(self) -> None:
        """Re-render the table from `self._rows` filtered by `self._visible`.

        Named `_refresh_view` (not `_render`) to avoid shadowing
        `textual.widget.Widget._render`, which would break screen mounting.
        """
        table = self.query_one("#diff-table", DataTable)
        table.clear()
        for status, name, detail, rel, cos, _ in self._rows:
            if status not in self._visible:
                continue
            color = self.STATUS_STYLES.get(status, "white")
            table.add_row(
                Text(status, style=color),
                name,
                detail,
                rel,
                cos,
                "",
            )

        # Update summary
        summary = self.query_one("#diff-summary", Static)
        c = self._counts
        active = ", ".join(sorted(self._visible)) if self._visible else "none"
        summary.update(
            f"[green]equal:[/green] {c['equal']}   "
            f"[cyan]close:[/cyan] {c['close']}   "
            f"[red]differ:[/red] {c['differ']}   "
            f"[magenta]incompatible:[/magenta] {c['incompatible']}   "
            f"[yellow]missing:[/yellow] {c['missing']}   "
            f"[dim]\u2022 showing: {active}[/dim]"
        )

    def _set_filter(self, statuses: set[str]) -> None:
        self._visible = statuses
        if self._loaded:
            self._refresh_view()

    def action_filter_all(self) -> None:
        self._set_filter({"equal", "close", "differ", "incompatible", "missing"})

    def action_filter_differ(self) -> None:
        self._set_filter({"differ"})

    def action_filter_equal(self) -> None:
        # "equal" filter shows both bitwise-equal and within-tolerance tensors
        self._set_filter({"equal", "close"})

    def action_filter_missing(self) -> None:
        self._set_filter({"missing"})

    def action_filter_incompatible(self) -> None:
        self._set_filter({"incompatible"})


class LoraModeScreen(Screen):
    """Dedicated full-screen mode for everything LoRA.

    This is the single home for LoRA-specific operations. The main browser
    is intentionally LoRA-agnostic — pressing `L` enters this mode.

    Layout:
      • Rich header: file, format, rank, alpha/scale, modules, params
      • Pair DataTable: module, rank, eff. rank, SV95, ‖A‖, ‖B‖
      • Footer with sub-commands

    For Kohya files, the pair table is hidden and only `k` (convert)
    is offered — analyze/compress require PEFT layout.

    Stats are computed in a background thread so the UI stays responsive
    on adapters with hundreds of pairs.
    """

    CSS = """
    LoraModeScreen {
        layout: vertical;
    }

    #lora-header {
        height: auto;
        background: $accent 30%;
        border: tall $accent;
        padding: 0 2;
    }

    #lora-title-line {
        text-style: bold;
    }

    #lora-meta-line {
        color: $text-muted;
    }

    #lora-body {
        height: 1fr;
        width: 100%;
        padding: 1 2;
    }

    #lora-table {
        height: 1fr;
        width: 100%;
    }

    #lora-kohya-notice {
        height: auto;
        padding: 2 4;
        background: $warning 20%;
        border: tall $warning;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "exit_mode", "Back", show=True),
        Binding("L", "exit_mode", "Back", show=False),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("c", "compress", "Compress", show=True),
        Binding("k", "convert_format", "Convert", show=True),
        Binding("i", "show_info", "Info", show=True),
        # Explicitly suppress App-level bindings that don't apply in LoRA Mode.
        # Textual's binding lookup is screen-first, so these overrides hide
        # the (now-irrelevant) main-browser shortcuts from the footer.
        Binding("slash", "noop", "", show=False),
        Binding("m", "noop", "", show=False),
        Binding("enter", "noop", "", show=False),
        Binding("D", "noop", "", show=False),
        Binding("tab", "noop", "", show=False),
        Binding("g", "noop", "", show=False),
        Binding("G", "noop", "", show=False),
    ]

    def action_noop(self) -> None:
        """No-op: swallow App-level bindings that don't apply in LoRA Mode."""

    def __init__(
        self,
        file_path: Path,
        index: TensorIndex,
        lora_format: str,
        lora_info: LoRAInfo | None,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.index = index
        self.lora_format = lora_format  # 'peft' or 'kohya'
        self.lora_info = lora_info  # only populated for PEFT
        self._sort_idx = 0
        self._stats: dict[str, dict[str, float]] = {}
        self._tensor_cache: dict | None = None

    def compose(self) -> ComposeResult:
        yield Footer()
        with Container(id="lora-header"):
            yield Static(self._title_line(), id="lora-title-line")
            yield Static(self._meta_line(), id="lora-meta-line")
        with Container(id="lora-body"):
            if self.lora_format == "peft" and self.lora_info:
                table: DataTable = DataTable(id="lora-table", zebra_stripes=True)
                table.cursor_type = "row"
                yield table
            else:
                yield Static(
                    "[bold yellow]Kohya-format LoRA detected.[/bold yellow]\n\n"
                    "Analyze and Compress operate on PEFT-layout adapters.\n"
                    "Press [bold]k[/bold] to convert this file to PEFT first,\n"
                    "then re-open the converted file with [bold]sft browse[/bold].",
                    id="lora-kohya-notice",
                )

    _LORA_BASE_LABELS: dict[str, str] = {
        "module": "Module",
        "rank": "Rank",
        "eff_rank": "Eff. Rank",
        "sv95": "SV95",
        "norm_a": "‖A‖",
        "norm_b": "‖B‖",
    }

    _LORA_SORT_COLUMN_MAP: dict[LoraSortMode, tuple[str, str]] = {
        LoraSortMode.MODULE_ASC: ("module", " ↑"),
        LoraSortMode.MODULE_DESC: ("module", " ↓"),
        LoraSortMode.RANK_DESC: ("rank", " ↓"),
        LoraSortMode.RANK_ASC: ("rank", " ↑"),
        LoraSortMode.EFF_RANK_DESC: ("eff_rank", " ↓"),
        LoraSortMode.EFF_RANK_ASC: ("eff_rank", " ↑"),
        LoraSortMode.SV95_DESC: ("sv95", " ↓"),
        LoraSortMode.SV95_ASC: ("sv95", " ↑"),
        LoraSortMode.NORM_A_DESC: ("norm_a", " ↓"),
        LoraSortMode.NORM_A_ASC: ("norm_a", " ↑"),
        LoraSortMode.NORM_B_DESC: ("norm_b", " ↓"),
        LoraSortMode.NORM_B_ASC: ("norm_b", " ↑"),
    }

    _LORA_COLUMN_SORT: dict[str, tuple[LoraSortMode, LoraSortMode]] = {
        "module": (LoraSortMode.MODULE_ASC, LoraSortMode.MODULE_DESC),
        "rank": (LoraSortMode.RANK_ASC, LoraSortMode.RANK_DESC),
        "eff_rank": (LoraSortMode.EFF_RANK_ASC, LoraSortMode.EFF_RANK_DESC),
        "sv95": (LoraSortMode.SV95_ASC, LoraSortMode.SV95_DESC),
        "norm_a": (LoraSortMode.NORM_A_ASC, LoraSortMode.NORM_A_DESC),
        "norm_b": (LoraSortMode.NORM_B_ASC, LoraSortMode.NORM_B_DESC),
    }

    _SORT_MISSING_ASC = float("inf")
    _SORT_MISSING_DESC = float("-inf")

    def on_mount(self) -> None:
        if self.lora_format == "peft" and self.lora_info:
            table = self.query_one("#lora-table", DataTable)
            table.add_column("Module", key="module")
            table.add_column("Rank", key="rank")
            table.add_column("Eff. Rank", key="eff_rank")
            table.add_column("SV95", key="sv95")
            table.add_column("‖A‖", key="norm_a")
            table.add_column("‖B‖", key="norm_b")
            # Pre-size each column to its widest possible label (base + arrow).
            # add_column sets auto_width=True, so content_width (not width) drives
            # the render width. Seed it here so no column starts too narrow.
            arrow_widths = {
                col_key: len(self._LORA_BASE_LABELS[col_key] + arrow)
                for _, (col_key, arrow) in self._LORA_SORT_COLUMN_MAP.items()
            }
            for key, base_label in self._LORA_BASE_LABELS.items():
                target_w = arrow_widths.get(key, len(base_label))
                col = table.columns.get(key)
                if col is not None:
                    col.content_width = max(col.content_width, target_w)
            self._populate_initial()
            table.focus()
            self._compute_stats()
            self._update_sort_indicator()

    _STATS_REFRESH_EVERY = 64

    def _title_line(self) -> str:
        fmt_label = self.lora_format.upper()
        return (
            f"LoRA Mode — [bold]{self.file_path.name}[/bold] ([cyan]{fmt_label}[/cyan])"
        )

    def _meta_line(self) -> str:
        info = self.lora_info
        if info is None:
            return f"[dim]Format:[/dim] {self.lora_format}  •  [dim]Not analyzable until converted to PEFT[/dim]"
        parts = [f"rank {info.rank}"]
        if info.alpha is not None:
            parts.append(f"α {info.alpha:g}")
            if info.effective_scale is not None:
                parts.append(f"scale {info.effective_scale:.2f}")
        parts.append(f"{len(info.pairs)} pairs")
        parts.append(f"{info.num_layers} layers")
        parts.append(f"modules: {', '.join(info.target_modules)}")
        parts.append(f"{format_number(info.total_params)} params")
        return "  •  ".join(parts)

    def _populate_initial(self) -> None:
        table = self.query_one("#lora-table", DataTable)
        table.clear()
        for pair in self._sorted_pairs():
            table.add_row(
                self._short_module_name(pair),
                str(pair.rank),
                "…",
                "…",
                "…",
                "…",
                key=pair.module_key,
            )
        self._sync_module_column_width(table)

    def _sync_module_column_width(self, table: DataTable) -> None:
        if self.lora_info is None:
            return
        max_len = max(
            (len(self._short_module_name(p)) for p in self.lora_info.pairs),
            default=16,
        )
        col = table.columns.get("module")
        if col is not None:
            col.content_width = max(col.content_width, max_len)
        table.refresh(layout=True)

    def _short_module_name(self, pair: LoRAPair) -> str:
        from sft.ops.lora.detect import format_lora_module_display

        return format_lora_module_display(pair.module_key)

    def _sorted_pairs(self) -> list[LoRAPair]:
        if self.lora_info is None:
            return []
        mode = LORA_SORT_ORDER[self._sort_idx]
        pairs = list(self.lora_info.pairs)

        def stat(name: str, key: str, *, ascending: bool) -> float:
            missing = self._SORT_MISSING_ASC if ascending else self._SORT_MISSING_DESC
            return self._stats.get(name, {}).get(key, missing)

        if mode == LoraSortMode.MODULE_ASC:
            pairs.sort(key=lambda p: natural_sort_key(p.module_key))
        elif mode == LoraSortMode.MODULE_DESC:
            pairs.sort(key=lambda p: natural_sort_key(p.module_key), reverse=True)
        elif mode == LoraSortMode.RANK_DESC:
            pairs.sort(key=lambda p: (-p.rank, natural_sort_key(p.module_key)))
        elif mode == LoraSortMode.RANK_ASC:
            pairs.sort(key=lambda p: (p.rank, natural_sort_key(p.module_key)))
        elif mode == LoraSortMode.EFF_RANK_DESC:
            pairs.sort(
                key=lambda p: (
                    -stat(p.module_key, "eff_rank", ascending=False),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.EFF_RANK_ASC:
            pairs.sort(
                key=lambda p: (
                    stat(p.module_key, "eff_rank", ascending=True),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.SV95_DESC:
            pairs.sort(
                key=lambda p: (
                    -stat(p.module_key, "sv95", ascending=False),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.SV95_ASC:
            pairs.sort(
                key=lambda p: (
                    stat(p.module_key, "sv95", ascending=True),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.NORM_A_DESC:
            pairs.sort(
                key=lambda p: (
                    -stat(p.module_key, "norm_a", ascending=False),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.NORM_A_ASC:
            pairs.sort(
                key=lambda p: (
                    stat(p.module_key, "norm_a", ascending=True),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.NORM_B_DESC:
            pairs.sort(
                key=lambda p: (
                    -stat(p.module_key, "norm_b", ascending=False),
                    natural_sort_key(p.module_key),
                )
            )
        elif mode == LoraSortMode.NORM_B_ASC:
            pairs.sort(
                key=lambda p: (
                    stat(p.module_key, "norm_b", ascending=True),
                    natural_sort_key(p.module_key),
                )
            )
        return pairs

    def _update_sort_indicator(self) -> None:
        if self.lora_format != "peft":
            return
        mode = LORA_SORT_ORDER[self._sort_idx]
        table = self.query_one("#lora-table", DataTable)
        sorted_col, arrow = self._LORA_SORT_COLUMN_MAP[mode]
        for key, base_label in self._LORA_BASE_LABELS.items():
            label_text = base_label + (arrow if key == sorted_col else "")
            col = table.columns.get(key)
            if col is not None:
                col.label = Text.from_markup(label_text)
                # auto_width=True columns render from content_width, not width.
                # Expand content_width so the label (including arrow) always fits.
                col.content_width = max(col.content_width, len(label_text))
        table.refresh(layout=True)

    def _refresh_table(self) -> None:
        if self.lora_format != "peft" or self.lora_info is None:
            return
        table = self.query_one("#lora-table", DataTable)
        cursor_key = None
        if table.cursor_row is not None and table.row_count > 0:
            try:
                cursor_key = table.coordinate_to_cell_key(
                    table.cursor_coordinate
                ).row_key.value
            except Exception:
                cursor_key = None

        table.clear()
        new_cursor = 0
        for i, pair in enumerate(self._sorted_pairs()):
            stats = self._stats.get(pair.module_key, {})
            table.add_row(
                self._short_module_name(pair),
                str(pair.rank),
                self._fmt_eff_rank(stats.get("eff_rank")),
                self._fmt_int(stats.get("sv95")),
                self._fmt_float(stats.get("norm_a")),
                self._fmt_float(stats.get("norm_b")),
                key=pair.module_key,
            )
            if pair.module_key == cursor_key:
                new_cursor = i
        if table.row_count > 0:
            table.move_cursor(row=new_cursor)
        self._sync_module_column_width(table)
        self._update_sort_indicator()

    @staticmethod
    def _fmt_float(v: float | None) -> str:
        return "…" if v is None else f"{v:.3f}"

    @staticmethod
    def _fmt_int(v: float | None) -> str:
        return "…" if v is None else f"{int(v)}"

    @staticmethod
    def _fmt_eff_rank(v: float | None) -> str:
        return "…" if v is None else f"{v:.1f}"

    @work(thread=True, exclusive=True)
    def _compute_stats(self) -> None:
        import numpy as np

        from sft.ops.lora.svd import _qr_svd
        from sft.utils.tensor_io import read_tensors

        if self.lora_info is None:
            return

        try:
            tensors = read_tensors(self.file_path)
        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Failed to load tensors: {e}", severity="error"
            )
            return

        self._tensor_cache = tensors
        total = len(self.lora_info.pairs)

        for i, pair in enumerate(self.lora_info.pairs):
            try:
                a = tensors[pair.lora_a_name]
                b = tensors[pair.lora_b_name]

                (s,) = _qr_svd(a, b, compute_uv=False)
                s_sq = s**2
                total_energy = float(s_sq.sum())
                if total_energy > 0:
                    eff_rank = total_energy / float(s_sq.max())
                    cumvar = np.cumsum(s_sq) / total_energy
                    sv95 = int(np.searchsorted(cumvar, 0.95)) + 1
                else:
                    eff_rank = 0.0
                    sv95 = 0

                stats = {
                    "norm_a": float(np.linalg.norm(a)),
                    "norm_b": float(np.linalg.norm(b)),
                    "eff_rank": eff_rank,
                    "sv95": sv95,
                    "sigma_max": float(s.max()) if s.size else 0.0,
                    "singular_values": s.tolist(),
                }
            except Exception:  # noqa: BLE001 — keep partial results
                stats = {}

            self._stats[pair.module_key] = stats
            if (i + 1) % self._STATS_REFRESH_EVERY == 0 or (i + 1) == total:
                self.app.call_from_thread(self._refresh_table)

    # --- Actions ---

    def action_exit_mode(self) -> None:
        self.app.pop_screen()

    def _set_sort_mode(self, mode: LoraSortMode) -> None:
        if self.lora_format != "peft":
            return
        self._sort_idx = LORA_SORT_ORDER.index(mode)
        self._refresh_table()

    def action_cycle_sort(self) -> None:
        if self.lora_format != "peft":
            return
        self._sort_idx = (self._sort_idx + 1) % len(LORA_SORT_ORDER)
        self._refresh_table()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Click a column header to toggle asc/desc for that metric."""
        if self.lora_format != "peft" or event.control.id != "lora-table":
            return
        col_key = event.column_key.value
        modes = self._LORA_COLUMN_SORT.get(col_key)
        if modes is None:
            return
        asc, desc = modes
        current = LORA_SORT_ORDER[self._sort_idx]
        if current == asc:
            self._set_sort_mode(desc)
        elif current == desc:
            self._set_sort_mode(asc)
        else:
            self._set_sort_mode(desc if col_key != "module" else asc)

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        self.action_show_spectrum()

    def _selected_pair(self) -> LoRAPair | None:
        if self.lora_info is None:
            return None
        try:
            table = self.query_one("#lora-table", DataTable)
        except Exception:
            return None
        if table.cursor_row is None or table.row_count == 0:
            return None
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            return None
        for p in self.lora_info.pairs:
            if p.module_key == key:
                return p
        return None

    def action_show_spectrum(self) -> None:
        pair = self._selected_pair()
        if pair is None:
            return
        stats = self._stats.get(pair.module_key, {})
        sv = stats.get("singular_values")
        if not sv:
            self.notify(
                "Spectrum not yet computed — please wait",
                severity="information",
            )
            return
        self.app.push_screen(SvdSpectrumScreen(pair, sv))

    def action_compress(self) -> None:
        """Open a resize prompt to reduce the whole file's LoRA rank."""
        if self.lora_info is None:
            self.notify(
                "Compress requires PEFT layout — convert from Kohya first",
                severity="warning",
            )
            return
        if self.lora_info.rank <= 1:
            self.notify("Already at rank 1 — cannot reduce further", severity="warning")
            return
        self.app.push_screen(
            LoraResizePromptScreen(self.lora_info.rank), self._on_resize_rank
        )

    def _on_resize_rank(self, choice: int | tuple[str, int] | None) -> None:
        """Handle the resize prompt result. `choice` is an int rank, an
        ("auto", margin) tuple, or None to cancel."""
        if choice is None:
            return
        from sft.ops.lora.resize import resize_lora
        from sft.utils.output import resolve_output

        if isinstance(choice, tuple) and choice[0] == "auto":
            margin = choice[1]
            target_rank, auto_margin = None, margin
            suffix = "rauto" if margin == 1 else f"rauto+{margin}"
        else:
            target_rank = int(choice)  # type: ignore[arg-type]
            auto_margin = None
            suffix = f"r{target_rank}"

        output = resolve_output(None, self.file_path, suffix)
        try:
            result = resize_lora(
                self.file_path,
                output,
                target_rank=target_rank,
                auto_margin=auto_margin,
            )
        except ValueError as e:
            self.notify(f"Compress failed: {e}", severity="error")
            return

        max_err = max(result.errors.values()) if result.errors else 0.0
        min_energy = min(result.energies.values()) if result.energies else 1.0
        if auto_margin is not None:
            ranks = result.per_module_ranks.values()
            rank_desc = f"auto ({min(ranks)}\u2013{max(ranks)})"
        else:
            rank_desc = str(result.new_rank)
        self.notify(
            f"Saved {output.name} "
            f"(rank {result.original_rank}\u2192{rank_desc}, "
            f"max err {max_err:.4f}, min energy {min_energy:.3f})",
            title="LoRA Compressed",
        )

    def action_convert_format(self) -> None:
        """Open Kohya<->PEFT conversion confirmation dialog."""
        self.app.push_screen(
            KohyaConvertScreen(self.file_path), self._on_convert_result
        )

    def _on_convert_result(self, target: str | None) -> None:
        if target is None:
            return
        from sft.ops.lora.convert import convert_lora
        from sft.utils.output import resolve_output

        output = resolve_output(None, self.file_path, target)
        try:
            result = convert_lora(self.file_path, output, target=target)
        except ValueError as e:
            self.notify(f"Conversion failed: {e}", severity="error")
            return

        self.notify(
            f"Saved {output.name} "
            f"({result.source_format}\u2192{result.target_format}, "
            f"{result.modules_converted} modules)",
            title="LoRA Converted",
        )

    def action_show_info(self) -> None:
        """Open a detailed LoRA info modal (full `sft lora info` style)."""
        if self.lora_info is None:
            self.notify(
                "Info requires PEFT layout — convert from Kohya first",
                severity="warning",
            )
            return
        self.app.push_screen(LoraInfoScreen(self.file_path, self.lora_info))


class LoraInfoScreen(ModalScreen):
    """Detailed LoRA breakdown modal — equivalent to `sft lora info` output."""

    CSS = """
    LoraInfoScreen {
        align: center middle;
    }

    #lora-info-container {
        width: 95%;
        min-width: 50;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #lora-info-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #lora-info-body {
        height: auto;
        max-height: 70vh;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("i", "dismiss", "Close"),
    ]

    def __init__(self, file_path: Path, lora_info: LoRAInfo) -> None:
        super().__init__()
        self.file_path = file_path
        self.lora_info = lora_info

    def compose(self) -> ComposeResult:
        info = self.lora_info
        with Container(id="lora-info-container"):
            yield Label("LoRA Info", id="lora-info-title")
            with VerticalScroll(id="lora-info-body"):
                yield Static(f"[dim]File:[/dim]   {self.file_path.name}")
                yield Static(f"[dim]Rank:[/dim]   {info.rank}")
                if info.alpha is not None:
                    extra = (
                        f"  ([dim]scale[/dim] {info.effective_scale:.2f})"
                        if info.effective_scale is not None
                        else ""
                    )
                    yield Static(f"[dim]Alpha:[/dim]  {info.alpha:g}{extra}")
                yield Static(f"[dim]Modules:[/dim] {', '.join(info.target_modules)}")
                yield Static(f"[dim]Layers:[/dim] {info.num_layers}")
                yield Static(
                    f"[dim]Params:[/dim] {info.total_params:,} "
                    f"({format_number(info.total_params)})"
                )
                yield Static("\n[bold]Pairs[/bold]")
                for pair in info.pairs:
                    a_shape = format_shape(pair.lora_a_shape)
                    b_shape = format_shape(pair.lora_b_shape)
                    short = self._short_pair_name(pair)
                    yield Static(
                        f"  [cyan]{short}[/cyan]   "
                        f"A {a_shape}   B {b_shape}   {format_dtype(pair.dtype)}"
                    )
            yield Static("\n[dim]Press ESC or i to close[/dim]")

    @staticmethod
    def _short_pair_name(pair: LoRAPair) -> str:
        from sft.ops.lora.detect import format_lora_module_display

        return format_lora_module_display(pair.module_key)


class LoraResizePromptScreen(ModalScreen):
    """Modal to enter a target rank for compress (resize).

    Accepts either a positive integer < current rank, or "auto" / "auto+N"
    to truncate each pair to ``ceil(stable_rank) + N`` (N defaults to 1).
    Dismisses with either an `int` (fixed rank) or a tuple `("auto", N)`.
    """

    CSS = """
    LoraResizePromptScreen {
        align: center middle;
    }

    #resize-container {
        width: 78;
        height: auto;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }

    #resize-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    .resize-section {
        margin-bottom: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, current_rank: int) -> None:
        super().__init__()
        self.current_rank = current_rank

    def compose(self) -> ComposeResult:
        with Container(id="resize-container"):
            yield Label("Compress LoRA", id="resize-title")
            yield Static(
                f"[dim]Current rank:[/dim] {self.current_rank}",
                classes="resize-section",
            )

            yield Static(
                "[bold]Options:[/bold]",
                classes="resize-section",
            )
            yield Static(
                "  [cyan]integer[/cyan] (e.g. [bold]8[/bold])\n"
                f"      Truncate every pair to the same rank "
                f"(must be in [1, {self.current_rank - 1}]).",
                classes="resize-section",
            )
            yield Static(
                "  [cyan]auto[/cyan]\n"
                "      Shrink each pair individually to the smallest rank that "
                "captures essentially all of its information. Output ranks "
                "vary per pair: aggressive on pairs with simple updates, "
                "conservative on pairs with rich ones.",
                classes="resize-section",
            )
            yield Static(
                "  [cyan]auto+N[/cyan] (e.g. [bold]auto+2[/bold])\n"
                "      Same as auto, but keep N extra singular values per pair "
                "as a safety margin.",
                classes="resize-section",
            )

            yield Input(placeholder="enter target rank...", id="rank-input")
            yield Static(
                "\n[dim]Enter: confirm   ESC: cancel[/dim]",
                classes="resize-section",
            )

    def on_mount(self) -> None:
        self.query_one("#rank-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        from sft.commands.lora import _parse_rank_spec

        try:
            target_rank, auto_margin = _parse_rank_spec(event.value)
        except ValueError as e:
            self.notify(str(e), severity="warning")
            return
        if target_rank is not None and target_rank >= self.current_rank:
            self.notify(
                f"Rank must be between 1 and {self.current_rank - 1} "
                f"(or use 'auto'/'auto+N')",
                severity="warning",
            )
            return
        # Dismiss with int for fixed mode, tuple ("auto", N) for auto mode
        if auto_margin is not None:
            self.dismiss(("auto", auto_margin))
        else:
            self.dismiss(target_rank)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SvdSpectrumScreen(ModalScreen):
    """Drill-down view of the singular value spectrum of a single LoRA pair.

    Displays singular values as a horizontal bar chart, with effective rank
    and variance-threshold cutoffs (90/95/99%) called out.
    """

    CSS = """
    SvdSpectrumScreen {
        align: center middle;
    }

    #svd-container {
        width: 95%;
        min-width: 50;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #svd-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #svd-meta {
        color: $text-muted;
        margin-bottom: 1;
        height: auto;
    }

    #svd-stats {
        color: $text-muted;
        height: auto;
        margin-top: 1;
    }

    #svd-chart-wrap {
        height: auto;
        max-height: 60vh;
    }

    #svd-chart {
        height: auto;
    }

    #svd-footer {
        dock: bottom;
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("enter", "dismiss", "Close"),
    ]

    def __init__(self, pair: LoRAPair, singular_values: list[float]) -> None:
        super().__init__()
        self.pair = pair
        self.sv = [float(v) for v in singular_values]

    def compose(self) -> ComposeResult:
        from sft.ops.lora.detect import format_lora_module_display

        module_label = format_lora_module_display(self.pair.module_key)
        with Container(id="svd-container"):
            yield Label(f"SVD Spectrum — {module_label}", id="svd-title")
            yield Static(
                f"[dim]Module:[/dim] {self.pair.module_key}\n"
                f"[dim]Rank:[/dim] {self.pair.rank}   "
                f"[dim]A:[/dim] {format_shape(self.pair.lora_a_shape)}   "
                f"[dim]B:[/dim] {format_shape(self.pair.lora_b_shape)}",
                id="svd-meta",
            )
            with VerticalScroll(id="svd-chart-wrap"):
                yield Static(self._build_chart(40), id="svd-chart")
            yield Static(self._build_stats(), id="svd-stats")
            yield Static("[dim]ESC / Enter: close[/dim]", id="svd-footer")

    def on_mount(self) -> None:
        self._rebuild_chart()

    def on_resize(self) -> None:
        self._rebuild_chart()

    def _rebuild_chart(self) -> None:
        """Size the bar chart to the available chart-wrap width."""
        try:
            wrap = self.query_one("#svd-chart-wrap")
            chart = self.query_one("#svd-chart", Static)
        except Exception:
            return
        # Reserve room for label ("σ NNN  ") + padding + value column.
        # Value formats like "0.1234" are typically ~6 chars; allow 10 for safety.
        available = wrap.size.width - 18
        bar_width = max(10, min(120, available))
        chart.update(self._build_chart(bar_width))

    def _build_chart(self, bar_width: int) -> str:
        """Horizontal bar chart of singular values, normalized to sigma_max."""
        if not self.sv:
            return "[dim]No singular values[/dim]"
        smax = max(self.sv) or 1.0
        lines: list[str] = []
        for i, s in enumerate(self.sv):
            ratio = s / smax
            filled = int(round(ratio * bar_width))
            bar = "\u2588" * filled + "\u00b7" * (bar_width - filled)
            lines.append(f"[dim]\u03c3[/dim]{i + 1:>3}  [cyan]{bar}[/cyan]  {s:.4g}")
        return "\n".join(lines)

    def _build_stats(self) -> str:
        """Variance-threshold cutoffs and effective rank line."""
        import numpy as np

        if not self.sv:
            return ""
        sv = np.array(self.sv)
        sq = sv**2
        total = float(sq.sum())
        if total == 0:
            return "[dim]All singular values are zero[/dim]"
        cumvar = np.cumsum(sq) / total
        sv90 = int(np.searchsorted(cumvar, 0.90)) + 1
        sv95 = int(np.searchsorted(cumvar, 0.95)) + 1
        sv99 = int(np.searchsorted(cumvar, 0.99)) + 1
        eff_rank = total / float(sq.max())
        return (
            f"[dim]Effective rank:[/dim] {eff_rank:.2f}   "
            f"[dim]SV90:[/dim] {sv90}   "
            f"[dim]SV95:[/dim] {sv95}   "
            f"[dim]SV99:[/dim] {sv99}   "
            f"[dim]σ_max:[/dim] {float(sv.max()):.4g}"
        )


class SftApp(App):
    """Interactive browser for .safetensors files."""

    TITLE = "sft"
    COMMANDS = {SftCommands} if _HAS_COMMAND_PALETTE else set()

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr 2fr;
    }

    /* Reset the main-browser grid layout for modals — otherwise every
       popup ends up as the first cell of a 2-column grid and gets
       crushed into ~1/3 of the viewport width. */
    ModalScreen {
        layout: vertical;
        grid-size: 1 1;
        grid-columns: 1fr;
    }

    /* Textual's command palette defaults to a full-width Vertical, which
       looks bloated. Constrain it to a centered, modestly sized box. */
    CommandPalette > Vertical {
        width: 60%;
        max-width: 90;
        min-width: 40;
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
        Binding("m", "show_metadata", "Metadata", show=True),
        Binding("enter", "show_stats", "Stats", show=True),
        Binding("c", "cast_file", "Cast", show=True),
        Binding("L", "show_lora_mode", "LoRA Mode", show=True),
        Binding("D", "diff_file", "Diff", show=True),
        Binding("colon", "command_palette", "Commands", show=True),
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
        self.lora_info: LoRAInfo | None = None
        self.lora_format: str | None = None  # 'peft', 'kohya', 'mixed', or None

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Footer()

        # Parse the file
        try:
            self.index = TensorIndex.from_file(self.file_path)
            self.prefix_tree = PrefixTree(self.index)
            self._all_tensors = self.index.tensors.copy()
            self._base_tensors = self.index.tensors.copy()
        except Exception as e:
            yield Static(f"Error loading file: {e}", id="error")
            return

        try:
            from sft.ops.lora.convert import detect_format
            from sft.ops.lora.detect import detect_lora

            fmt_info = detect_format(self.file_path)
            self.lora_format = fmt_info.format
            self.lora_info = detect_lora(self.file_path)
        except Exception:
            self.lora_format = None
            self.lora_info = None

        yield HierarchyTree(self.prefix_tree)
        yield TensorTable()
        yield SearchInput()

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

        # If this is a LoRA file, surface a one-shot hint about LoRA Mode
        if self.lora_format in ("peft", "kohya"):
            self.notify(
                f"{self.lora_format.upper()} LoRA detected — press L to enter LoRA Mode",
                title="LoRA",
                timeout=8,
            )

    def on_hierarchy_tree_node_selected(
        self, event: HierarchyTree.NodeSelected
    ) -> None:
        """Handle tree node selection."""
        self._current_prefix = event.prefix

        # Get tensors under this prefix from the active tree
        tree = self.query_one(HierarchyTree)
        tensors = tree.active_tree.get_tensors_under(event.prefix)
        self._base_tensors = tensors.copy()
        self._all_tensors = tensors.copy()

        # Update tensor table
        table = self.query_one(TensorTable)
        table.update_tensors(tensors, event.prefix)

        # Apply current sort
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

    def action_show_metadata(self) -> None:
        """Show file metadata popup."""
        if self.index:
            self.push_screen(MetadataScreen(self.index.metadata, self.file_path))

    def action_show_stats(self) -> None:
        """Show computed statistics for the selected tensor."""
        table = self.query_one(TensorTable)
        tensor = table.get_selected_tensor()
        if tensor:
            self.push_screen(TensorStatsScreen(tensor, self.file_path))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a TensorTable row opens the Stats popup.

        DataTable consumes `enter` into a RowSelected message, so the
        App-level `enter` binding never fires while the table is focused.
        We translate the message into the same action here. Restricted to
        TensorTable so sub-screens (e.g. LoRA Mode) keep their own
        enter-on-row behaviour.
        """
        if isinstance(event.data_table, TensorTable):
            self.action_show_stats()

    def action_cast_file(self) -> None:
        """Open cast dialog to convert file to a different dtype."""
        if self.index is None:
            return
        self.push_screen(CastScreen(self.file_path), self._on_cast_result)

    def action_show_lora_mode(self) -> None:
        """Enter the dedicated LoRA Mode screen (PEFT or Kohya)."""
        if self.index is None:
            return
        if self.lora_format is None:
            self.notify(
                "Not a LoRA file — no Kohya or PEFT modules detected",
                severity="information",
            )
            return
        if self.lora_format == "mixed":
            self.notify(
                "File contains both Kohya and PEFT modules — normalize first",
                severity="warning",
            )
            return
        self.push_screen(
            LoraModeScreen(self.file_path, self.index, self.lora_format, self.lora_info)
        )

    def action_diff_file(self) -> None:
        """Open file picker to choose a second file, then run the diff."""
        if self.index is None:
            return
        self.push_screen(
            DiffFilePickerScreen(self.file_path), self._on_diff_target_picked
        )

    def _on_diff_target_picked(self, other: Path | None) -> None:
        if other is None:
            return
        self.push_screen(DiffResultScreen(self.file_path, other))

    def _on_cast_result(self, dtype: str | None) -> None:
        """Handle cast screen result."""
        if dtype is None:
            return
        from sft.ops.cast import cast_file
        from sft.utils.output import resolve_output

        output = resolve_output(None, self.file_path, dtype)
        try:
            result = cast_file(self.file_path, output, dtype)
            self.notify(
                f"Saved to {output.name} ({result.cast_count} tensors cast)",
                title="Cast Complete",
            )
        except Exception as e:
            self.notify(f"Cast failed: {e}", severity="error")

    def _do_cast_fp16(self) -> None:
        self._do_cast("fp16")

    def _do_cast_fp32(self) -> None:
        self._do_cast("fp32")

    def _do_cast_bf16(self) -> None:
        self._do_cast("bf16")

    def _do_cast(self, dtype: str) -> None:
        from sft.ops.cast import cast_file
        from sft.utils.output import resolve_output

        output = resolve_output(None, self.file_path, dtype)
        try:
            result = cast_file(self.file_path, output, dtype)
            self.notify(
                f"Saved to {output.name} ({result.cast_count} tensors cast)",
                title="Cast Complete",
            )
        except Exception as e:
            self.notify(f"Cast failed: {e}", severity="error")

    def _do_check_file(self) -> None:
        from sft.ops.check import check_file

        result = check_file(self.file_path)
        if result.healthy:
            self.notify("File is healthy", title="Check")
        else:
            issues: list[str] = []
            if result.nan_tensors:
                issues.append(f"NaN in: {', '.join(result.nan_tensors)}")
            if result.inf_tensors:
                issues.append(f"Inf in: {', '.join(result.inf_tensors)}")
            if result.header_error:
                issues.append(f"Header: {result.header_error}")
            if result.offsets_error:
                issues.append(f"Offsets: {result.offsets_error}")
            self.notify(
                "\n".join(issues) or "Unknown issue",
                title="Issues Found",
                severity="warning",
            )

    def _do_show_metadata(self) -> None:
        self.action_show_metadata()

    def _do_show_stats(self) -> None:
        self.action_show_stats()

    def _do_show_info(self) -> None:
        from sft.ops.info import summarize
        from sft.utils.formatting import format_bytes, format_number

        try:
            summary = summarize(self.file_path)
            dtypes = ", ".join(
                f"{format_dtype(d.dtype)}({d.count})" for d in summary.dtypes
            )
            msg = (
                f"Tensors: {summary.total_tensors}\n"
                f"Params: {format_number(summary.total_parameters)}\n"
                f"Size: {format_bytes(summary.total_tensor_bytes)}\n"
                f"Dtypes: {dtypes}"
            )
            self.notify(msg, title=summary.file_name)
        except Exception as e:
            self.notify(f"Info failed: {e}", severity="error")

    def _do_show_details(self) -> None:
        self.action_show_details()

    def _do_show_lora_mode(self) -> None:
        self.action_show_lora_mode()

    def _do_diff_file(self) -> None:
        self.action_diff_file()

    def _do_start_search(self) -> None:
        self.action_start_search()

    def _do_cycle_sort(self) -> None:
        self.action_cycle_sort()

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
