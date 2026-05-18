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

try:
    from textual.command import Hit, Hits, Provider

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
from sft.utils.formatting import format_bytes, format_dtype, format_shape


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

    def __init__(self, tensor: TensorInfo) -> None:
        super().__init__()
        self.tensor = tensor

    def compose(self) -> ComposeResult:
        t = self.tensor
        with Container(id="detail-container"):
            yield Label("Tensor Details", id="detail-title")
            yield Static(f"[dim]Name:[/dim]  {t.full_name}", classes="detail-row")
            yield Static(
                f"[dim]Shape:[/dim] {format_shape(t.shape)}", classes="detail-row"
            )
            yield Static(f"[dim]Rank:[/dim]  {t.rank}", classes="detail-row")
            yield Static(
                f"[dim]Dtype:[/dim] {format_dtype(t.dtype)}", classes="detail-row"
            )
            yield Static(
                f"[dim]Size:[/dim]  {format_bytes(t.nbytes)} ({t.nbytes:,} bytes)",
                classes="detail-row",
            )
            yield Static(f"[dim]Numel:[/dim] {t.numel:,}", classes="detail-row")
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

    #metadata-scroll {
        max-height: 60vh;
    }

    #metadata-content {
        height: auto;
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
                with VerticalScroll(id="metadata-scroll"):
                    yield Static(f"\n{formatted}", id="metadata-content")
            else:
                yield Static("\n[dim]No metadata found in file[/dim]")

            yield Static("\n[dim]Press ESC or M to close[/dim]")


class TensorStatsScreen(ModalScreen):
    """Modal screen showing computed tensor statistics."""

    CSS = """
    TensorStatsScreen {
        align: center middle;
    }

    #stats-container {
        width: 65;
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
        Binding("S", "dismiss", "Close"),
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
                ("Cast to fp16", "Cast all tensors to float16", "cast_fp16"),
                ("Cast to fp32", "Cast all tensors to float32", "cast_fp32"),
                ("Cast to bf16", "Cast all tensors to bfloat16", "cast_bf16"),
                ("Check file", "Validate file integrity", "check_file"),
                ("Show metadata", "View file metadata", "show_metadata"),
                ("Show stats", "Compute statistics for selected tensor", "show_stats"),
                ("File info", "Show file summary", "show_info"),
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

    #lora-header {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 2;
        column-span: 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "toggle_panel", "Switch Panel", show=True),
        Binding("slash", "start_search", "Search", show=True),
        Binding("escape", "cancel_search", "Cancel", show=False),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("space", "show_details", "Details", show=True),
        Binding("m", "show_metadata", "Metadata", show=True),
        Binding("S", "show_stats", "Stats", show=True),
        Binding("c", "cast_file", "Cast", show=True),
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
        self.lora_info = None

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
            from sft.ops.lora.detect import detect_lora

            self.lora_info = detect_lora(self.file_path)
        except Exception:
            self.lora_info = None

        if self.lora_info:
            yield Static(self._lora_header_text(), id="lora-header")

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

    def action_show_details(self) -> None:
        """Show tensor details popup."""
        table = self.query_one(TensorTable)
        tensor = table.get_selected_tensor()

        if tensor:
            self.push_screen(TensorDetailScreen(tensor))

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

    def action_cast_file(self) -> None:
        """Open cast dialog to convert file to a different dtype."""
        if self.index is None:
            return
        self.push_screen(CastScreen(self.file_path), self._on_cast_result)

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

    def _lora_header_text(self) -> str:
        """Build the LoRA header bar text."""
        from sft.utils.formatting import format_number

        info = self.lora_info
        parts = ["LoRA Adapter", f"Rank: {info.rank}"]
        if info.alpha is not None:
            parts.append(f"Alpha: {info.alpha:.0f}")
            if info.effective_scale is not None:
                parts.append(f"Scale: {info.effective_scale:.1f}")
        parts.append(f"Modules: {', '.join(info.target_modules)}")
        parts.append(f"Params: {format_number(info.total_params)}")
        return " │ ".join(parts)

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
