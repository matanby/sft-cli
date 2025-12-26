"""Textual TUI application for browsing safetensors files."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
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
            yield Static(f"[dim]Dtype:[/dim] {t.dtype}", classes="detail-row")
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
            self.push_screen(TensorDetailScreen(tensor))

    def action_show_metadata(self) -> None:
        """Show file metadata popup."""
        if self.index:
            self.push_screen(MetadataScreen(self.index.metadata, self.file_path))

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
