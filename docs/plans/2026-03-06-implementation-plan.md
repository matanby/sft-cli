# sft Toolkit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `sft` from a TUI browser into a full CLI toolkit for safetensors files — inspection, transforms, diffing, and LoRA operations.

**Architecture:** Three-layer architecture: `ops/` (pure logic), `commands/` (CLI wrappers), `browser.py` (TUI). Typer subcommand groups with a callback-based default that preserves `sft <file>` behavior. Short aliases for frequent commands. Smart output defaults so `-o` is rarely needed. All operations stream tensor-by-tensor for memory efficiency.

**Tech Stack:** Python 3.9+, Typer (CLI), Textual (TUI), safetensors + numpy (tensor I/O), pytest (testing).

**Design doc:** `docs/plans/2026-03-06-sft-toolkit-design.md`

---

## Testing Strategy

### Principles

1. **TDD for every command.** Write the failing test first, then implement.
2. **Tiny fixtures.** Test safetensors files use 4x4 or 8x8 matrices (not realistic sizes). Tests must be fast.
3. **CLI integration tests.** Use `typer.testing.CliRunner` to invoke commands and assert on stdout, stderr, and exit codes.
4. **Roundtrip property tests.** Operations that transform files should roundtrip correctly (e.g., split then cat = original).
5. **No mocking of safetensors I/O.** Tests create real `.safetensors` files in `tmp_path` fixtures. This is fast with tiny tensors.

### Test Fixtures (conftest.py)

All fixtures produce small files (<1KB) that are fast to create and verify:

- **`mini_model`** — A tiny transformer-like model (2 layers, 4x4 weight matrices, fp32 + fp16 mix, metadata).
- **`lora_adapter`** — A LoRA adapter file with `lora_A`/`lora_B` matrix pairs for 2 modules, rank 4.
- **`lora_base_model`** — A base model whose shapes are compatible with `lora_adapter`.
- **`finetuned_model`** — Same structure as `lora_base_model` with different values (for LoRA extract tests).
- **`model_with_nans`** — A model file containing NaN and Inf values (for `check`/`stat` tests).
- **`sharded_model`** — Two shard files + an index.json.

### Running Tests

```bash
uv run pytest tests/ -v                    # all tests
uv run pytest tests/test_info.py -v        # single command
uv run pytest tests/ -k "lora" -v          # all LoRA tests
```

---

## Phase 1: Foundation + Scriptable Inspection

### Task 1: Set Up Testing Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `pyproject.toml` (add dev deps + required deps)

**Step 1: Add test dependencies to pyproject.toml**

Add `pytest` to dev deps. Add `safetensors` and `numpy` as required dependencies:

```toml
dependencies = [
    "textual>=0.40",
    "typer>=0.9",
    "safetensors>=0.4",
    "numpy>=1.24",
]

[dependency-groups]
dev = [
    "pre-commit>=4.3.0",
    "ruff>=0.14.10",
    "pytest>=8.0",
]
```

**Step 2: Create tests/__init__.py**

Empty file.

**Step 3: Create tests/conftest.py with fixtures**

```python
"""Shared test fixtures for sft tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def mini_model(tmp_path: Path) -> Path:
    """A tiny transformer-like model with 2 layers, mixed dtypes, and metadata."""
    tensors = {
        "model.embed_tokens.weight": np.random.randn(16, 8).astype(np.float16),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float16),
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(8, 8).astype(np.float16),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(16, 8).astype(np.float16),
        "model.layers.0.mlp.down_proj.weight": np.random.randn(8, 16).astype(np.float16),
        "model.layers.0.input_layernorm.weight": np.random.randn(8).astype(np.float16),
        "model.layers.1.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float16),
        "model.layers.1.self_attn.k_proj.weight": np.random.randn(8, 8).astype(np.float16),
        "model.layers.1.mlp.gate_proj.weight": np.random.randn(16, 8).astype(np.float16),
        "model.layers.1.mlp.down_proj.weight": np.random.randn(8, 16).astype(np.float16),
        "model.layers.1.input_layernorm.weight": np.random.randn(8).astype(np.float16),
        "model.norm.weight": np.random.randn(8).astype(np.float16),
        "lm_head.weight": np.random.randn(16, 8).astype(np.float16),
        "model.layers.0.rotary_emb.inv_freq": np.random.randn(4).astype(np.float32),
    }
    metadata = {"format": "pt", "model_type": "llama"}
    path = tmp_path / "mini_model.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    return path


@pytest.fixture
def lora_base_model(tmp_path: Path) -> Path:
    """A base model compatible with lora_adapter fixture."""
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((8, 8), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.ones((8, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.ones((16, 8), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        "model.embed_tokens.weight": np.ones((16, 8), dtype=np.float32),
    }
    path = tmp_path / "base_model.safetensors"
    save_file(tensors, str(path))
    return path


@pytest.fixture
def finetuned_model(tmp_path: Path) -> Path:
    """A finetuned model — same structure as lora_base_model but different values."""
    rng = np.random.RandomState(42)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((8, 8), dtype=np.float32) + 0.1 * rng.randn(8, 8).astype(np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.ones((8, 8), dtype=np.float32) + 0.1 * rng.randn(8, 8).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.ones((16, 8), dtype=np.float32) + 0.1 * rng.randn(16, 8).astype(np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        "model.embed_tokens.weight": np.ones((16, 8), dtype=np.float32),
    }
    path = tmp_path / "finetuned_model.safetensors"
    save_file(tensors, str(path))
    return path


@pytest.fixture
def lora_adapter(tmp_path: Path) -> Path:
    """A LoRA adapter file with rank-4 A/B pairs for 2 modules."""
    rank = 4
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(rank, 8).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(8, rank).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": np.random.randn(rank, 8).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": np.random.randn(8, rank).astype(np.float32),
    }
    metadata = {
        "rank": "4",
        "alpha": "8",
        "target_modules": "q_proj,v_proj",
    }
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    return path


@pytest.fixture
def model_with_nans(tmp_path: Path) -> Path:
    """A model file containing NaN and Inf values."""
    w1 = np.random.randn(4, 4).astype(np.float32)
    w1[0, 0] = np.nan
    w2 = np.random.randn(4, 4).astype(np.float32)
    w2[1, 1] = np.inf
    w3 = np.random.randn(4, 4).astype(np.float32)
    tensors = {
        "has_nan.weight": w1,
        "has_inf.weight": w2,
        "clean.weight": w3,
    }
    path = tmp_path / "model_with_nans.safetensors"
    save_file(tensors, str(path))
    return path
```

**Step 4: Verify fixtures work**

```bash
uv run pytest tests/conftest.py --collect-only
```

**Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "feat: add testing infrastructure with safetensors fixtures"
```

---

### Task 2: Refactor CLI for Subcommands

This is the most critical foundational change. We move from a single-command Typer app to a subcommand-based app while preserving `sft <file>` backward compatibility. We also set up the ops/ and commands/ directory structure with short aliases.

**Files:**
- Modify: `src/sft/cli.py`
- Create: `src/sft/ops/__init__.py`
- Create: `src/sft/commands/__init__.py`
- Create: `tests/test_cli.py`

**Step 1: Write tests for the CLI refactor**

Create `tests/test_cli.py`:

```python
"""Tests for CLI entry point and subcommand routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "sft" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output or "usage" in result.output.lower()


def test_browse_subcommand_validates_extension(tmp_path: Path):
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("not a safetensors file")
    result = runner.invoke(app, ["browse", str(bad_file)])
    assert result.exit_code == 1
    assert "safetensors" in result.output.lower()


def test_browse_subcommand_rejects_missing_file():
    result = runner.invoke(app, ["browse", "/nonexistent/model.safetensors"])
    assert result.exit_code != 0
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: failures because `browse` subcommand doesn't exist yet.

**Step 3: Refactor cli.py to support subcommands**

Rewrite `src/sft/cli.py`:

```python
"""CLI entry point for sft."""

from __future__ import annotations

from pathlib import Path

import typer

from sft import __version__

app = typer.Typer(
    name="sft",
    help="The Swiss army knife for .safetensors files.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"sft {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """The Swiss army knife for .safetensors files."""
    pass


def validate_safetensors(path: Path) -> Path:
    """Validate that a path points to a readable .safetensors file."""
    if not path.exists():
        typer.secho(f"Error: File not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if path.suffix.lower() != ".safetensors":
        typer.secho(
            f"Error: Expected a .safetensors file, got '{path.suffix}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return path


def default_output(input_path: Path, suffix: str) -> Path:
    """Generate a smart default output path: {stem}.{suffix}.safetensors"""
    return input_path.parent / f"{input_path.stem}.{suffix}.safetensors"


@app.command()
@app.command("b", hidden=True)  # short alias
def browse(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to browse.",
        resolve_path=True,
    ),
) -> None:
    """Open an interactive TUI browser for a .safetensors file."""
    file = validate_safetensors(file)
    from sft.browser import SftApp

    app_instance = SftApp(file)
    app_instance.run()
```

Note: The exact mechanism for Typer aliases may need adjustment during implementation (e.g., registering the same function under two names, or using a `rich_help_panel` for grouping). The intent is that `sft b model.safetensors` works identically to `sft browse model.safetensors`.

Also create empty `src/sft/ops/__init__.py` and `src/sft/commands/__init__.py` to set up the package structure.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

**Step 5: Commit**

```bash
git add src/sft/cli.py src/sft/ops/ src/sft/commands/ tests/test_cli.py
git commit -m "refactor: restructure CLI for subcommands, add browse command with alias"
```

---

### Task 3: Utility Modules

Shared utilities used by multiple commands. Build and test these before any commands that depend on them.

**Files:**
- Create: `src/sft/utils/__init__.py`
- Create: `src/sft/utils/formatting.py`
- Create: `src/sft/utils/glob.py`
- Create: `src/sft/utils/output.py`
- Create: `tests/test_utils.py`

**Step 1: Write tests for formatting utilities**

```python
"""Tests for shared utility modules."""

from __future__ import annotations

from sft.utils.formatting import format_bytes, format_number, format_shape, format_dtype


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
        assert format_dtype("F16") == "fp16"

    def test_bfloat16(self):
        assert format_dtype("BF16") == "bf16"

    def test_float32(self):
        assert format_dtype("F32") == "fp32"

    def test_passthrough(self):
        assert format_dtype("I32") == "I32"
```

**Step 2: Write tests for glob matching**

Add to `tests/test_utils.py`:

```python
from sft.utils.glob import tensor_matches


class TestTensorGlob:
    def test_wildcard_single_segment(self):
        assert tensor_matches("model.layers.0.weight", "model.layers.*.weight")

    def test_wildcard_no_match(self):
        assert not tensor_matches("model.layers.0.bias", "model.layers.*.weight")

    def test_double_star(self):
        assert tensor_matches("model.layers.0.self_attn.q_proj.weight", "**.weight")

    def test_exact_match(self):
        assert tensor_matches("lm_head.weight", "lm_head.weight")

    def test_prefix_star(self):
        assert tensor_matches("model.layers.0.self_attn.q_proj.weight", "model.layers.0.*")

    def test_include_exclude(self):
        from sft.utils.glob import filter_tensors
        names = [
            "model.layers.0.weight",
            "model.layers.0.bias",
            "model.layers.1.weight",
            "lm_head.weight",
        ]
        result = filter_tensors(names, include="model.layers.*", exclude="*.bias")
        assert result == ["model.layers.0.weight", "model.layers.1.weight"]
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_utils.py -v
```

**Step 4: Implement formatting.py**

Create `src/sft/utils/formatting.py`:

```python
"""Human-readable formatting utilities."""

from __future__ import annotations

_DTYPE_DISPLAY = {
    "F16": "fp16",
    "F32": "fp32",
    "F64": "fp64",
    "BF16": "bf16",
    "F8_E4M3": "fp8_e4m3",
    "F8_E5M2": "fp8_e5m2",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "BOOL": "bool",
}


def format_bytes(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    else:
        return f"{nbytes / 1024 / 1024 / 1024:.2f} GB"


def format_number(n: int) -> str:
    """Format large numbers as human-readable (e.g. 6.7B)."""
    if n < 1_000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    elif n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    else:
        return f"{n / 1_000_000_000:.1f}B"


def format_shape(shape: tuple[int, ...]) -> str:
    """Format tensor shape as string."""
    if len(shape) == 0:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return f"({', '.join(str(d) for d in shape)})"


def format_dtype(dtype: str) -> str:
    """Map internal safetensors dtype names to human-readable form."""
    return _DTYPE_DISPLAY.get(dtype, dtype)
```

**Step 5: Implement glob.py**

Create `src/sft/utils/glob.py`:

```python
"""Tensor name glob matching."""

from __future__ import annotations

import fnmatch
import re


def tensor_matches(name: str, pattern: str) -> bool:
    """Check if a tensor name matches a glob pattern.

    Uses `.` as the path separator. `*` matches a single segment,
    `**` matches any number of segments.
    """
    regex = _glob_to_regex(pattern)
    return bool(re.fullmatch(regex, name))


def filter_tensors(
    names: list[str],
    include: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    """Filter tensor names by include/exclude glob patterns."""
    result = names
    if include is not None:
        result = [n for n in result if tensor_matches(n, include)]
    if exclude is not None:
        result = [n for n in result if not tensor_matches(n, exclude)]
    return result


def _glob_to_regex(pattern: str) -> str:
    """Convert a dot-separated glob pattern to a regex."""
    parts = pattern.split(".")
    regex_parts: list[str] = []
    for part in parts:
        if part == "**":
            regex_parts.append(r"(?:[^.]+\.)*[^.]+")
        elif "*" in part or "?" in part or "[" in part:
            regex_parts.append(fnmatch.translate(part).rstrip(r"\Z$"))
        else:
            regex_parts.append(re.escape(part))
    return r"\.".join(regex_parts)
```

**Step 6: Implement output.py (smart output defaults)**

Create `src/sft/utils/output.py`:

```python
"""Smart output path generation."""

from __future__ import annotations

from pathlib import Path


def default_output(input_path: Path, suffix: str) -> Path:
    """Generate a default output path: {stem}.{suffix}.safetensors"""
    return input_path.parent / f"{input_path.stem}.{suffix}.safetensors"


def resolve_output(
    explicit: Path | None,
    input_path: Path,
    suffix: str,
) -> Path:
    """Resolve the output path: use explicit -o if given, otherwise smart default."""
    if explicit is not None:
        return explicit
    return default_output(input_path, suffix)
```

**Step 7: Create `src/sft/utils/__init__.py`**

Empty file.

**Step 8: Run tests**

```bash
uv run pytest tests/test_utils.py -v
```

**Step 9: Also update browser.py to use shared formatting**

Replace the local `format_bytes` and `format_shape` in `browser.py` with imports from `sft.utils.formatting`. Search for the two functions and replace them with:

```python
from sft.utils.formatting import format_bytes, format_shape
```

Then delete the local `format_bytes` and `format_shape` function definitions from `browser.py`.

**Step 9: Sanity check — make sure the TUI still works**

```bash
# Only if a test_model.safetensors exists, otherwise skip
uv run sft browse test_model.safetensors
```

**Step 10: Commit**

```bash
git add src/sft/utils/ tests/test_utils.py src/sft/browser.py
git commit -m "feat: add shared formatting and tensor glob utilities"
```

---

### Task 4: `sft info`

**Files:**
- Create: `src/sft/ops/info.py` (pure logic)
- Create: `src/sft/commands/info.py` (CLI wrapper)
- Modify: `src/sft/cli.py` (register command + `i` alias)
- Create: `tests/test_info.py`

**Step 1: Write tests**

Create `tests/test_info.py`:

```python
"""Tests for sft info command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_info_shows_file_summary(mini_model: Path):
    result = runner.invoke(app, ["info", str(mini_model)])
    assert result.exit_code == 0
    assert "mini_model.safetensors" in result.output
    assert "14" in result.output  # tensor count
    assert "fp16" in result.output


def test_info_shows_metadata(mini_model: Path):
    result = runner.invoke(app, ["info", str(mini_model)])
    assert result.exit_code == 0
    assert "llama" in result.output


def test_info_json_output(mini_model: Path):
    result = runner.invoke(app, ["info", str(mini_model), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "file" in data
    assert "tensors" in data
    assert "total_parameters" in data
    assert data["tensors"] == 14


def test_info_rejects_non_safetensors(tmp_path: Path):
    bad = tmp_path / "model.txt"
    bad.write_text("nope")
    result = runner.invoke(app, ["info", str(bad)])
    assert result.exit_code == 1


def test_info_rejects_missing_file():
    result = runner.invoke(app, ["info", "/nonexistent/model.safetensors"])
    assert result.exit_code != 0
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_info.py -v
```

**Step 3: Implement `src/sft/ops/info.py` (pure logic)**

This module computes the file summary and returns a data structure — no printing, no CLI concerns:

```python
"""File summary logic."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sft.index import TensorIndex
from sft.utils.formatting import format_dtype


@dataclass
class FileSummary:
    file_name: str
    file_size: int
    total_tensors: int
    total_parameters: int
    total_tensor_bytes: int
    dtype_counts: dict[str, int]
    dtype_bytes: dict[str, int]
    metadata: dict[str, str]


def summarize(path: Path) -> FileSummary:
    """Compute a summary of a safetensors file."""
    index = TensorIndex.from_file(path)

    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    total_params = 0

    for t in index.tensors:
        d = format_dtype(t.dtype)
        dtype_counts[d] += 1
        dtype_bytes[d] += t.nbytes
        total_params += t.numel

    return FileSummary(
        file_name=path.name,
        file_size=path.stat().st_size,
        total_tensors=index.total_tensors,
        total_parameters=total_params,
        total_tensor_bytes=index.total_bytes,
        dtype_counts=dict(dtype_counts),
        dtype_bytes=dict(dtype_bytes),
        metadata=index.metadata,
    )
```

**Step 4: Implement `src/sft/commands/info.py` (CLI wrapper)**

Thin wrapper that calls ops/ and formats output:

```python
"""sft info — CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import validate_safetensors
from sft.ops.info import summarize
from sft.utils.formatting import format_bytes, format_number


def info(
    file: Path = typer.Argument(..., help="Path to a .safetensors file.", resolve_path=True),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Print a summary of a .safetensors file."""
    file = validate_safetensors(file)
    summary = summarize(file)

    if json_output:
        data = {
            "file": summary.file_name,
            "file_size": summary.file_size,
            "tensors": summary.total_tensors,
            "total_parameters": summary.total_parameters,
            "total_tensor_bytes": summary.total_tensor_bytes,
            "dtypes": summary.dtype_counts,
            "metadata": summary.metadata,
        }
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"File:        {summary.file_name}")
        typer.echo(f"File size:   {format_bytes(summary.file_size)}")
        typer.echo(f"Tensors:     {summary.total_tensors}")
        typer.echo(f"Parameters:  {summary.total_parameters:,} ({format_number(summary.total_parameters)})")
        typer.echo()
        typer.echo("Dtype breakdown:")
        for dtype, count in sorted(summary.dtype_counts.items(), key=lambda x: -summary.dtype_bytes.get(x[0], 0)):
            size = summary.dtype_bytes[dtype]
            pct = 100.0 * size / summary.total_tensor_bytes if summary.total_tensor_bytes else 0
            typer.echo(f"  {dtype:<8}  {count} tensors  {format_bytes(size):>10}  {pct:5.1f}%")
        if summary.metadata:
            typer.echo()
            typer.echo("Metadata:")
            for k, v in sorted(summary.metadata.items()):
                typer.echo(f"  {k}: {v}")
```

**Step 5: Register the info command in cli.py (with `i` alias)**

Register both `info` and its short alias `i` in `src/sft/cli.py`. The exact Typer mechanism for aliases will be determined during implementation (e.g., registering the same function twice, or using a hidden command). The key requirement: `sft info <file>` and `sft i <file>` must both work.

**Step 7: Run tests**

```bash
uv run pytest tests/test_info.py -v
```

**Step 8: Commit**

```bash
git add src/sft/commands/ tests/test_info.py src/sft/cli.py
git commit -m "feat: add sft info command with JSON output"
```

---

### Task 5: `sft tree`

**Files:**
- Create: `src/sft/commands/tree.py`
- Modify: `src/sft/cli.py` (register command)
- Create: `tests/test_tree.py`

**Step 1: Write tests**

Create `tests/test_tree.py`:

```python
"""Tests for sft tree command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_tree_shows_hierarchy(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model)])
    assert result.exit_code == 0
    assert "model" in result.output
    assert "layers" in result.output
    assert "self_attn" in result.output
    assert "q_proj" in result.output


def test_tree_shows_tensor_info(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model)])
    assert result.exit_code == 0
    # Should show shape and dtype for leaf tensors
    assert "fp16" in result.output or "F16" in result.output


def test_tree_depth_limit(mini_model: Path):
    result = runner.invoke(app, ["tree", str(mini_model), "--depth", "1"])
    assert result.exit_code == 0
    # Depth 1 should show top-level groups but not deep leaves
    assert "model" in result.output
    # q_proj is at depth 4 — should not appear
    assert "q_proj" not in result.output
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tree.py -v
```

**Step 3: Implement `src/sft/commands/tree.py`**

The implementation renders the `PrefixTree` from `index.py` as an ASCII tree using `├──`, `└──`, `│` box-drawing characters. Leaf nodes include `[shape] dtype (size)`.

**Step 4: Register in cli.py, run tests, commit**

```bash
uv run pytest tests/test_tree.py -v
git add src/sft/commands/tree.py tests/test_tree.py src/sft/cli.py
git commit -m "feat: add sft tree command"
```

---

### Task 6: `sft ls`

**Files:**
- Create: `src/sft/commands/ls.py`
- Modify: `src/sft/cli.py`
- Create: `tests/test_ls.py`

**Step 1: Write tests**

```python
"""Tests for sft ls command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_ls_multiple_files(tmp_path: Path):
    for name in ["a.safetensors", "b.safetensors"]:
        tensors = {"w": np.zeros((4, 4), dtype=np.float32)}
        save_file(tensors, str(tmp_path / name))
    result = runner.invoke(app, ["ls", str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")])
    assert result.exit_code == 0
    assert "a.safetensors" in result.output
    assert "b.safetensors" in result.output


def test_ls_json(tmp_path: Path):
    tensors = {"w": np.zeros((4, 4), dtype=np.float32)}
    path = tmp_path / "m.safetensors"
    save_file(tensors, str(path))
    result = runner.invoke(app, ["ls", str(path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["file"] == "m.safetensors"
```

**Step 2: Implement, test, commit**

```bash
uv run pytest tests/test_ls.py -v
git add src/sft/commands/ls.py tests/test_ls.py src/sft/cli.py
git commit -m "feat: add sft ls command for multi-file summary"
```

---

### Task 7: `sft check`

**Files:**
- Create: `src/sft/commands/check.py`
- Modify: `src/sft/cli.py`
- Create: `tests/test_check.py`

**Step 1: Write tests**

```python
"""Tests for sft check command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def test_check_healthy_file(mini_model: Path):
    result = runner.invoke(app, ["check", str(mini_model)])
    assert result.exit_code == 0
    assert "healthy" in result.output.lower() or "✓" in result.output


def test_check_detects_nans(model_with_nans: Path):
    result = runner.invoke(app, ["check", str(model_with_nans)])
    assert result.exit_code == 1
    assert "nan" in result.output.lower() or "NaN" in result.output


def test_check_skip_values(model_with_nans: Path):
    result = runner.invoke(app, ["check", str(model_with_nans), "--skip-values"])
    # Structural check passes; only NaN scan would fail
    assert result.exit_code == 0


def test_check_corrupted_file(tmp_path: Path):
    bad = tmp_path / "corrupt.safetensors"
    bad.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00garbage")
    result = runner.invoke(app, ["check", str(bad)])
    assert result.exit_code == 1
```

**Step 2: Implement, test, commit**

The implementation opens the file, validates the header, checks offsets against file size, then optionally scans tensor data for NaN/Inf using numpy.

```bash
uv run pytest tests/test_check.py -v
git add src/sft/commands/check.py tests/test_check.py src/sft/cli.py
git commit -m "feat: add sft check command for file health diagnostics"
```

---

## Phase 2: Core Transforms

All transform commands share common patterns:
- Read tensors from input with `safetensors.numpy.load_file` or streaming read
- Apply transformation
- Write to output with `safetensors.numpy.save_file`
- Support `--dry-run`, `--include`/`--exclude`

### Task 8: Tensor I/O Utilities

**Files:**
- Create: `src/sft/utils/tensor_io.py`
- Create: `tests/test_tensor_io.py`

Build streaming read/write helpers that load one tensor at a time. This is the foundation for memory-efficient transforms on large files.

Key functions:
- `iter_tensors(path) -> Iterator[(name, numpy.ndarray)]` — yields tensors one at a time
- `read_tensor(path, name) -> numpy.ndarray` — read a single tensor by name
- `write_tensors(path, tensors: Iterator, metadata: dict)` — streaming write (uses `safetensors.numpy.save_file` internally, which requires all tensors at once, so this buffers — but the interface prepares us for future streaming optimization)
- `copy_with_transform(src, dst, transform_fn, include, exclude, metadata)` — the core pattern: read src, apply transform per tensor, write dst

Tests verify roundtrip: create file → read back → values match.

---

### Task 9: `sft cast`

**Files:**
- Create: `src/sft/commands/cast.py`
- Create: `src/sft/utils/dtypes.py`
- Create: `tests/test_cast.py`

**Tests:**
- Cast fp32 → fp16: output file has fp16 tensors.
- Cast with `--include`: only matching tensors change dtype, others stay.
- Cast with `--exclude`: excluded tensors keep original dtype.
- Roundtrip: cast fp32 → fp16 → fp32 produces approximately equal values.
- `--dry-run`: no output file written, but report printed.
- Error: output path required.

**`src/sft/utils/dtypes.py`** maps CLI dtype names ("bf16", "fp16", "fp32") to numpy dtypes and safetensors dtype strings.

---

### Task 10: `sft diff`

**Files:**
- Create: `src/sft/commands/diff.py`
- Create: `tests/test_diff.py`

**Tests:**
- Two identical files: "no differences" message.
- Files with added/removed tensors: reports additions/removals.
- Files with shape changes: reports shape mismatch.
- Files with dtype changes: reports dtype mismatch.
- `--delta` flag: reports L2 norm and cosine similarity per tensor.
- `--json` output parses correctly.

---

### Task 11: `sft slice` and `sft strip`

**Files:**
- Create: `src/sft/commands/slice.py`
- Create: `src/sft/commands/strip.py`
- Create: `tests/test_slice.py`
- Create: `tests/test_strip.py`

**Tests:**
- Slice with `--include "model.layers.0.*"`: output contains only layer 0 tensors.
- Strip with `--exclude "*.bias"`: output contains everything except biases.
- `--dry-run`: reports what would be included/removed.
- Slice + cat roundtrip: slice into parts, cat back = original.

---

### Task 12: `sft cat`

**Files:**
- Create: `src/sft/commands/cat.py`
- Create: `tests/test_cat.py`

**Tests:**
- Cat two non-overlapping files: output contains all tensors from both.
- Duplicate tensor names: error by default.
- `--allow-duplicates`: last file wins.
- `--dry-run`: shows merged tensor list.

---

### Task 13: `sft split`

**Files:**
- Create: `src/sft/commands/split.py`
- Create: `tests/test_split.py`

**Tests:**
- Split a file with `--max-size`: produces multiple shards.
- Generates a valid `model.safetensors.index.json`.
- Roundtrip: split then cat = original.
- Edge case: file smaller than max-size produces single shard.

---

### Task 14: `sft rename`

**Files:**
- Create: `src/sft/commands/rename.py`
- Create: `tests/test_rename.py`

**Tests:**
- Simple prefix substitution: `--sub "model" "transformer"` renames all matching keys.
- Regex substitution: `--sub "layers\.(\d+)" "blocks.\1"`.
- Multiple `--sub` pairs applied in order.
- `--dry-run`: shows old → new mappings.
- No matches: file is copied unchanged, with a warning.

---

### Task 15: `sft metadata`

**Files:**
- Create: `src/sft/commands/metadata.py`
- Create: `tests/test_metadata.py`

**Tests:**
- View metadata: prints key-value pairs.
- `--set key=value`: output file has new metadata, tensor data unchanged.
- `--unset key`: output file has key removed.
- `--json`: metadata output as JSON.
- No metadata: prints empty/message.

---

## Phase 3: LoRA Toolkit

### Task 16: LoRA Detection Utilities

**Files:**
- Create: `src/sft/lora/__init__.py`
- Create: `src/sft/lora/detect.py`
- Create: `tests/test_lora_detect.py`

Build utilities to detect LoRA structure from tensor names:
- Detect PEFT-style naming: `base_model.model.{module}.lora_A.weight` / `lora_B.weight`
- Parse rank from tensor shapes.
- Parse alpha from metadata.
- Group A/B pairs by target module.

This module is shared by all `sft lora` commands.

---

### Task 17: `sft lora info`

**Files:**
- Create: `src/sft/lora/info.py`
- Create: `src/sft/lora/cli.py` (Typer sub-app for `sft lora`)
- Modify: `src/sft/cli.py` (register lora sub-app)
- Create: `tests/test_lora_info.py`

**Tests:**
- Detects rank, target modules, parameter count.
- Shows alpha from metadata.
- `--json` output is parseable.
- Non-LoRA file: clear error message.

---

### Task 18: `sft lora merge`

**Files:**
- Create: `src/sft/lora/merge.py`
- Create: `tests/test_lora_merge.py`

**Tests:**
- Merge a known LoRA into base: `W_out = W_base + scale * (B @ A)`. Verify numerically with known small matrices.
- `--scale 0.5`: verify the scaling factor is applied.
- Non-target tensors pass through unchanged.
- Shape mismatch between base and LoRA: clear error.

---

### Task 19: `sft lora extract`

**Files:**
- Create: `src/sft/lora/extract.py`
- Create: `tests/test_lora_extract.py`

**Tests:**
- Extract LoRA from two models with known delta: verify the reconstruction `B @ A ≈ Δ` within tolerance.
- Reconstruction error is printed per module.
- Output has correct PEFT tensor naming.
- `--rank` controls the output rank.

---

### Task 20: `sft lora resize`

**Files:**
- Create: `src/sft/lora/resize.py`
- Create: `tests/test_lora_resize.py`

**Tests:**
- Resize rank 8 → rank 4: output tensors have the smaller dimension.
- Reconstruction error is reported.
- Error if target rank >= current rank.
- Roundtrip: resize should preserve the dominant singular values.

---

### Task 21: `sft lora add`

**Files:**
- Create: `src/sft/lora/add.py`
- Create: `tests/test_lora_add.py`

**Tests:**
- Add two LoRAs with weights [0.5, 0.5]: verify combined delta ≈ 0.5*Δ1 + 0.5*Δ2.
- Different weight ratios.
- Incompatible LoRAs (different modules or shapes): clear error.
- Three or more LoRAs.

---

### Task 22: `sft lora compat`

**Files:**
- Create: `src/sft/lora/compat.py`
- Create: `tests/test_lora_compat.py`

**Tests:**
- Compatible base + LoRA: exit code 0.
- Shape mismatch: exit code 1, reports which tensors mismatch.
- Missing modules in base: exit code 1, reports which are missing.

---

### Task 23: `sft lora svd`

**Files:**
- Create: `src/sft/lora/svd_cmd.py`
- Create: `tests/test_lora_svd.py`

**Tests:**
- Reports singular values per module.
- Suggested rank captures specified variance threshold.
- `--threshold 0.99` changes the suggestion.
- `--json` output is parseable.

---

## Phase 4: Convert + Stats

### Task 24: `sft stat`

**Files:**
- Create: `src/sft/commands/stat.py`
- Create: `tests/test_stat.py`

**Tests:**
- Reports mean, std, min, max for each tensor.
- Reports NaN/Inf counts.
- `--include` filters tensors.
- `--check` flag: exit code 1 when NaN/Inf present.
- Sparsity percentage is correct (known tensor with 25% zeros → reports 25%).

---

### Task 25: `sft convert`

**Files:**
- Create: `src/sft/commands/convert.py`
- Create: `tests/test_convert.py`

**Tests (require torch):**
- Convert a PyTorch `.bin` file → safetensors: output is valid safetensors.
- Convert a `.pt` file → safetensors.
- `--dtype` flag casts during conversion.
- Missing torch: clear error message pointing to `pip install sft-cli[torch]`.

Mark all convert tests with `@pytest.mark.skipif` if torch is not installed.

---

## Execution Strategy: Subagent Parallelization

Tasks within each phase have a dependency structure that allows significant
parallelization using subagents. The executor should dispatch independent tasks
concurrently (up to 4 at a time) while respecting the dependency ordering.

### Dependency Graph

```
Phase 1: Foundation
  Task 1 (test infra)
    → Task 2 (CLI refactor)
      → Task 3 (utils)
        → Task 4 (info)  ┐
        → Task 5 (tree)  ├── all 4 in parallel
        → Task 6 (ls)    │
        → Task 7 (check) ┘

Phase 2: Core Transforms
  Task 8 (tensor I/O)
    → Task 9 (cast)  ────────────┐
    → Task 10 (diff)             ├── parallel batch A (4 agents)
    → Task 11 (slice + strip)    │
    → Task 12 (cat)  ────────────┘
    → Task 13 (split)  ──────┐
    → Task 14 (rename)       ├── parallel batch B (3 agents)
    → Task 15 (metadata)  ───┘

Phase 3: LoRA Toolkit
  Task 16 (LoRA detect utils)
    → Task 17 (lora info + CLI sub-app)
      → Task 18 (lora merge)  ──┐
      → Task 19 (lora extract)  ├── parallel batch A (4 agents)
      → Task 20 (lora resize)   │
      → Task 21 (lora add)  ────┘
      → Task 22 (lora compat) ──┐
      → Task 23 (lora svd)  ────┘ parallel batch B (2 agents)

Phase 4: Convert + Stats
  Task 24 (stat) ──┐
  Task 25 (convert) ┘ parallel (2 agents)

Phase 5: TUI Operations
  Task 26 (command palette + contextual keybindings)
    → Task 27 (multi-file tabs)
    → Task 28 (selection mode + operations)
    → Task 29 (LoRA-aware mode)
    → Task 30 (diff view)
```

### Execution Rules

1. **Sequential gates.** Tasks 1→2→3, 8, 16→17 must complete before their
   dependents. Run these with a single agent and verify tests pass before
   dispatching parallel work.
2. **Parallel batches.** Once a gate task passes, dispatch all independent
   tasks as concurrent subagents (max 4). Each subagent gets the full context:
   which files to create, the test cases, the shared utilities API.
3. **Integration after each batch.** After parallel subagents complete, run the
   full test suite (`uv run pytest tests/ -v`) and fix any integration issues
   (e.g., import conflicts, CLI registration order) before proceeding.
4. **Phase gate.** After all tasks in a phase pass, run the verification
   checklist below before starting the next phase.

### What Each Subagent Needs

Each subagent prompt should include:
- The specific task description from this plan (files, tests, implementation notes)
- The current state of `src/sft/cli.py` (so it can register its command correctly)
- The API of shared utilities (`utils/formatting.py`, `utils/glob.py`, `utils/tensor_io.py`, `utils/output.py`)
- The `conftest.py` fixtures available to it
- Reminder to follow the ops/ pattern: logic in `src/sft/ops/`, CLI wrapper in `src/sft/commands/`
- Reminder to use smart output defaults via `utils/output.py` for write commands
- Instruction to run `uv run pytest tests/test_<command>.py -v` and `uv run ruff check` before finishing

---

## Verification Checklist

After each phase, verify:

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Linting passes: `uv run ruff check src/ tests/`
- [ ] Formatting passes: `uv run ruff format --check src/ tests/`
- [ ] CLI help is coherent: `uv run sft --help`, `uv run sft info --help`, etc.
- [ ] Existing TUI still works: `uv run sft browse <file>`
- [ ] `--version` still works: `uv run sft --version`
- [ ] All write commands have smart output defaults (never require `-o`, never mutate input)
- [ ] All read commands support `--json` where specified
- [ ] All write commands support `--dry-run` where specified
- [ ] Short aliases work: `sft i`, `sft b`, `sft t`
- [ ] Help output is grouped by category (Inspect, Transform, LoRA, Convert)
