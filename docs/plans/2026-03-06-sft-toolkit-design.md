# sft — The Swiss Army Knife for Safetensors Files

**Date**: 2026-03-06
**Status**: Draft

---

## 1. Vision

`sft` is a fast, intuitive CLI toolkit for working with `.safetensors` model files. It's the tool ML engineers reach for whenever they need to inspect, understand, transform, or operate on model checkpoints — replacing dozens of throwaway Python scripts with a single, well-designed command-line tool.

Today, `sft` does one thing beautifully: browse safetensors files in an interactive TUI. The vision is to extend it into a **complete checkpoint toolkit** while keeping the same ethos — fast, zero-config, does what you mean.

## 2. Target Audience

**Primary**: ML engineers and researchers who work with LoRA fine-tuning and model checkpoints daily.

**Their typical day**:

- **Training**: Kick off a LoRA training run, check intermediate checkpoints as they land. "Did training diverge? Are there NaNs?"
- **Evaluation**: Training is done — inspect the adapter, merge it into the base model, evaluate. "What rank is this? What modules were trained?"
- **Optimization**: The LoRA is too large — compactify it. The merged model needs bf16 for serving — cast it. It needs sharding for multi-GPU — split it.
- **Experimentation**: Combine multiple task-specific LoRAs into one, experiment with weights, compare results.
- **Debugging**: "Why is this model outputting garbage?" — check for NaN/Inf. "Is this the right checkpoint?" — quick inspect. "What changed between step 1000 and 5000?" — diff.

Every one of these tasks currently requires a throwaway Python script. `sft` should make them one-liners.

## 3. Design Principles

1. **Never mutate inputs.** All write operations produce a new file. Smart output defaults mean users rarely need to type `-o` explicitly, but the input is never modified. `--in-place` is available for those who want it.
2. **Memory-efficient streaming.** Process tensors one-by-one where possible. A 70B model should be castable on a machine with 16GB of RAM.
3. **Progressive disclosure of complexity.** `sft model.safetensors` just works. Advanced operations are subcommands you discover when you need them.
4. **Ergonomics for power users.** Short aliases for frequent commands (`i`, `b`, `t`). Smart output defaults so you rarely spell out `-o`. Grouped, scannable `--help`. Minimal keystrokes for common operations, explicit flags for rare ones. No brain fatigue.
5. **Two interfaces, one tool.** Every operation is available both as a CLI subcommand (for scripting/automation) and from within the TUI (for interactive exploration). The underlying logic lives in a shared `ops/` layer.
6. **Unix philosophy.** Pipe-friendly output (`--json`), meaningful exit codes, composable with other tools.
7. **Minimal required dependencies.** Core operations use `safetensors` + `numpy`. Heavy deps like `torch` are optional extras.
8. **Progress and feedback.** Any operation that reads tensor data shows a progress bar with ETA. `--dry-run` is available on every write operation.
9. **Consistent CLI conventions.** `--dtype` for type, `--include`/`--exclude` for tensor glob patterns — these work the same everywhere.

## 4. CLI Architecture

The CLI transitions from a single-command tool to a subcommand-based tool using Typer's command groups.

```
sft [browse] <file>              # TUI browser (default when given a bare file)
sft <subcommand> [args] [flags]  # Operations
```

Running `sft <file>` with no subcommand launches the TUI browser, preserving full backward compatibility.

### 4.1. Default Behavior

When the user runs `sft model.safetensors` (no subcommand), the tool detects that the argument is a `.safetensors` file and launches the TUI browser. This is the existing behavior and remains the default.

### 4.2. Subcommand Groups

Commands are organized into intuitive groups with short aliases for the most frequent operations:

```
$ sft --help

The Swiss army knife for .safetensors files.

Inspect:
  browse (b)    Interactive TUI browser [default]
  info   (i)    Print file summary
  ls            List multiple files
  tree   (t)    Print tensor hierarchy
  stat          Compute tensor statistics
  check         Validate file integrity

Transform:
  cast          Convert tensor dtypes
  diff          Compare two files
  slice         Extract tensors by pattern
  strip         Remove tensors by pattern
  cat           Merge multiple files
  split         Shard a large file
  rename        Rename tensor keys
  metadata      View/edit file metadata

LoRA:
  lora info     LoRA adapter analysis
  lora merge    Apply LoRA to base model
  lora extract  Create LoRA from weight delta
  lora resize   Reduce LoRA rank (compactify)
  lora add      Combine LoRAs (task arithmetic)
  lora compat   Check base/LoRA compatibility
  lora svd      Singular value analysis

Convert:
  convert       Convert PyTorch → safetensors
```

### 4.3. Short Aliases

The most frequently used commands get single-letter aliases to minimize keystrokes:

| Alias | Command |
|-------|---------|
| `b`   | `browse` |
| `i`   | `info`   |
| `t`   | `tree`   |

These are the daily-driver inspection commands. Transform and LoRA commands are less frequent and don't need aliases — their names are already short and memorable.

### 4.4. Smart Output Defaults

Write commands auto-generate an output path when `-o` is omitted, using a predictable `{stem}.{suffix}.safetensors` pattern:

| Command | Default output |
|---------|---------------|
| `sft cast model.safetensors --dtype bf16` | `model.bf16.safetensors` |
| `sft lora merge base.safetensors adapter.safetensors` | `base.merged.safetensors` |
| `sft lora resize adapter.safetensors --rank 8` | `adapter.r8.safetensors` |
| `sft lora extract base.safetensors ft.safetensors --rank 16` | `base.lora-r16.safetensors` |
| `sft rename model.safetensors --sub ...` | `model.renamed.safetensors` |
| `sft slice model.safetensors --include ...` | `model.sliced.safetensors` |
| `sft strip model.safetensors --exclude ...` | `model.stripped.safetensors` |
| `sft split model.safetensors --max-size 4GB` | `model-00001.safetensors`, `model-00002.safetensors`, ... |
| `sft cat a.safetensors b.safetensors` | `merged.safetensors` |
| `sft convert model.bin` | `model.safetensors` |

The generated output path is always printed so there is no ambiguity. Users can override with `-o` when they need a specific name.

## 5. Feature Specification

### 5.1. INSPECT — "What's in this file?"

#### `sft browse <file>` (alias: `sft <file>`)

The existing interactive TUI browser. Becomes an explicit subcommand but remains the default behavior when `sft` is invoked with a bare `.safetensors` file.

No changes to current behavior.

#### `sft info <file>`

Non-interactive, pipe-friendly summary printed to stdout.

```
$ sft info model.safetensors

File:        model.safetensors
File size:   13.5 GB
Tensors:     291
Parameters:  6,738,415,616 (6.7B)

Dtype breakdown:
  bf16    291 tensors    13.5 GB    100.0%

Top-level groups:
  model.layers     (256 tensors, 12.8 GB)
  model.embed      (2 tensors, 512 MB)
  model.norm       (1 tensor, 8 KB)
  lm_head          (1 tensor, 256 MB)

Metadata:
  format: pt
```

**Flags**:
- `--json` — Machine-readable JSON output for scripting and CI.
- `--no-metadata` — Omit the metadata section.

#### `sft stat <file>`

Compute tensor-level statistics. Unlike `info`, this reads actual tensor data.

```
$ sft stat model.safetensors

Tensor                              dtype   shape            mean      std       min       max    sparsity   nan   inf
model.embed_tokens.weight           bf16    [32000, 4096]    0.0002    0.0124   -0.1875    0.1836    0.1%     0     0
model.layers.0.self_attn.q_proj     bf16    [4096, 4096]    -0.0001    0.0088   -0.0547    0.0508    0.0%     0     0
...
```

**Flags**:
- `--include <glob>` / `--exclude <glob>` — Filter which tensors to compute stats for.
- `--json` — Machine-readable output.
- `--check` — Exit with code 1 if any NaN or Inf values are found. Useful in CI pipelines.

#### `sft check <file>`

Quick health diagnostic. Validates file integrity without computing full statistics.

```
$ sft check model.safetensors
✓ Header: valid JSON, 291 tensors
✓ Offsets: all data offsets within file bounds
✓ Dtypes: all tensors bf16
✓ Values: no NaN or Inf detected
✓ File healthy

$ echo $?
0
```

Checks performed:
- Header is valid JSON and parseable.
- All data offsets are within file bounds and non-overlapping.
- No NaN or Inf values in any tensor.
- File size matches expected size from header + tensor data.

**Flags**:
- `--skip-values` — Only check structural integrity, skip the NaN/Inf scan (much faster for huge files).

#### `sft ls <files...>`

Tabular summary of multiple files, like `ls -lh` for safetensors.

```
$ sft ls checkpoints/*.safetensors

File                        Tensors    Params     Size     Dtypes
step-1000.safetensors       291        6.7B       13.5 GB  bf16
step-2000.safetensors       291        6.7B       13.5 GB  bf16
step-3000.safetensors       291        6.7B       13.5 GB  bf16
adapter.safetensors         48         21M        42 MB    bf16
```

**Flags**:
- `--json` — Machine-readable output.
- `--sort <field>` — Sort by name, size, params, or tensors.

#### `sft tree <file>`

Print the tensor namespace hierarchy as an ASCII tree (non-interactive version of the TUI's tree panel).

```
$ sft tree model.safetensors

model (291 tensors, 13.5 GB)
├── embed_tokens
│   └── weight [32000, 4096] bf16 (256 MB)
├── layers
│   ├── 0 (8 tensors, 402 MB)
│   │   ├── self_attn
│   │   │   ├── q_proj.weight [4096, 4096] bf16 (32 MB)
│   │   │   ├── k_proj.weight [4096, 1024] bf16 (8 MB)
│   │   │   ...
│   │   └── mlp
│   │       ├── gate_proj.weight [14336, 4096] bf16 (112 MB)
│   │       ...
│   ├── 1 ...
│   ...
├── norm
│   └── weight [4096] bf16 (8 KB)
└── lm_head
    └── weight [32000, 4096] bf16 (256 MB)
```

**Flags**:
- `--depth <n>` — Limit tree depth.
- `--include <glob>` / `--exclude <glob>` — Filter which tensors to show.

### 5.2. COMPARE — "How are these different?"

#### `sft diff <file_a> <file_b>`

Compare two safetensors files. Shows structural differences by default, with optional value-level comparison.

**Structural diff (default, header-only, instant)**:

```
$ sft diff base.safetensors finetuned.safetensors

Compared: base.safetensors (291 tensors) vs finetuned.safetensors (293 tensors)

Added (2):
  + classifier.weight    [4096, 10]    bf16    80 KB
  + classifier.bias      [10]          bf16    20 B

Shape changed (0):
  (none)

Dtype changed (0):
  (none)

Unchanged: 291 tensors
```

**Value diff (reads tensor data)**:

```
$ sft diff base.safetensors finetuned.safetensors --delta

Tensor                              L2 norm(Δ)    cosine sim    % changed
model.layers.0.self_attn.q_proj     0.0342        0.9998        100.0%
model.layers.0.self_attn.k_proj     0.0287        0.9999        100.0%
model.layers.31.mlp.gate_proj       0.1205        0.9987        100.0%
model.embed_tokens.weight           0.0001        1.0000          0.3%
...

Summary: 289/291 tensors changed, avg cosine similarity 0.9994
```

**Flags**:
- `--delta` — Compute value-level differences (L2 norm of delta, cosine similarity).
- `--include <glob>` / `--exclude <glob>` — Filter tensors.
- `--json` — Machine-readable output.
- `--threshold <float>` — Only show tensors with cosine similarity below threshold (to find the most-changed layers).

### 5.3. TRANSFORM — "Change this checkpoint"

All transform commands produce a new file. They never modify the input.

#### `sft cast <file> --dtype <dtype>`

Convert tensor dtypes.

```bash
sft cast model.safetensors --dtype bf16
# → writes model.bf16.safetensors

sft cast model.safetensors --dtype fp16 --include "*.weight" --exclude "*.embed*" -o custom.safetensors
```

**Supported dtypes**: `fp32` (`float32`), `fp16` (`float16`), `bf16` (`bfloat16`), `fp8_e4m3`, `fp8_e5m2`.

**Flags**:
- `--dtype <dtype>` — Target dtype (required).
- `--include <glob>` / `--exclude <glob>` — Only cast matching tensors; leave others unchanged.
- `-o <path>` — Output file (default: `{stem}.{dtype}.safetensors`).
- `--dry-run` — Show what would be converted without writing.

#### `sft slice <file>`

Extract a subset of tensors by pattern.

```bash
sft slice model.safetensors --include "model.layers.0-3.*"
# → writes model.sliced.safetensors

sft slice model.safetensors --include "*.attention.*" -o attention-only.safetensors
```

**Flags**:
- `--include <glob>` / `--exclude <glob>` — Tensor name patterns (at least one required).
- `-o <path>` — Output file (default: `{stem}.sliced.safetensors`).
- `--dry-run` — Show which tensors would be included.

#### `sft cat <files...>`

Merge multiple safetensors files into one.

```bash
sft cat shard-00001.safetensors shard-00002.safetensors
# → writes merged.safetensors

sft cat shard-*.safetensors -o full-model.safetensors
```

**Flags**:
- `-o <path>` — Output file (default: `merged.safetensors`).
- `--allow-duplicates` — If tensor names collide, keep the version from the last file listed. Default behavior is to error on duplicates.
- `--dry-run` — Show what the merged file would contain.

#### `sft split <file> --max-size <size>`

Shard a large file by size threshold.

```bash
sft split model.safetensors --max-size 4GB
# → writes model-00001.safetensors, model-00002.safetensors, ...
# → writes model.safetensors.index.json
```

Generates shards plus a HuggingFace-compatible `model.safetensors.index.json` mapping tensor names to shard files.

**Flags**:
- `--max-size <size>` — Maximum shard size, e.g. `4GB`, `2GB`, `500MB` (required).
- `-o <pattern>` — Output pattern with `{index}` placeholder (default: `{stem}-{index}.safetensors`).
- `--dry-run` — Show how tensors would be distributed across shards.

#### `sft rename <file>`

Rename tensor keys using regex substitution.

```bash
sft rename model.safetensors --sub "transformer\." "model."
# → writes model.renamed.safetensors
```

Supports multiple `--sub` pairs applied in order.

**Flags**:
- `--sub <pattern> <replacement>` — Regex substitution pair. Can be repeated.
- `-o <path>` — Output file (default: `{stem}.renamed.safetensors`).
- `--dry-run` — Show old → new name mappings without writing.

#### `sft strip <file>`

Remove tensors matching a pattern. The inverse of `slice --include`.

```bash
sft strip checkpoint.safetensors --exclude "optimizer.*"
# → writes checkpoint.stripped.safetensors

sft strip model.safetensors --exclude "*.bias" -o no-bias.safetensors
```

**Flags**:
- `--exclude <glob>` — Pattern of tensors to remove (required).
- `-o <path>` — Output file (default: `{stem}.stripped.safetensors`).
- `--dry-run` — Show which tensors would be removed.

#### `sft metadata <file>`

View or edit safetensors file metadata without touching tensor data.

```bash
# View metadata
sft metadata model.safetensors

# Set a key
sft metadata model.safetensors --set format=pt --set base_model=llama-3-8b -o model-tagged.safetensors

# Remove a key
sft metadata model.safetensors --unset some_key -o cleaned.safetensors
```

**Flags**:
- `--set <key>=<value>` — Set a metadata key-value pair. Can be repeated.
- `--unset <key>` — Remove a metadata key. Can be repeated.
- `-o <path>` — Output file (required when `--set` or `--unset` is used).
- `--json` — Output metadata as JSON.

### 5.4. LORA — "Work with LoRA adapters"

All LoRA operations live under the `sft lora` subcommand group.

#### `sft lora info <file>`

LoRA-specific analysis of an adapter file.

```
$ sft lora info adapter.safetensors

LoRA Adapter: adapter.safetensors
Format:       PEFT

Rank:         16
Alpha:        32 (effective scale: 2.0)
Target modules:
  q_proj      weight_A [16, 4096]   weight_B [4096, 16]   bf16
  k_proj      weight_A [16, 1024]   weight_B [1024, 16]   bf16
  v_proj      weight_A [16, 1024]   weight_B [1024, 16]   bf16
  o_proj      weight_A [16, 4096]   weight_B [4096, 16]   bf16
  gate_proj   weight_A [16, 4096]   weight_B [14336, 16]  bf16
  up_proj     weight_A [16, 4096]   weight_B [14336, 16]  bf16
  down_proj   weight_A [16, 14336]  weight_B [4096, 16]   bf16

Layers:       32
Parameters:   21,233,664 (21.2M)
Size:         42.5 MB
Compression:  0.32% of full fine-tune
```

**Flags**:
- `--json` — Machine-readable output.

#### `sft lora merge <base> <lora>`

Apply a LoRA adapter to a base model, producing a merged full-rank model.

```bash
sft lora merge base.safetensors adapter.safetensors
# → writes base.merged.safetensors

sft lora merge base.safetensors adapter.safetensors --scale 0.8 -o custom.safetensors
```

Processes tensors one-by-one in a streaming fashion to avoid loading the full model into memory. For each target module: `W_merged = W_base + scale * (B @ A)`.

**Flags**:
- `--scale <float>` — Scale the LoRA contribution (default: uses alpha/rank from metadata, or 1.0).
- `-o <path>` — Output file (default: `{base_stem}.merged.safetensors`).
- `--dry-run` — Show which tensors would be modified.

#### `sft lora extract <base> <finetuned> --rank <r>`

Create a LoRA adapter by computing the low-rank SVD decomposition of the weight delta between a base model and a fine-tuned model.

```bash
sft lora extract base.safetensors finetuned.safetensors --rank 16
# → writes base.lora-r16.safetensors
```

For each weight matrix: `Δ = W_finetuned - W_base`, then `Δ ≈ B @ A` via truncated SVD at the specified rank.

Reports per-layer reconstruction error so the user can judge whether the chosen rank captures enough of the fine-tuning signal.

**Flags**:
- `--rank <int>` — Target rank for the LoRA decomposition (required).
- `--include <glob>` / `--exclude <glob>` — Which weight matrices to decompose. Defaults to linear projections (attention + MLP).
- `-o <path>` — Output file (default: `{base_stem}.lora-r{rank}.safetensors`).
- `--alpha <float>` — Alpha value to store in metadata (default: same as rank).

#### `sft lora resize <lora> --rank <r>`

Reduce LoRA rank via truncated SVD (compactify). Takes a rank-R LoRA and produces a rank-r LoRA where r < R.

```bash
sft lora resize adapter.safetensors --rank 8
# → writes adapter.r8.safetensors

Resizing adapter.safetensors: rank 16 → 8

Module                    Reconstruction error
q_proj (layers 0-31)      0.23%
k_proj (layers 0-31)      0.18%
v_proj (layers 0-31)      0.15%
...

Overall reconstruction:   99.81%
Saved to compact.safetensors (21.2 MB → 10.6 MB)
```

For each LoRA pair: reconstruct `Δ = B @ A`, then re-decompose at the lower rank via truncated SVD.

**Flags**:
- `--rank <int>` — Target reduced rank (required, must be less than current rank).
- `-o <path>` — Output file (default: `{stem}.r{rank}.safetensors`).

#### `sft lora add <lora_a> <lora_b> [<lora_c> ...]`

Weighted linear combination of LoRA adapters (task arithmetic).

```bash
sft lora add code-lora.safetensors chat-lora.safetensors --weights 0.6 0.4
# → writes combined.safetensors

sft lora add lora1.safetensors lora2.safetensors lora3.safetensors --weights 0.5 0.3 0.2 -o custom.safetensors
```

All input LoRAs must target the same modules and have compatible shapes. The output rank equals the input rank (combination happens on the reconstructed deltas, then re-decomposed at the original rank).

**Flags**:
- `--weights <floats...>` — Weight for each LoRA (must sum to... anything, but 1.0 is typical). If omitted, equal weights.
- `--rank <int>` — Optionally change the output rank (re-decompose at a different rank).
- `-o <path>` — Output file (default: `combined.safetensors`).

#### `sft lora compat <base> <lora>`

Check whether a LoRA adapter is compatible with a base model before attempting a merge.

```
$ sft lora compat llama-8b.safetensors adapter.safetensors

Checking compatibility...

✓ All 7 target modules found in base model
✓ All shapes compatible

Compatible: yes

$ echo $?
0
```

```
$ sft lora compat wrong-base.safetensors adapter.safetensors

Checking compatibility...

✗ Shape mismatch:
  q_proj: base [4096, 4096] vs lora_A [16, 2048]
✗ Missing modules:
  gate_proj not found in base model

Compatible: no

$ echo $?
1
```

Exits with code 0 if compatible, 1 otherwise. Designed for use in scripts and CI.

#### `sft lora svd <lora>`

Analyze the singular value spectrum of each LoRA matrix pair. Reveals how many singular values actually carry signal (the effective rank) and suggests an optimal reduced rank.

```
$ sft lora svd adapter.safetensors

Module          Rank    SV 90%    SV 95%    SV 99%    Suggested rank
q_proj           16       4         7        12         7
k_proj           16       3         5         9         5
v_proj           16       3         6        10         6
o_proj           16       5         8        13         8
gate_proj        16       6         9        14         9
up_proj          16       5         8        13         8
down_proj        16       4         7        11         7

SV X%: number of singular values needed to capture X% of total variance.
Suggested rank: captures 95% of variance.
```

**Flags**:
- `--threshold <float>` — Variance threshold for the suggested rank (default: 0.95).
- `--json` — Machine-readable output.

### 5.5. CONVERT — "Get it into safetensors"

#### `sft convert <file>`

Convert from other formats into safetensors. Auto-detects source format.

```bash
sft convert model.bin              # → writes model.safetensors
sft convert pytorch_model.pt       # → writes pytorch_model.safetensors
```

Supported source formats:
- PyTorch pickle-based checkpoints (`.bin`, `.pt`, `.pth`)

Requires `torch` — available via optional dependency: `pip install sft-cli[torch]`.

**Flags**:
- `-o <path>` — Output file (default: `{stem}.safetensors`).
- `--dtype <dtype>` — Optionally cast during conversion.

## 6. TUI Operations

The TUI is not just a viewer — it's a **home base** from which users can launch any operation interactively. Every operation available as a CLI subcommand is also accessible from within the TUI.

### 6.1. Command Palette

Press `:` (vim-style) to open the command palette. Start typing to filter:

```
┌─────────────────────────────────────────────────┐
│ : cast                                          │
│                                                 │
│   cast --dtype <dtype>    Convert tensor dtypes  │
│   check                   Validate integrity     │
│   diff <file>             Compare with file      │
│   lora info               LoRA adapter analysis  │
│   stat                    Compute statistics     │
│   ...                                           │
└─────────────────────────────────────────────────┘
```

Commands typed here operate on the currently open file. Output paths use the same smart defaults as the CLI. Results are shown in a modal or notification.

### 6.2. Contextual Keybindings

These actions are available directly from the main TUI view:

| Key | Action | Notes |
|-----|--------|-------|
| `c` | **Cast** — dtype picker modal, writes new file | Operates on full file or selected tensors |
| `d` | **Diff** — file picker, opens side-by-side diff view | Tree highlights added/removed/changed |
| `S` | **Stats** — compute statistics for selected tensor | Shows mean, std, min, max, NaN count in a modal |
| `x` | **Export/Slice** — export selected tensors to new file | Multi-select tensors first with `Space` |
| `D` | **Delete/Strip** — strip selected tensors, write new file | Multi-select tensors first with `Space` |
| `O` | **Open** — open another file (new tab) | Multi-file tabs |
| `M` | **Merge** — file picker for LoRA merge | Only shown for LoRA adapter files |
| `R` | **Resize** — rank input, compactify LoRA | Only shown for LoRA adapter files |

### 6.3. Multi-File Tabs

- Press `O` to open additional files in new tabs.
- Tab bar shows at the top: `[1: model.safetensors] [2: adapter.safetensors]`.
- Switch tabs with `1`, `2`, `3` number keys, or `Ctrl+Tab`.
- `Ctrl+D` opens a diff view comparing the current tab with a file picker.

### 6.4. LoRA-Aware Mode

When the TUI detects a LoRA adapter file (by tensor naming patterns), it auto-activates LoRA mode:

- Header shows LoRA metadata: rank, alpha, target modules, parameter count.
- Additional keybindings appear: `M` (merge), `R` (resize), `V` (SVD analysis).
- The tree groups tensors by target module and shows A/B pairs together.

### 6.5. Selection Mode

- Press `Space` on a tensor or tree node to select/deselect it.
- Selected items are highlighted. A status bar shows: "3 tensors selected (24 MB)".
- With a selection active, `x` exports only selected tensors, `c` casts only selected tensors, `D` strips selected tensors.
- Press `Esc` to clear selection.

### 6.6. Operations and Feedback

- All operations from the TUI show a progress modal for long-running tasks.
- On completion, a notification shows the output path: "Saved to model.bf16.safetensors (6.7 GB)".
- Errors are shown in a modal with the full error message.
- After a write operation, the user is offered to open the output file in a new tab.

## 7. Shared CLI Conventions

These flags behave identically across all commands that support them:

| Flag | Meaning |
|------|---------|
| `-o <path>` | Output file path. Optional — smart defaults generate a path when omitted (see §4.4). |
| `--include <glob>` | Only operate on tensors whose names match this glob. |
| `--exclude <glob>` | Skip tensors whose names match this glob. |
| `--dtype <dtype>` | Target dtype. Accepts: `fp32`, `float32`, `fp16`, `float16`, `bf16`, `bfloat16`, `fp8_e4m3`, `fp8_e5m2`. |
| `--json` | Machine-readable JSON output (for read commands). |
| `--dry-run` | Show what would happen without writing any files (for write commands). |
| `--in-place` | Overwrite the input file instead of writing to a new path. Use with caution. |
| `--quiet` / `-q` | Suppress progress bars and non-essential output. |

### Tensor Glob Patterns

Tensor name matching uses a simple glob syntax:

- `*` matches any single path segment (e.g., `model.*.weight`)
- `**` matches any number of segments
- `[0-3]` matches a numeric range
- Patterns are case-sensitive

Examples:
- `"model.layers.0.*"` — all tensors in layer 0
- `"*.self_attn.*"` — all attention tensors across all layers
- `"model.layers.[0-3].*"` — layers 0 through 3
- `"!*.bias"` — negate: exclude all bias tensors

## 8. Dependency Strategy

| Tier | Packages | Install |
|------|----------|---------|
| **Core** (TUI, header inspection) | `textual`, `typer` | `pip install sft-cli` |
| **Operations** (cast, diff, LoRA, etc.) | + `safetensors`, `numpy` | `pip install sft-cli` |
| **Convert** | + `torch` | `pip install sft-cli[torch]` |

`safetensors` and `numpy` are added as required dependencies. They are lightweight and widely installed in ML environments. `torch` remains an optional extra because it's 2GB+ and only needed for format conversion.

## 9. Implementation Phases

### Phase 1 — Foundation + Scriptable Inspection

Establish the subcommand architecture, ops/ layer, and basic inspection.

- Refactor CLI to support subcommands (Typer command groups) while preserving `sft <file>` default behavior.
- Shared utilities: formatting, glob matching, tensor I/O, smart output defaults.
- `sft info` (`i`) — non-interactive summary with `--json`.
- `sft ls` — multi-file summary.
- `sft tree` (`t`) — ASCII tree output.
- `sft check` — structural validation.

### Phase 2 — Core Transforms

The most-requested checkpoint operations, built as ops/ + CLI wrappers.

- `sft cast` — dtype conversion.
- `sft diff` — structural and value-level comparison.
- `sft slice` + `sft strip` — extract/remove tensors.
- `sft cat` — merge files.
- `sft rename` — regex key renaming.
- `sft split` — sharding with index.json.
- `sft metadata` — view/edit metadata.

### Phase 3 — LoRA Toolkit

The differentiating feature set.

- `sft lora info` — adapter analysis.
- `sft lora merge` — apply LoRA to base model.
- `sft lora extract` — create LoRA from weight delta.
- `sft lora resize` — rank reduction (compactify).
- `sft lora compat` — compatibility check.
- `sft lora svd` — singular value analysis.
- `sft lora add` — weighted LoRA combination.

### Phase 4 — Format Conversion + Statistics

- `sft convert` — PyTorch → safetensors.
- `sft stat` — tensor-level statistics with NaN/Inf detection.

### Phase 5 — TUI Operations

Extend the existing TUI browser with interactive operations.

- Command palette (`:` keybinding).
- Contextual keybindings for cast, diff, stats, slice, strip.
- Multi-file tabs with tab switching.
- Selection mode (multi-select tensors, operate on selection).
- LoRA-aware mode with auto-detection.
- Progress modals and completion notifications.

## 10. Out of Scope

The following are explicitly **not** goals for `sft`:

- **Training or inference.** `sft` operates on files, not models.
- **Converting safetensors → PyTorch.** We convert *into* safetensors, not out of it. Safetensors is the target format.
- **Exporting tensors to PyTorch/pickle formats.** For the same reason — we encourage the safetensors ecosystem, not the legacy pickle ecosystem.
- **GGUF conversion.** Complex format with its own tooling (`llama.cpp`). Out of scope.
- **Quantization (GPTQ, AWQ, etc.).** Requires calibration data and GPU. Different category of tool.
- **HuggingFace Hub integration.** That's what `huggingface-cli` is for.
- **GPU-dependent operations.** Everything should run on CPU. Operations may be slower without GPU but must still work.

## 11. Project Structure

The key architectural decision: **ops/ contains all logic, CLI and TUI are thin interfaces.**

```
src/sft/
├── __init__.py
├── cli.py              # Typer app, subcommand registration, aliases
├── browser.py          # TUI browser + interactive operations
├── index.py            # Header parsing, TensorInfo, PrefixTree (existing)
├── ops/                # Pure operation logic — NO CLI or TUI code
│   ├── __init__.py
│   ├── info.py         # File summary logic
│   ├── stat.py         # Tensor statistics
│   ├── check.py        # File validation / health checks
│   ├── diff.py         # Structural + value diff
│   ├── cast.py         # Dtype conversion
│   ├── slice.py        # Extract tensors by pattern
│   ├── cat.py          # Merge multiple files
│   ├── split.py        # Shard by size
│   ├── rename.py       # Regex key renaming
│   ├── metadata.py     # Read/write metadata
│   ├── convert.py      # Format conversion (PyTorch → safetensors)
│   └── lora/           # LoRA operations
│       ├── __init__.py
│       ├── detect.py   # LoRA format detection, A/B pair grouping
│       ├── info.py     # LoRA-specific analysis
│       ├── merge.py    # Apply LoRA to base model
│       ├── extract.py  # Create LoRA from weight delta (SVD)
│       ├── resize.py   # Rank reduction / compactify
│       ├── add.py      # Weighted LoRA combination
│       ├── compat.py   # Base/LoRA compatibility check
│       └── svd.py      # Singular value analysis
├── commands/           # CLI layer — thin Typer wrappers around ops/
│   ├── __init__.py
│   ├── info.py         # sft info
│   ├── ls.py           # sft ls
│   ├── tree.py         # sft tree
│   ├── check.py        # sft check
│   ├── stat.py         # sft stat
│   ├── diff.py         # sft diff
│   ├── cast.py         # sft cast
│   ├── slice.py        # sft slice
│   ├── strip.py        # sft strip (uses ops/slice with inverted logic)
│   ├── cat.py          # sft cat
│   ├── split.py        # sft split
│   ├── rename.py       # sft rename
│   ├── metadata.py     # sft metadata
│   ├── convert.py      # sft convert
│   └── lora.py         # sft lora subcommand group (all lora subcommands)
└── utils/
    ├── __init__.py
    ├── tensor_io.py    # Streaming tensor read/write helpers
    ├── glob.py         # Tensor name glob matching
    ├── dtypes.py       # Dtype mapping and conversion
    ├── formatting.py   # Human-readable sizes, tables, progress bars
    ├── output.py       # Smart output path generation
    └── validation.py   # File validation helpers
```

### Layer Responsibilities

| Layer | What it does | What it does NOT do |
|-------|-------------|-------------------|
| `ops/` | All computation and file I/O. Returns data structures. Raises exceptions on errors. | No printing, no CLI args parsing, no TUI widgets. |
| `commands/` | Parses CLI args (Typer), calls ops/, formats output, handles errors. | No computation or file I/O beyond calling ops/. |
| `browser.py` | Renders TUI, handles keybindings, calls ops/ for actions, shows results in modals. | No computation or file I/O beyond calling ops/. |
| `utils/` | Shared helpers used by all layers. | No business logic. |
