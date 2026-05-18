---
name: sft
description: Use `sft`, a fast read-only CLI, whenever you need to inspect, audit, diff, or edit `.safetensors` files (model checkpoints, LoRA adapters, finetunes). Triggers include questions about tensor names, shapes, dtypes, parameter counts, file metadata, NaN/Inf scans, LoRA rank/alpha/target modules, comparing two checkpoints, slicing/splitting/stripping/renaming tensors, converting PyTorch `.pt`/`.pth`/`.bin` checkpoints, or any task where you would otherwise write a Python script that uses `safetensors` / `torch.load` to read tensor headers. Header-only operations are O(ms) even on multi-GB files, so prefer `sft` over loading the file in Python.
---

# sft — Swiss army knife for `.safetensors` files

`sft` is a CLI installed on this system. It reads `.safetensors` headers without loading tensor data, so it is dramatically faster than Python-based introspection and works on multi-gigabyte files instantly.

## Golden rule

**When you will parse the output, always pass `--json`. Every command supports it.**

```bash
sft info  model.safetensors --json
sft ls    *.safetensors --json
sft stat  model.safetensors --json
sft diff  a.safetensors b.safetensors --json
sft lora info adapter.safetensors --json
```

Without `--json` the output is human-formatted (aligned tables, units, ANSI). With `--json` it is a stable contract you can pipe through `jq` or parse directly.

## When to use sft (vs. writing Python)

Use `sft` for:

- "What's in this `.safetensors`?" → `sft info file.safetensors --json`
- "List/sort/filter tensors" → `sft ls`, `sft tree`, `sft stat`
- "Is this file corrupt / has NaNs?" → `sft check` or `sft stat --check`
- "Compare two checkpoints" → `sft diff base.safetensors finetuned.safetensors --json`
- "Inspect a LoRA adapter" → `sft lora info adapter.safetensors --json`
- "Edit metadata / rename keys / slice / strip / split / cast dtype / merge files" → `sft metadata`, `sft rename`, `sft slice`, `sft strip`, `sft split`, `sft cast`, `sft cat`
- "Convert PyTorch `.pt`/`.pth`/`.bin` to safetensors" → `sft convert`

Do **not** use `sft` for:

- Running inference, training, or anything that requires tensor *values* outside of stats and NaN/Inf scans.
- File formats other than `.safetensors` (except `sft convert` which reads PyTorch checkpoints as input).

## Quick command map

| Goal | Command | Notes |
| --- | --- | --- |
| One-line summary of a file | `sft info FILE --json` | Tensor count, params, size, dtypes, metadata. |
| Tabular summary of many files | `sft ls *.safetensors --json` | Sortable; use `--sort=size`, `--sort=params`. |
| Per-tensor stats | `sft stat FILE --json` | mean/std/min/max/sparsity/NaN/Inf. Use `--check` to exit 1 on NaN/Inf. |
| Integrity & NaN/Inf scan | `sft check FILE` | Use `--skip-values` to skip the data scan on huge files. |
| Tree of tensor namespaces | `sft tree FILE --depth 3` | |
| Compare two files | `sft diff A B --json` | Add `--delta` for L2 norm and cosine similarity. |
| View/edit file metadata | `sft metadata FILE [--set k=v] [--unset k]` | Always writes a new file. |
| Rename tensor keys (regex) | `sft rename FILE --sub 'PAT' 'REPL'` | Repeatable; use `--dry-run`. |
| Keep/remove tensors by glob | `sft slice FILE --include='**.weight'` / `sft strip FILE --exclude='*lora_A*'` | |
| Shard by size | `sft split FILE --max-size 4GB` | Writes an index JSON. |
| Cast dtype | `sft cast FILE --dtype fp16` | `--include`/`--exclude` supported. |
| Merge multiple files | `sft cat a.safetensors b.safetensors -o merged.safetensors` | `--allow-duplicates` if names collide. |
| Convert `.pt`/`.pth`/`.bin` | `sft convert ckpt.pt` | Requires `torch`. Optional `--dtype`. |
| **LoRA: inspect adapter** | `sft lora info adapter.safetensors --json` | Rank, alpha, target modules, layer count. |
| LoRA: SVD spectrum | `sft lora svd adapter.safetensors --json` | Suggests a viable lower rank. |
| LoRA: check vs base | `sft lora compat base.safetensors adapter.safetensors` | Exit 1 if incompatible. |
| LoRA: extract from delta | `sft lora extract base.safetensors ft.safetensors --rank 16` | |
| LoRA: resize rank | `sft lora resize adapter.safetensors --rank 8` | Truncated SVD. |
| LoRA: combine adapters | `sft lora add a.safetensors b.safetensors -w 0.7 -w 0.3` | Weighted task arithmetic. |
| LoRA: convert Kohya↔PEFT | `sft lora convert adapter.safetensors --to peft` | |
| LoRA: merge into base | `sft lora merge base.safetensors adapter.safetensors` | |

Every write-side command (`cast`, `cat`, `convert`, `metadata`, `rename`, `slice`, `split`, `strip`, `lora *`) supports `--dry-run` (where it makes sense) and `-o/--output`.

## Full command reference

See `REFERENCE.md` next to this file. It is auto-generated from the CLI and lists every command, flag, and example.

## Cookbook (cross-command patterns)

Single-command examples live in `REFERENCE.md`. This section is for things `REFERENCE.md` can't show you — multi-step workflows and ways to consume `--json` output.

### Glob syntax (used by `--include` / `--exclude`)

- `*` matches any single segment (no dots).
- `**` matches any number of segments — e.g. `**.weight` matches `*.weight` at any depth.
- Plain substrings: `*lora_A*` matches anything containing `lora_A`.

### "When `info` isn't enough"

- `info` → file summary (header-only, ms).
- `tree` → namespace hierarchy.
- `stat` → per-tensor mean/std/min/max/sparsity/NaN/Inf (loads tensor data, slower).
- `check` → integrity-only health check.
- `diff` → comparing two files.

### Assert structural assumptions

```bash
sft info model.safetensors --json \
  | jq -e '.tensors > 0 and (.dtypes | has("F16"))'
```

Exit non-zero if the assertion fails — handy in CI.

### "Did this finetune actually move the layers I care about?"

```bash
sft diff base.safetensors ft.safetensors --delta --include='*self_attn*' --json \
  | jq '.value_diffs | to_entries
        | map(select(.value.l2_norm > 0.01))
        | length'
```

### Find tensors with NaN/Inf

```bash
sft stat model.safetensors --json \
  | jq '[ .[] | select(.nan_count + .inf_count > 0) | .name ]'
```

### Audit many files at once

```bash
for f in checkpoints/*.safetensors; do
  sft check "$f" --skip-values || echo "BAD: $f"
done
```

### Pre-publish checklist for a LoRA adapter

```bash
sft lora info     adapter.safetensors --json
sft lora compat   base.safetensors adapter.safetensors        # exit 1 if incompatible
sft check         adapter.safetensors                          # exit 1 if corrupt / NaN / Inf
sft metadata      adapter.safetensors --json
```

### Cheap LoRA sanity check before merging

```bash
sft lora svd      adapter.safetensors --json   # is the current rank too high?
sft lora compat   base.safetensors adapter.safetensors
sft lora merge    base.safetensors adapter.safetensors --dry-run
```

## Behaviour tips

- Exit codes are meaningful: `0` = ok, `1` = error or "issues found" (e.g. NaN/Inf with `--check`, incompatible LoRA, missing file).
- Errors go to stderr in red; `--json` output goes to stdout cleanly.
- `sft FILE.safetensors` (no subcommand) launches an interactive TUI. **Never use the bare form from an agent** — it requires a terminal. Always pass an explicit subcommand like `info`, `ls`, `stat`, or `tree`.
- `sft` only reads `.safetensors` files unless you pass `convert` (which reads PyTorch checkpoints). Other formats will be rejected with exit code 1.
