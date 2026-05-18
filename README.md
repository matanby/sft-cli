# sft

> The Swiss army knife for `.safetensors` files.

[![PyPI](https://img.shields.io/pypi/v/sft-cli?style=flat-square)](https://pypi.org/project/sft-cli/)
[![Python](https://img.shields.io/pypi/pyversions/sft-cli?style=flat-square)](https://pypi.org/project/sft-cli/)
[![CI](https://img.shields.io/github/actions/workflow/status/matanby/sft-cli/ci.yml?branch=main&style=flat-square)](https://github.com/matanby/sft-cli/actions)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

<img src="https://vhs.charm.sh/vhs-3nFXGLuvC5swABYCewgchD.gif" alt="sft demo">

`sft` is a single-binary CLI for inspecting, editing, and diffing `.safetensors` files — and an interactive terminal browser for poking around large checkpoints. Most commands read the file header only, so multi-gigabyte models open in milliseconds.

It also ships a [skill](#-ai-agents) that teaches AI coding agents (Claude Code, Cursor, Codex CLI) when to reach for it and how to parse the output.

## Install

```bash
uv tool install sft-cli                 # recommended
pip install sft-cli                     # or pip
uv tool install 'sft-cli[torch]'        # + .pt/.pth conversion
```

## 🚀 Quick start

```bash
sft model.safetensors                   # open the interactive browser
sft info model.safetensors              # one-shot summary
sft info model.safetensors --json       # machine-readable
```

The bare `sft <file>` form is a shortcut for `sft browse <file>`. Inside the browser: `↑↓` to navigate, `/` to filter, `L` on a LoRA file for LoRA Mode, `D` to diff against another file, `q` to quit.

## What it does

**Inspect** a file without loading it:

```bash
sft info  model.safetensors             # size, tensor count, dtypes, metadata
sft ls    model.safetensors             # flat list, sort/filter friendly
sft tree  model.safetensors --depth=2   # hierarchical view
sft stat  model.safetensors             # per-tensor mean/std/min/max/sparsity
sft check model.safetensors             # corruption + NaN/Inf scan
```

**Compare** two checkpoints:

```bash
sft diff base.safetensors finetuned.safetensors --delta \
    --include='**.self_attn.**'         # cosine, L2, max-abs per tensor
```

**Edit** without writing Python:

```bash
sft slice  big.safetensors --include='**.weight' -o weights-only.safetensors
sft strip  big.safetensors --exclude='*lora_*'
sft cast   model.safetensors --dtype fp16
sft cat    a.safetensors b.safetensors -o merged.safetensors
sft rename model.safetensors --sub 'model\.' 'backbone.'
sft split  model.safetensors --max-size 4GB
sft convert pytorch_model.bin           # → safetensors
```

Every write command supports `--dry-run` and never overwrites the input — outputs default to `{stem}.{suffix}.safetensors`.

**Adapter workflows** (PEFT and Kohya):

```bash
sft lora info    adapter.safetensors            # rank, alpha, target modules
sft lora svd     adapter.safetensors            # singular-value spectrum
sft lora compat  base.safetensors adapter.safetensors
sft lora extract base.safetensors ft.safetensors --rank 16
sft lora resize  adapter.safetensors --rank auto    # per-pair adaptive rank
sft lora stack   a.safetensors b.safetensors -a 0.7 -b 0.3
sft lora merge   base.safetensors adapter.safetensors
sft lora convert adapter.safetensors --to peft      # Kohya ↔ PEFT
```

`--rank auto` picks each pair's output rank from its singular-value spectrum (`ceil(stable_rank) + 1`), so over-parameterized pairs compress harder than rich ones. `auto+N` adds a safety margin.

## The browser

Press a key, get a result.

| Key | Action |
|:---:|---|
| `↑` `↓` | Navigate |
| `←` `→` | Collapse / expand tree |
| `Tab` | Switch between tree and table |
| `/` | Search / filter |
| `s` | Cycle sort |
| `Enter` | Tensor stats popup |
| `m` | File metadata |
| `c` | Cast file dtype |
| `L` | LoRA Mode (per-pair stats, SVD, compress) |
| `D` | Diff against another file |
| `:` | Command palette |
| `q` | Quit |

## 🤖 AI agents

```bash
sft skill install                       # auto-detects Claude / Cursor / Codex
sft skill status
sft skill uninstall
```

The installer symlinks `sft`'s skill into your agent's well-known skills directory (`~/.claude/skills/sft`, `~/.cursor/skills/sft`, etc.), so it stays in sync when you `uv tool upgrade sft-cli`. Pass `--mode copy` for a frozen snapshot.

Every command supports `--json` for clean parsing:

```bash
sft info model.safetensors --json | jq '.tensors'
sft lora info adapter.safetensors --json | jq '.rank'
sft stat model.safetensors --json --include='**.q_proj.*'
```

## License

MIT — see [LICENSE](LICENSE).
