# sft

A fast, interactive terminal browser for `.safetensors` files.

<p align="center">
  <img src="https://vhs.charm.sh/vhs-6eQ3Cv0oexkfUshZ7PmO3b.gif" alt="sft demo">
</p>

## Why?

If you work with ML models, you've probably found yourself wondering "what's actually in this .safetensors file?" — the layer names, shapes, dtypes, sizes. Maybe you want to check if a model has the layers you expect, compare two checkpoints, or just explore an unfamiliar architecture.

`sft` lets you do that instantly from your terminal. No Python scripts, no notebooks, no waiting for tensors to load into memory. It reads only the file header, so even multi-gigabyte models open in milliseconds.

## ⚡ Installation

The recommended way to install is via [uv](https://docs.astral.sh/uv/):

```bash
uv tool install sft-cli
```

This makes `sft` available globally as a command.

Or install with pip:

```bash
pip install sft-cli
```

## Usage

```bash
sft model.safetensors
```

That's it. Navigate with arrow keys, search with `/`, quit with `q`.

## ✨ Features

- **Hierarchical browser** — Tensors grouped by namespace (e.g., `model.layers.0.attention`)
- **Instant startup** — Header-only parsing, works on multi-GB files
- **Search** — Filter tensors by name with `/`
- **Sort** — By name, size, or rank with `s`
- **Inspect** — View full tensor details with `Space`
- **Metadata** — See embedded file metadata with `m`
- **Read-only** — Never touches your model files

## ⌨️ Keybindings

| Key | Action |
|-----|--------|
| `↑`/`↓` | Navigate |
| `←`/`→` | Collapse/expand tree |
| `Tab` | Switch panels |
| `/` | Search |
| `s` | Cycle sort mode |
| `Space` | Tensor details |
| `m` | File metadata |
| `f` | Filter by dtype |
| `q` | Quit |

## License

MIT
