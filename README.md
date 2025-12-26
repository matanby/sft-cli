# sft

An interactive terminal browser for `.safetensors` files.

## Installation

The recommended way to install `sft` is via [uv](https://docs.astral.sh/uv/):

```bash
uv tool install sft-cli
```

This makes `sft` available globally as a command-line tool.

Alternatively, install via pip:

```bash
pip install sft-cli
```

Or install from source:

```bash
git clone https://github.com/matanby/sft-cli
cd sft-cli
pip install -e .
```

## Usage

```bash
sft model.safetensors
```

## Features

- **Interactive TUI** — Browse tensors with keyboard navigation
- **Hierarchy View** — Tensors organized by namespace (e.g., `unet.down_blocks.0`)
- **Fast** — Header-only parsing, instant startup even for multi-GB files
- **Safe** — Read-only, never loads tensor data
- **Search** — Find tensors by name with `/`
- **Filter** — Filter by dtype with `f`
- **Sort** — Sort by name, size, or rank with `s`
- **Details** — View tensor details with `Space`
- **Metadata** — View file metadata with `m`

## Keybindings

### Navigation
| Key | Action |
|-----|--------|
| `↑`/`↓` | Move selection |
| `←`/`→` | Collapse/Expand tree node |
| `Enter` | Select/focus node |
| `Tab` | Switch between tree and table |
| `g`/`G` | Go to top/bottom |

### Search & Filter
| Key | Action |
|-----|--------|
| `/` | Start search |
| `f` | Open filter palette |
| `Esc` | Cancel search/close dialogs |

### Sorting
| Key | Action |
|-----|--------|
| `s` | Cycle sort mode (name ↑↓, size ↑↓, rank ↑↓) |

### Inspection
| Key | Action |
|-----|--------|
| `Space` | Show tensor details |
| `m` | Show file metadata |

### Application
| Key | Action |
|-----|--------|
| `q` | Quit |

## Technical Details

- **Header-only parsing** — sft reads only the safetensors header, never loading tensor data
- **Instant startup** — Even multi-GB model files open instantly
- **Memory efficient** — Uses minimal memory regardless of file size

## License

MIT
