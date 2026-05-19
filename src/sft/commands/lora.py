"""CLI sub-app for sft lora commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sft.cli import app, validate_safetensors
from sft.utils.formatting import format_dtype, format_number

lora_app = typer.Typer(
    name="lora",
    help="LoRA adapter operations.",
    no_args_is_help=True,
)
app.add_typer(lora_app, name="lora", rich_help_panel="LoRA")


@lora_app.command("extract", no_args_is_help=True)
def lora_extract_cmd(
    base: Path = typer.Argument(
        ...,
        help="Path to the base model .safetensors file.",
        resolve_path=True,
    ),
    finetuned: Path = typer.Argument(
        ...,
        help="Path to the fine-tuned model .safetensors file.",
        resolve_path=True,
    ),
    rank: int = typer.Option(
        ...,
        "--rank",
        "-r",
        help="Target LoRA rank.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (default: {base_stem}.lora-r{rank}.safetensors).",
    ),
    alpha: float | None = typer.Option(
        None,
        "--alpha",
        help="LoRA alpha for metadata (default: same as rank).",
    ),
    include: str | None = typer.Option(
        None,
        "--include",
        help="Glob pattern for tensor names to include.",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Glob pattern for tensor names to exclude.",
    ),
) -> None:
    """Extract a LoRA adapter from the delta between base and fine-tuned models.

    Computes low-rank approximation of (finetuned - base) weight deltas
    using SVD decomposition.

    Examples:
      sft lora extract base.safetensors finetuned.safetensors --rank 16
      sft lora extract base.safetensors ft.safetensors -r 64 -o adapter.safetensors
      sft lora extract base.safetensors ft.safetensors -r 32 --include='*self_attn*'
    """
    base = validate_safetensors(base)
    finetuned = validate_safetensors(finetuned)

    from sft.ops.lora.extract import extract_lora

    result = extract_lora(
        base_path=base,
        finetuned_path=finetuned,
        dst=output,
        rank=rank,
        include=include,
        exclude=exclude,
        alpha=alpha,
    )

    if not result.modules:
        typer.secho(
            "No eligible weight matrices found.", fg=typer.colors.YELLOW, err=True
        )
        raise typer.Exit(code=1)

    typer.echo(f"Extracted LoRA adapter (rank {result.rank}):")
    for module in result.modules:
        err = result.errors.get(module, 0.0)
        typer.echo(f"  {module}  error={err:.6f}")
    typer.echo(f"\nWrote {result.output_path}")


@lora_app.command("info", no_args_is_help=True)
def lora_info_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Analyze a LoRA adapter file.

    Shows rank, alpha, target modules, layer count, and per-pair shapes.

    Examples:
      sft lora info adapter.safetensors
      sft lora info adapter.safetensors --json
    """
    file = validate_safetensors(file)

    from sft.ops.lora.detect import format_lora_module_display
    from sft.ops.lora.info import lora_info

    try:
        info = lora_info(file)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        data = {
            "file": file.name,
            "rank": info.rank,
            "alpha": info.alpha,
            "effective_scale": info.effective_scale,
            "target_modules": info.target_modules,
            "num_layers": info.num_layers,
            "total_params": info.total_params,
            "pairs": len(info.pairs),
            "metadata": info.metadata,
        }
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo(f"LoRA Adapter:    {file.name}")
    typer.echo(f"Rank:            {info.rank}")
    if info.alpha is not None:
        typer.echo(
            f"Alpha:           {info.alpha:.0f} (effective scale: {info.effective_scale:.1f})"
        )
    typer.echo(f"Target modules:  {', '.join(info.target_modules)}")
    typer.echo(f"Layers:          {info.num_layers}")
    typer.echo(
        f"Parameters:      {info.total_params:,} ({format_number(info.total_params)})"
    )
    typer.echo()

    typer.echo("Pairs:")
    for pair in info.pairs:
        module_short = format_lora_module_display(pair.module_key)
        a_shape = "x".join(str(d) for d in pair.lora_a_shape)
        b_shape = "x".join(str(d) for d in pair.lora_b_shape)
        typer.echo(
            f"  {module_short:<30}  A [{a_shape}]  B [{b_shape}]  {format_dtype(pair.dtype)}"
        )


def _parse_rank_spec(spec: str) -> tuple[int | None, int | None]:
    """Parse a rank spec into (target_rank, auto_margin).

    Accepts:
      "8"      -> (8, None)
      "auto"   -> (None, 1)
      "auto+N" -> (None, N) for any non-negative N
    """
    s = spec.strip().lower()
    if s == "auto":
        return None, 1
    if s.startswith("auto+"):
        try:
            margin = int(s[len("auto+") :])
        except ValueError as exc:
            raise ValueError(
                f"Invalid auto margin in {spec!r}; expected 'auto+N' with integer N"
            ) from exc
        if margin < 0:
            raise ValueError(f"auto margin must be >= 0, got {margin}")
        return None, margin
    try:
        rank = int(s)
    except ValueError as exc:
        raise ValueError(
            f"Invalid rank {spec!r}; expected a positive integer, 'auto', or 'auto+N'"
        ) from exc
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return rank, None


@lora_app.command("resize", no_args_is_help=True)
def lora_resize_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
    rank: str = typer.Option(
        ...,
        "--rank",
        "-r",
        help=(
            "Target rank — one of:\n\n"
            "  • A positive integer (e.g. 8): every pair is truncated to "
            "the same rank.\n\n"
            "  • 'auto': each pair is shrunk individually to the smallest "
            "rank that captures essentially all of its information. Pairs "
            "whose updates barely use their full rank get compressed more "
            "than pairs with rich updates, so the output file has "
            "heterogeneous per-pair ranks.\n\n"
            "  • 'auto+N': same as 'auto' but adds N as a safety margin to "
            "each pair's chosen rank (e.g. 'auto+2' keeps 2 extra singular "
            "values per pair). Use this if 'auto' compresses too aggressively."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.r{rank}.safetensors).",
        resolve_path=True,
    ),
) -> None:
    """Reduce LoRA rank via truncated SVD.

    Each A/B pair has a singular-value spectrum — most pairs concentrate
    almost all of their information in a small number of singular values,
    even when the LoRA was trained at a higher rank. `resize` exploits
    this by reconstructing each pair from fewer singular values, producing
    a smaller adapter that behaves nearly identically.

    Two ways to pick the target rank:

      • Fixed: `--rank 8` truncates every pair to rank 8.

      • Auto: `--rank auto` looks at each pair's singular values
        independently and picks the smallest rank that captures
        essentially all of its information. Pairs with simple updates
        get compressed aggressively; pairs with rich updates stay
        closer to their original rank. The output file has different
        ranks across pairs (PEFT handles this fine). Add a safety margin
        with `auto+N` (e.g. `auto+2` keeps 2 extra singular values per
        pair).

    Run `sft lora svd <file>` first to see each pair's spectrum and
    decide whether `auto` is right for your adapter.

    Examples:
      sft lora resize adapter.safetensors --rank 8
      sft lora resize adapter.safetensors --rank auto
      sft lora resize adapter.safetensors --rank auto+2 -o out.safetensors
    """
    file = validate_safetensors(file)

    from sft.ops.lora.resize import resize_lora
    from sft.utils.output import resolve_output

    try:
        target_rank, auto_margin = _parse_rank_spec(rank)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    suffix = (
        f"r{target_rank}"
        if target_rank is not None
        else ("rauto" if auto_margin == 1 else f"rauto+{auto_margin}")
    )
    dst = resolve_output(output, file, suffix)

    try:
        result = resize_lora(
            file, dst, target_rank=target_rank, auto_margin=auto_margin
        )
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if auto_margin is not None:
        ranks = result.per_module_ranks.values()
        typer.echo(
            f"Resized rank {result.original_rank} \u2192 auto (range "
            f"{min(ranks)}\u2013{max(ranks)}, margin +{auto_margin})"
        )
    else:
        typer.echo(f"Resized rank {result.original_rank} \u2192 {result.new_rank}")

    typer.echo(f"Modules resized: {result.modules_resized}")
    if result.errors:
        max_err = max(result.errors.values())
        min_energy = min(result.energies.values())
        typer.echo(f"Max reconstruction error: {max_err:.6f}")
        typer.echo(f"Min energy retained:      {min_energy:.4f}")
    typer.echo(f"Written to: {dst}")


@lora_app.command("add", no_args_is_help=True)
def lora_add_cmd(
    files: list[Path] = typer.Argument(
        ...,
        help="Two or more LoRA adapter .safetensors files to combine.",
        resolve_path=True,
    ),
    weights: list[float] | None = typer.Option(
        None,
        "--weights",
        "-w",
        help="Weight for each LoRA (default: equal weights).",
    ),
    output_rank: int | None = typer.Option(
        None,
        "--rank",
        "-r",
        help="Output rank (default: same as input).",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: combined.safetensors).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be combined without writing.",
    ),
) -> None:
    """Combine LoRA adapters via weighted task arithmetic.

    Adds multiple LoRA deltas together with optional per-adapter weights.

    Examples:
      sft lora add style.safetensors content.safetensors -o combined.safetensors
      sft lora add a.safetensors b.safetensors -w 0.7 -w 0.3
      sft lora add a.safetensors b.safetensors --rank 16 --dry-run
    """
    for f in files:
        validate_safetensors(f)

    from sft.ops.lora.add import add_loras

    try:
        result = add_loras(
            lora_paths=files,
            weights=weights,
            dst=output,
            output_rank=output_rank,
            dry_run=dry_run,
        )
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if dry_run:
        typer.echo("Dry run — no file written.")
        typer.echo(
            f"Would combine {result.combined_modules} module(s) at rank {result.output_rank}"
        )
        return

    typer.secho(
        f"Combined {result.combined_modules} module(s) at rank {result.output_rank}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Output: {result.output_path}")


@lora_app.command("stack", no_args_is_help=True)
def lora_stack_cmd(
    file_a: Path = typer.Argument(
        ...,
        help="First PEFT LoRA adapter.",
        resolve_path=True,
    ),
    file_b: Path = typer.Argument(
        ...,
        help="Second PEFT LoRA adapter.",
        resolve_path=True,
    ),
    coeff_a: float = typer.Option(
        1.0,
        "-a",
        "--coeff-a",
        help="Coefficient applied to file A.",
    ),
    coeff_b: float = typer.Option(
        1.0,
        "-b",
        "--coeff-b",
        help="Coefficient applied to file B.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {a_stem}.stack.safetensors).",
        resolve_path=True,
    ),
    target_rank: int | None = typer.Option(
        None,
        "--target-rank",
        "-r",
        help=(
            "If set, SVD-truncate each merged pair back to this rank. "
            "Default: keep the lossless merged rank (r_a + r_b)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show counts and warnings without writing.",
    ),
) -> None:
    """Lossless weighted stack of two PEFT LoRAs.

    Per module, stacks low-rank factors so the effective delta is
    `a · (B_a @ A_a) + b · (B_b @ A_b)` exactly, with merged rank `r_a + r_b`.

    Modules in only one file are kept and scaled by that side's coefficient.
    Non-LoRA tensors are passed through; collisions (same name, different
    values) produce warnings. Use `sft lora convert` first on Kohya files.

    For lossy combination at a fixed output rank, see `sft lora add`.

    Examples:
      sft lora stack a.safetensors b.safetensors
      sft lora stack a.safetensors b.safetensors -a 0.7 -b 0.3 -o mix.safetensors
      sft lora stack a.safetensors b.safetensors --target-rank 32
    """
    file_a = validate_safetensors(file_a)
    file_b = validate_safetensors(file_b)

    from sft.ops.lora.stack import stack_loras
    from sft.utils.output import resolve_output

    dst = resolve_output(output, file_a, "stack")

    try:
        result = stack_loras(
            file_a,
            file_b,
            dst,
            coeff_a=coeff_a,
            coeff_b=coeff_b,
            target_rank=target_rank,
            dry_run=dry_run,
        )
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    for msg in result.collisions:
        typer.secho(f"warning: {msg}", fg=typer.colors.YELLOW, err=True)

    typer.echo(f"Modules in both:      {result.n_both}")
    typer.echo(f"Modules only in A:    {result.n_a_only}")
    typer.echo(f"Modules only in B:    {result.n_b_only}")
    if result.n_skipped_shape:
        typer.secho(
            f"Skipped (shape mismatch): {result.n_skipped_shape}",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"Passthrough tensors:  {result.n_passthrough}")
    if target_rank is not None:
        typer.echo(f"Truncated to rank:    {target_rank}")
    else:
        typer.echo("Rank:                 lossless (r_A + r_B per module)")
    if dry_run:
        typer.echo("\n(dry run — no file written)")
    else:
        typer.secho(f"Wrote {dst}", fg=typer.colors.GREEN)


@lora_app.command("svd", no_args_is_help=True)
def lora_svd_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
    threshold: float = typer.Option(
        0.95,
        "--threshold",
        "-t",
        help="Variance threshold for suggested rank (0.0–1.0).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Analyze the singular value spectrum of LoRA matrices.

    Shows how many singular values are needed to capture 90/95/99% of
    variance, helping decide if a lower rank is viable.

    Examples:
      sft lora svd adapter.safetensors
      sft lora svd adapter.safetensors --threshold 0.99
      sft lora svd adapter.safetensors --json
    """
    file = validate_safetensors(file)

    from sft.ops.lora.svd import analyze_svd

    try:
        analysis = analyze_svd(file, threshold=threshold)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        data = {
            "threshold": analysis.threshold,
            "modules": [
                {
                    "module_key": m.module_key,
                    "module": m.module,
                    "rank": m.rank,
                    "sv_90": m.sv_90,
                    "sv_95": m.sv_95,
                    "sv_99": m.sv_99,
                    "suggested_rank": m.suggested_rank,
                    "singular_values": m.singular_values,
                }
                for m in analysis.modules
            ],
        }
        typer.echo(json.dumps(data, indent=2))
        return

    col_width = max(16, *(len(m.module) for m in analysis.modules))
    header = (
        f"{'Module':<{col_width}}{'Rank':>6}{'SV 90%':>10}{'SV 95%':>10}"
        f"{'SV 99%':>10}{'Suggested rank':>16}"
    )
    typer.echo(header)
    for m in analysis.modules:
        typer.echo(
            f"{m.module:<{col_width}}{m.rank:>6}{m.sv_90:>10}{m.sv_95:>10}"
            f"{m.sv_99:>10}{m.suggested_rank:>16}"
        )

    typer.echo()
    typer.echo(
        "SV X%: number of singular values needed to capture X% of total variance."
    )
    typer.echo(f"Suggested rank: captures {analysis.threshold:.0%} of variance.")


@lora_app.command("compat", no_args_is_help=True)
def lora_compat_cmd(
    base: Path = typer.Argument(
        ...,
        help="Path to the base model .safetensors file.",
        resolve_path=True,
    ),
    adapter: Path = typer.Argument(
        ...,
        help="Path to the LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
) -> None:
    """Check if a LoRA adapter is compatible with a base model.

    Verifies that target modules exist in the base model and shapes align.

    Examples:
      sft lora compat base.safetensors adapter.safetensors
    """
    base = validate_safetensors(base)
    adapter = validate_safetensors(adapter)

    from sft.ops.lora.compat import check_compat

    try:
        result = check_compat(base, adapter)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    for m in result.shape_mismatches:
        base_dims = list(m.base_shape)
        lora_dims = list(m.lora_a_shape)
        typer.secho(
            f"\u2717 Shape mismatch: {m.module}: "
            f"base {base_dims} vs lora_A {lora_dims}",
            fg=typer.colors.RED,
            err=True,
        )

    for mod in result.missing_modules:
        typer.secho(
            f"\u2717 Missing modules: {mod} not found in base model",
            fg=typer.colors.RED,
            err=True,
        )

    if result.compatible:
        total = result.matched_modules
        typer.secho(
            f"\u2713 All {total} target modules found in base model",
            fg=typer.colors.GREEN,
        )
        typer.secho("\u2713 All shapes compatible", fg=typer.colors.GREEN)
        typer.echo("Compatible: yes")
    else:
        typer.echo("Compatible: no")
        raise typer.Exit(code=1)


@lora_app.command("convert", no_args_is_help=True)
def lora_convert_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a LoRA adapter .safetensors file (Kohya or PEFT).",
        resolve_path=True,
    ),
    target: str | None = typer.Option(
        None,
        "--to",
        help="Target format ('kohya' or 'peft'). Default: auto-detect and "
        "convert to the other format.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.{target}.safetensors).",
        resolve_path=True,
    ),
) -> None:
    """Convert a LoRA adapter between Kohya and PEFT naming conventions."""
    file = validate_safetensors(file)

    from sft.ops.lora.convert import convert_lora, detect_format
    from sft.utils.output import resolve_output

    if target is not None and target not in ("kohya", "peft"):
        typer.secho(
            f"Error: --to must be 'kohya' or 'peft', got {target!r}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    info = detect_format(file)
    source = info.format
    if source is None:
        typer.secho(
            f"Error: no LoRA tensors found in {file.name}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if source == "mixed":
        typer.secho(
            f"Error: {file.name} contains both Kohya and PEFT modules.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    target_fmt = target or ("peft" if source == "kohya" else "kohya")
    dst = resolve_output(output, file, target_fmt)

    try:
        result = convert_lora(file, dst, target=target_fmt)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.secho(
        f"Converted {result.source_format} \u2192 {result.target_format}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Modules converted: {result.modules_converted}")
    if result.alpha is not None and result.rank is not None:
        typer.echo(f"Alpha / rank:      {result.alpha:g} / {result.rank}")
    if result.passthrough:
        typer.echo(f"Passthrough:       {result.passthrough}")
    typer.echo(f"Output:            {result.output_path}")


@lora_app.command("merge", no_args_is_help=True)
def lora_merge_cmd(
    base: Path = typer.Argument(
        ...,
        help="Path to the base model .safetensors file.",
        resolve_path=True,
    ),
    adapter: Path = typer.Argument(
        ...,
        help="Path to the LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {base_stem}.merged.safetensors).",
    ),
    scale: float | None = typer.Option(
        None,
        "--scale",
        help="Override LoRA scale (default: alpha/rank from metadata, or 1.0).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show which tensors would be modified without writing.",
    ),
) -> None:
    """Merge a LoRA adapter into a base model.

    Applies W' = W + scale * (B @ A) for each target module.

    Examples:
      sft lora merge base.safetensors adapter.safetensors
      sft lora merge base.safetensors adapter.safetensors --scale 0.5
      sft lora merge base.safetensors adapter.safetensors -o merged.safetensors --dry-run
    """
    base = validate_safetensors(base)
    adapter = validate_safetensors(adapter)

    from sft.ops.lora.merge import merge_lora

    try:
        result = merge_lora(base, adapter, dst=output, scale=scale, dry_run=dry_run)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if dry_run:
        typer.echo("Dry run — no file written.")
        typer.echo(f"Scale: {result.scale:.4f}")
        typer.echo(f"Would merge {len(result.merged_modules)} tensor(s):")
        for name in result.merged_modules:
            typer.echo(f"  {name}")
        return

    typer.secho(
        f"Merged {len(result.merged_modules)} tensor(s) (scale={result.scale:.4f})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Unchanged: {result.unchanged_tensors}")
    typer.echo(f"Output:    {result.output_path}")
