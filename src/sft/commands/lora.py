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


@lora_app.command("extract")
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
    """Extract a LoRA adapter from the delta between base and fine-tuned models."""
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


@lora_app.command("info")
def lora_info_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Analyze a LoRA adapter file."""
    file = validate_safetensors(file)

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
        module_short = pair.module_key.split(".")[-2] + "." + pair.target_module
        a_shape = "x".join(str(d) for d in pair.lora_a_shape)
        b_shape = "x".join(str(d) for d in pair.lora_b_shape)
        typer.echo(
            f"  {module_short:<30}  A [{a_shape}]  B [{b_shape}]  {format_dtype(pair.dtype)}"
        )


@lora_app.command("resize")
def lora_resize_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a LoRA adapter .safetensors file.",
        resolve_path=True,
    ),
    rank: int = typer.Option(
        ...,
        "--rank",
        "-r",
        help="Target rank (must be less than current rank).",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: {stem}.r{rank}.safetensors).",
        resolve_path=True,
    ),
) -> None:
    """Reduce LoRA rank via truncated SVD."""
    file = validate_safetensors(file)

    from sft.ops.lora.resize import resize_lora
    from sft.utils.output import resolve_output

    dst = resolve_output(output, file, f"r{rank}")

    try:
        result = resize_lora(file, dst, target_rank=rank)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Resized rank {result.original_rank} → {result.new_rank}")
    typer.echo(f"Modules resized: {result.modules_resized}")
    if result.errors:
        max_err = max(result.errors.values())
        typer.echo(f"Max reconstruction error: {max_err:.6f}")
    typer.echo(f"Written to: {dst}")


@lora_app.command("add")
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
    """Combine LoRA adapters via weighted task arithmetic."""
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


@lora_app.command("svd")
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
    """Analyze the singular value spectrum of LoRA matrices."""
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

    header = f"{'Module':<16}{'Rank':>6}{'SV 90%':>10}{'SV 95%':>10}{'SV 99%':>10}{'Suggested rank':>16}"
    typer.echo(header)
    for m in analysis.modules:
        typer.echo(
            f"{m.module:<16}{m.rank:>6}{m.sv_90:>10}{m.sv_95:>10}{m.sv_99:>10}{m.suggested_rank:>16}"
        )

    typer.echo()
    typer.echo(
        "SV X%: number of singular values needed to capture X% of total variance."
    )
    typer.echo(f"Suggested rank: captures {analysis.threshold:.0%} of variance.")


@lora_app.command("compat")
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
    """Check if a LoRA adapter is compatible with a base model."""
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


@lora_app.command("merge")
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
    """Merge a LoRA adapter into a base model."""
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
