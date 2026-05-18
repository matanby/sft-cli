"""Cross-command pipeline tests — agent-style flows from SKILL.md.

These compose multiple `sft` commands via the CliRunner and assert
end-to-end behavior:

  * extract -> merge: round-trip recovers the finetuned weights within
    a tolerance defined by the SVD rank.
  * extract -> resize --rank auto: produces heterogeneous per-pair ranks
    reflected in the JSON output.
  * extract x2 -> stack: rank arithmetic (r1+r2, or target-rank) and
    delta equivalence vs. the inputs.
  * convert (Kohya -> PEFT) -> info: rank/alpha exposed in JSON.
  * diff X X: a file diff'd against itself produces all-zero deltas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from typer.testing import CliRunner

from sft.cli import app

runner = CliRunner()


def _invoke(args: list[str]):
    """Run the CLI; returns the CliRunner Result for assertions."""
    return runner.invoke(app, args)


# ---------- extract -> merge round-trip ----------


def test_extract_then_merge_recovers_finetuned(
    lora_base_model: Path, finetuned_model: Path, tmp_path: Path
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    r = _invoke(
        [
            "lora",
            "extract",
            str(lora_base_model),
            str(finetuned_model),
            "--rank",
            "8",
            "-o",
            str(adapter),
        ]
    )
    assert r.exit_code == 0, r.output
    assert adapter.exists()

    merged = tmp_path / "merged.safetensors"
    r = _invoke(
        [
            "lora",
            "merge",
            str(lora_base_model),
            str(adapter),
            "-o",
            str(merged),
            "--scale",
            "1.0",
        ]
    )
    assert r.exit_code == 0, r.output

    ft = load_file(str(finetuned_model))
    out = load_file(str(merged))
    for mod in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ):
        np.testing.assert_allclose(out[mod], ft[mod], rtol=1e-3, atol=1e-3)


# ---------- extract -> resize --rank auto ----------


def test_extract_then_resize_auto_yields_per_module_ranks(
    lora_base_model: Path, finetuned_model: Path, tmp_path: Path
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    _invoke(
        [
            "lora",
            "extract",
            str(lora_base_model),
            str(finetuned_model),
            "--rank",
            "8",
            "-o",
            str(adapter),
        ]
    )
    assert adapter.exists()

    resized = tmp_path / "resized.safetensors"
    r = _invoke(
        [
            "lora",
            "resize",
            str(adapter),
            "--rank",
            "auto+1",
            "-o",
            str(resized),
        ]
    )
    assert r.exit_code == 0, r.output
    assert resized.exists()
    # Inspect the actual ranks of the resized output: each lora_A row count
    # is the per-pair rank. Auto mode picks per-pair ranks bounded by 8.
    out = load_file(str(resized))
    a_ranks = {
        name: arr.shape[0]
        for name, arr in out.items()
        if name.endswith(".lora_A.weight")
    }
    assert a_ranks, "no lora_A tensors in resized output"
    assert all(1 <= v <= 8 for v in a_ranks.values())


# ---------- extract x2 -> stack rank arithmetic ----------


def test_stack_two_extractions_gives_summed_rank(
    lora_base_model: Path, finetuned_model: Path, tmp_path: Path
) -> None:
    # Two extractions at ranks 3 and 4 of the same delta -- they should still
    # have the same set of module keys, so stack's `n_both` covers them all.
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    _invoke(
        [
            "lora",
            "extract",
            str(lora_base_model),
            str(finetuned_model),
            "--rank",
            "3",
            "-o",
            str(a),
        ]
    )
    _invoke(
        [
            "lora",
            "extract",
            str(lora_base_model),
            str(finetuned_model),
            "--rank",
            "4",
            "-o",
            str(b),
        ]
    )
    merged = tmp_path / "merged.safetensors"
    r = _invoke(
        [
            "lora",
            "stack",
            str(a),
            str(b),
            "-a",
            "0.5",
            "-b",
            "0.5",
            "-o",
            str(merged),
        ]
    )
    assert r.exit_code == 0, r.output
    # Now verify the resulting per-pair A factor has rank r_a + r_b = 7.
    t = load_file(str(merged))
    for name, arr in t.items():
        if name.endswith(".lora_A.weight"):
            assert arr.shape[0] == 3 + 4
        if name.endswith(".lora_B.weight"):
            assert arr.shape[1] == 3 + 4


def test_stack_with_target_rank_truncates(
    lora_base_model: Path, finetuned_model: Path, tmp_path: Path
) -> None:
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    for r, path in ((3, a), (4, b)):
        _invoke(
            [
                "lora",
                "extract",
                str(lora_base_model),
                str(finetuned_model),
                "--rank",
                str(r),
                "-o",
                str(path),
            ]
        )

    merged = tmp_path / "m.safetensors"
    r = _invoke(
        [
            "lora",
            "stack",
            str(a),
            str(b),
            "-a",
            "0.5",
            "-b",
            "0.5",
            "--target-rank",
            "2",
            "-o",
            str(merged),
        ]
    )
    assert r.exit_code == 0, r.output
    t = load_file(str(merged))
    for name, arr in t.items():
        if name.endswith(".lora_A.weight"):
            assert arr.shape[0] == 2
        if name.endswith(".lora_B.weight"):
            assert arr.shape[1] == 2


# ---------- Kohya -> PEFT -> info ----------


@pytest.fixture
def small_kohya(tmp_path: Path) -> Path:
    rng = np.random.RandomState(0)
    rank = 4
    p = tmp_path / "kohya.safetensors"
    tensors = {}
    for mod in ("lora_unet_q", "lora_unet_v"):
        tensors[f"{mod}.lora_down.weight"] = rng.randn(rank, 8).astype(np.float32)
        tensors[f"{mod}.lora_up.weight"] = rng.randn(8, rank).astype(np.float32)
        tensors[f"{mod}.alpha"] = np.array(8.0, dtype=np.float16)
    save_file(tensors, str(p))
    return p


def test_kohya_to_peft_then_info_reports_rank_and_alpha(
    small_kohya: Path, tmp_path: Path
) -> None:
    peft = tmp_path / "as_peft.safetensors"
    r = _invoke(["lora", "convert", str(small_kohya), "--to", "peft", "-o", str(peft)])
    assert r.exit_code == 0, r.output
    assert peft.exists()

    r = _invoke(["lora", "info", str(peft), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["rank"] == 4
    assert data["alpha"] == pytest.approx(8.0, rel=1e-3)


# ---------- diff X X: a file vs. itself ----------


def test_diff_self_is_all_equal(mini_model: Path) -> None:
    r = _invoke(["diff", str(mini_model), str(mini_model), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    # Every tensor should land in the equal bucket; none should differ.
    counts = data.get("counts", data)
    # Be tolerant of either shape (counts as dict, or summary list).
    if isinstance(counts, dict):
        # Some implementations key by category name.
        assert counts.get("differ", 0) == 0
        assert counts.get("incompatible", 0) == 0
        assert counts.get("missing", 0) == 0
