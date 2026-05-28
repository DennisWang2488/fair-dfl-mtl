"""HC individual alpha-fair downstream driver (Experiment B, 2026-05-17).

Appendix-only robustness experiment for the FDFL paper. It reuses the HC v2
Variant A hyperparameters and switches the downstream decision objective from
group alpha-fair welfare to per-patient alpha-fair welfare via
``decision_mode="individual"``.

Three blocks are run:

  Block A - MAD group-MSE prediction fairness under individual downstream
            welfare.
  Block B - mean-DP prediction fairness under individual downstream welfare.
  Block C - W2-DP prediction fairness under individual downstream welfare.

Output layout:

  results/advisor_review/healthcare_individual_2026_05_17/
    variant_a/<fairness>/alpha_<a>/seed_<s>/
      stage_results.csv
      iter_logs.csv
      config.json

Cell counts:
  Each block: 2 alpha x 5 seeds x 36 method-lambda rows = 360 rows.
  Total: 3 fairness blocks x 10 cells/block = 30 cells, 1080 rows.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]

from fdfl_harness.hc_v2 import (
    HC_V2_ALPHAS,
    HC_V2_SEEDS_A,
    hc_v2_task_cfg,
    hc_v2_train_cfg_a,
)
from fdfl_harness.runner import run_one

OUT_ROOT = REPO_ROOT / "runs" / "healthcare_individual"

HC_INDIVIDUAL_FULL_POOL: List[str] = [
    "PTO",
    "FPTO",
    "SAA",
    "WDRO",
    "DFL",
    "FDFL",
    "FDFL-0.1",
    "FDFL-0.5",
    "FDFL-Scal",
    "FDFL-Scal-mu2",
    "FPLG",
    "CAGrad",
    "MGDA",
    "NashMTL",
    "PCGrad",
]

BLOCKS = {
    "block_a_mad": {
        "fairness_type": "mad",
        "methods": HC_INDIVIDUAL_FULL_POOL,
        "label_prefix": "hc_individual_a_mad",
    },
    "block_b_mean_dp": {
        "fairness_type": "dp",
        "methods": HC_INDIVIDUAL_FULL_POOL,
        "label_prefix": "hc_individual_b_dp",
    },
    "block_c_w2dp": {
        "fairness_type": "wasserstein2_dp",
        "methods": HC_INDIVIDUAL_FULL_POOL,
        "label_prefix": "hc_individual_c_w2dp",
    },
}


def run_block(
    block_name: str,
    *,
    alphas: Iterable[float] = HC_V2_ALPHAS,
    seeds: Iterable[int] = HC_V2_SEEDS_A,
    out_root: Path = OUT_ROOT,
    overwrite: bool = False,
) -> list[dict]:
    """Run one fairness block across all requested alpha/seed cells."""
    spec = BLOCKS[block_name]
    fairness = spec["fairness_type"]
    methods = spec["methods"]
    label_prefix = spec["label_prefix"]

    summary: list[dict] = []
    for alpha in alphas:
        for seed in seeds:
            task_cfg = hc_v2_task_cfg(
                fairness_type=fairness,
                alpha_fair=float(alpha),
                split_seed=int(seed),
                val_fraction=0.0,
                decision_mode="individual",
            )
            train_cfg = hc_v2_train_cfg_a(seeds=[int(seed)])
            sub = out_root / "variant_a" / fairness / f"alpha_{alpha}" / f"seed_{seed}"
            label = f"{label_prefix}_a{alpha}_s{seed}"
            t0 = time.time()
            stage_df, _, elapsed = run_one(
                out_dir=sub,
                task_cfg=task_cfg,
                train_cfg=train_cfg,
                methods=list(methods),
                label=label,
                overwrite=overwrite,
            )
            print(
                f"  [{block_name}] alpha={alpha} seed={seed}: "
                f"{len(stage_df)} stages in {elapsed:.1f}s "
                f"(cum {(time.time() - t0) / 60.0:.1f}m)"
            )
            summary.append({
                "block": block_name,
                "fairness": fairness,
                "alpha": float(alpha),
                "seed": int(seed),
                "n_stages": int(len(stage_df)),
                "elapsed_sec": float(elapsed),
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HC individual alpha-fair Experiment B runner.")
    parser.add_argument(
        "--block",
        choices=list(BLOCKS) + ["all"],
        default="all",
        help="Which block to run (default: all blocks).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        default=None,
        help="Restrict to a specific alpha (repeatable). Default: both 0.5 and 2.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Restrict to a specific seed (repeatable). Default: all 5 v2 seeds.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: run only alpha=2.0 and seed=11 for each selected block.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help=f"Output root (default: {OUT_ROOT}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage_results.csv files (default: skip).",
    )
    args = parser.parse_args()

    if args.smoke:
        alphas = [2.0]
        seeds = [11]
    else:
        alphas = args.alpha if args.alpha else HC_V2_ALPHAS
        seeds = args.seed if args.seed else HC_V2_SEEDS_A

    blocks = list(BLOCKS) if args.block == "all" else [args.block]

    t0 = time.time()
    all_summary: list[dict] = []
    for block_name in blocks:
        print(f"\n=== Running {block_name} ===")
        all_summary.extend(run_block(
            block_name,
            alphas=alphas,
            seeds=seeds,
            out_root=args.out_root,
            overwrite=args.overwrite,
        ))

    total_min = (time.time() - t0) / 60.0
    total_stages = sum(r["n_stages"] for r in all_summary)
    print(
        f"\n=== HC individual done: {len(all_summary)} cells, "
        f"{total_stages} stages, {total_min:.1f} min ==="
    )


if __name__ == "__main__":
    main()
