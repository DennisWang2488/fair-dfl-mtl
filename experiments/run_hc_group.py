"""Healthcare GROUP alpha-fair main experiment (paper Table: HC main).

Unified runner for the 17-method HC group table. Historically these methods
accreted across several runs; here they are run in a single pass per
(fairness, alpha, seed) cell. This is numerically identical to the original
separate runs because ``fair_dfl`` seeds every (method, seed) combination
independently of which other methods share the invocation (the predictor build
seed is ``13579 + seed*101 + 1`` and the per-stage RNG depends only on the seed
and stage index — never on method identity).

Configuration (Variant A, the canonical HC setup):
  - full 48,784-patient cohort (n_sample=0), 50% test, per-seed split
  - analytic decision-gradient backend, budget_rho=0.30
  - lr=1e-3, hidden=64, 2 layers, 70 steps, lambdas=[0, 0.5, 1, 2]
  - force_lambda_path_all_methods=False, so each method's lambda grid is implied
    by its objective flags: fair scalarized methods sweep all four lambdas;
    MOO and non-fair methods run a single lambda=0.

The published ``dfl`` row is ``fdfl`` at lambda=0 (decision-only equivalent);
it is produced by the aggregator, NOT run as a separate method here.

Output: runs/healthcare_group/<fairness>/alpha_<a>/seed_<s>/stage_results.csv

Aggregate with: python aggregate/aggregate_hc_main.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from fdfl_harness.hc_v2 import (
    HC_V2_ALPHAS,
    HC_V2_SEEDS_A,
    hc_v2_task_cfg,
    hc_v2_train_cfg_a,
)
from fdfl_harness.runner import run_one

PKG_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PKG_ROOT / "runs" / "healthcare_group"

# Methods present under every fairness measure (the dfl row is fdfl@lambda=0,
# aliased in aggregation, so it is not listed as a runnable method).
FULL_POOL = [
    "FPTO", "SAA", "WDRO",
    "FDFL", "FDFL-0.1", "FDFL-0.5", "FDFL-Scal",
    "FPLG", "PCGrad", "MGDA", "NashMTL",
]

# Exact (method x fairness) coverage of the published table.
METHODS_BY_FAIRNESS = {
    "mad":             FULL_POOL + ["CAGrad", "FDFL-Scal-mu2", "FDFL-Scal-mu0.01", "PLG-kappa1"],
    "dp":              FULL_POOL + ["CAGrad", "FDFL-Scal-mu2"],
    "atkinson":        FULL_POOL,
    "bias_parity":     FULL_POOL,
    "wasserstein2_dp": FULL_POOL + ["CAGrad", "FDFL-Scal-mu2", "PTO"],
}
FAIRNESS_TYPES = list(METHODS_BY_FAIRNESS)


def run_cell(*, fairness: str, alpha: float, seed: int, out_root: Path, overwrite: bool,
             n_sample: int = 0, steps: int | None = None):
    task_cfg = hc_v2_task_cfg(
        fairness_type=fairness,
        alpha_fair=float(alpha),
        split_seed=int(seed),
        val_fraction=0.0,
        decision_mode="group",
    )
    if n_sample:  # >0 subsamples the cohort (smoke runs only; 0 = full 48,784)
        task_cfg["n_sample"] = int(n_sample)
    train_cfg = hc_v2_train_cfg_a(seeds=[int(seed)])
    if steps:
        train_cfg["steps_per_lambda"] = int(steps)
    methods = METHODS_BY_FAIRNESS[fairness]
    sub = out_root / fairness / f"alpha_{alpha}" / f"seed_{seed}"
    return run_one(
        out_dir=sub,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=methods,
        label=f"hc_group_{fairness}_a{alpha}_s{seed}",
        overwrite=overwrite,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fairness", nargs="+", default=FAIRNESS_TYPES, choices=FAIRNESS_TYPES)
    ap.add_argument("--alpha", type=float, nargs="+", default=HC_V2_ALPHAS)
    ap.add_argument("--seed", type=int, nargs="+", default=HC_V2_SEEDS_A)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-sample", type=int, default=0,
                    help="Subsample N patients (0 = full 48,784 cohort). Use for smoke runs.")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override training steps per lambda (default 70). Use for smoke runs.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny smoke run: mad / alpha=2.0 / seed=11, 800 patients, 6 steps.")
    args = ap.parse_args()

    if args.smoke:
        fairness_list, alphas, seeds = ["mad"], [2.0], [11]
        if not args.n_sample:
            args.n_sample = 800
        if args.steps is None:
            args.steps = 6
    else:
        fairness_list, alphas, seeds = args.fairness, args.alpha, args.seed

    t0 = time.time()
    n_cells = 0
    for ft in fairness_list:
        for a in alphas:
            for s in seeds:
                stage_df, _, elapsed = run_cell(
                    fairness=ft, alpha=a, seed=s,
                    out_root=args.out_root, overwrite=args.overwrite,
                    n_sample=args.n_sample, steps=args.steps,
                )
                n_cells += 1
                print(f"[hc-group] {ft} alpha={a} seed={s}: "
                      f"{len(stage_df)} rows in {elapsed:.1f}s "
                      f"(cum {(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"\n=== HC group done: {n_cells} cells in {(time.time()-t0)/60:.1f} min ===")
    print(f"    raw outputs under {args.out_root}")


if __name__ == "__main__":
    main()
