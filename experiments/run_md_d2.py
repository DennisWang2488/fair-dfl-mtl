"""MD-main-v2: single-knob imbalance + cvxpylayers + 3 mode flags.

Modes:
  * main       — full method grid at MEDIUM imbalance only (both alphas)
                 ~32 method configs x 1 imbalance x 2 alphas x 5 seeds
                 = ~320 stages per fairness measure
  * imbalance  — REDUCED method pool across 5 imbalance levels, alpha=2 only
                 ~16 method configs x 5 imbalance x 1 alpha x 5 seeds
                 = ~400 stages per fairness measure
  * validation — PTO + DFL with val_set ON/OFF across 5 imbalance levels, both alphas
                 4 method configs x 5 imbalance x 2 alphas x 5 seeds
                 = 200 stages per fairness measure

Single-knob imbalance:
  imbalance ∈ {0.0, 0.2, 0.4, 0.6, 0.8}  (5 levels)
  benefit_group_bias  = imbalance * 0.9
  benefit_noise_ratio = 1.0 + imbalance * 1.0
  cost_group_bias     = imbalance * 0.9
  cost_noise_ratio    = 1.0 + imbalance * 1.0

cvxpylayer is the default decision-gradient backend.

Output: results/md_main_v2/<mode>/<fairness>/alpha_<a>/imb_<level>/<method_label>/
            stage_results.csv, iter_logs.csv, config.json

Locked spec — do not modify without updating CODEX_PROMPT.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from fdfl_harness.runner import (
    make_md_task_cfg,
    make_train_cfg,
    run_one,
)

OUT_BASE = REPO_ROOT / "runs" / "md_d2"

# ============================================================
# Single-knob imbalance design
# ============================================================
IMBALANCE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
LEVEL_NAMES = {0.0: "level0", 0.2: "level1", 0.4: "level2", 0.6: "level3", 0.8: "level4"}


def imbalance_params(level: float) -> dict:
    """Single-knob imbalance scales all 4 distortion params together."""
    return {
        "benefit_group_bias": level * 0.9,
        "benefit_noise_ratio": 1.0 + level * 1.0,
        "cost_group_bias": level * 0.9,
        "cost_noise_ratio": 1.0 + level * 1.0,
    }


# ============================================================
# Fairness measures (Gini deferred)
# ============================================================
FAIRNESS_TYPES = ["mad", "dp", "bias_parity", "atkinson"]
ALPHAS = [0.5, 2.0]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]

# Training settings (no validation, no early-stop, no grad clip — cvxpylayer is stable)
STEPS = 60
N_TRAIN = 200
N_VAL = 50  # split exists for logging but not used for early-stop
N_TEST = 100


# ============================================================
# Method config builders
# ============================================================
# A "method-config" is a (label, base_method_name, overrides, lambdas) tuple.
# `base_method_name` must exist in experiments.configs.ALL_METHOD_CONFIGS.

ScalConfigs = [
    # FDFL-Scal grid: mu in {0, 0.01, 0.1, 0.5, 1} x lambda in {0, 0.5, 1}
    # Implemented as overrides of the FDFL-Scal config: pred_weight_mode = "{mu}"
    (f"FDFL-Scal-mu{mu}", "FDFL-Scal", {"pred_weight_mode": str(mu) if mu > 0 else "zero",
                                          "use_pred": mu > 0})
    for mu in [0.0, 0.01, 0.1, 0.5, 1.0]
]

LAMBDA_SWEEP = [0.0, 0.5, 1.0]
LAMBDA_BASELINE = [0.0]
LAMBDA_EXT_HIGH = [2.0, 5.0]
LAMBDA_EXT_FULL = [0.0, 1.0, 5.0]


def method_configs_main() -> list[tuple]:
    """Full method grid for 'main' mode at medium imbalance.

    Returns a list of (label, base_method, overrides, lambdas, lambda_mode).

    lambda_mode is one of:
      "sweep"  -> use LAMBDA_SWEEP
      "zero"   -> use [0.0] only
    """
    out: list[tuple] = []

    # Baselines — unfair (no fairness gradient)
    out.append(("PTO",  "PTO",  {}, "zero"))
    out.append(("SAA",  "SAA",  {}, "zero"))
    out.append(("WDRO", "WDRO", {}, "zero"))

    # Baselines — fair LP (FPTO has use_fair=True)
    out.append(("FPTO", "FPTO", {}, "sweep"))

    # DFL (decision-only, no fair) — lambda=0 only
    out.append(("DFL", "DFL", {}, "zero"))

    # FDFL (decision + fair, mu=0) — lambda sweep
    out.append(("FDFL", "FDFL", {}, "sweep"))

    # FDFL-Scal grid: mu in {0, 0.01, 0.1, 0.5, 1} x lambda sweep
    for label, base, overrides in ScalConfigs:
        out.append((label, base, overrides, "sweep"))

    # FPLG with kappa=0 (no decay) and kappa=1 (paper-style decay)
    out.append(("FPLG-kappa0", "FPLG", {"mo_plg_kappa_decay": 0.0}, "sweep"))
    out.append(("FPLG-kappa1", "FPLG", {"mo_plg_kappa_decay": 0.01}, "sweep"))

    # MOO methods — lambda doesn't change anything for these (handler always includes fair grad)
    out.append(("PCGrad", "PCGrad", {}, "zero"))
    out.append(("MGDA",   "MGDA",   {}, "zero"))
    out.append(("NashMTL","NashMTL",{}, "zero"))

    return out


def method_configs_imbalance() -> list[tuple]:
    """Reduced method pool for 'imbalance' mode (5 imbalance levels, alpha=2 only)."""
    out: list[tuple] = []
    out.append(("PTO",  "PTO",  {}, "zero"))
    out.append(("FPTO", "FPTO", {}, "sweep"))
    out.append(("DFL",  "DFL",  {}, "zero"))
    out.append(("FDFL", "FDFL", {}, "sweep"))
    # FDFL-Scal at mu in {0.1, 1} x lambda in {0, 1}
    out.append(("FDFL-Scal-mu0.1", "FDFL-Scal",
                {"pred_weight_mode": "0.1"}, "sweep_short"))
    out.append(("FDFL-Scal-mu1",   "FDFL-Scal",
                {"pred_weight_mode": "fixed1"}, "sweep_short"))
    out.append(("PCGrad",  "PCGrad",  {}, "zero"))
    out.append(("NashMTL", "NashMTL", {}, "zero"))
    out.append(("FPLG-kappa1", "FPLG", {"mo_plg_kappa_decay": 0.01}, "sweep"))
    # SAA / WDRO / MGDA complete the archived D=2 method pool. In the original
    # study these arrived via separate makeup reruns; running them here in the
    # same pass is numerically identical (fair_dfl seeds per (method, seed)).
    out.append(("SAA",  "SAA",  {}, "zero"))
    out.append(("WDRO", "WDRO", {}, "zero"))
    out.append(("MGDA", "MGDA", {}, "zero"))
    return out


def method_configs_validation() -> list[tuple]:
    """Validation/early-stop comparison: PTO + DFL with val ON vs OFF."""
    out: list[tuple] = []
    out.append(("PTO_no_val",      "PTO", {}, "zero"))
    out.append(("PTO_with_val",    "PTO",
                {"_early_stop": True}, "zero"))
    out.append(("DFL_no_val",      "DFL", {}, "zero"))
    out.append(("DFL_with_val",    "DFL",
                {"_early_stop": True}, "zero"))
    return out


def method_configs_extension() -> list[tuple]:
    """Extension mode: higher lambdas + broader FDFL-Scal mu spectrum + CAGrad.

    All variants use NEW labels (to avoid overwriting existing 'imbalance' outputs).
    Combined with existing imbalance results, gives a full (mu, lambda) characterization.
    """
    out: list[tuple] = []
    # High-lambda extensions to existing methods (lambda in {2, 5})
    out.append(("FPTO-extlam",              "FPTO",      {},                        "ext_high"))
    out.append(("FDFL-extlam",              "FDFL",      {},                        "ext_high"))
    out.append(("FDFL-Scal-mu0.1-extlam",   "FDFL-Scal", {"pred_weight_mode": "0.1"},  "ext_high"))
    out.append(("FDFL-Scal-mu1-extlam",     "FDFL-Scal", {"pred_weight_mode": "fixed1"}, "ext_high"))
    # Broader mu spectrum for FDFL-Scal (lambda in {0, 1, 5})
    out.append(("FDFL-Scal-mu0.01",         "FDFL-Scal", {"pred_weight_mode": "0.01"},  "ext_full"))
    out.append(("FDFL-Scal-mu0.5",          "FDFL-Scal", {"pred_weight_mode": "0.5"},   "ext_full"))
    out.append(("FDFL-Scal-mu2",            "FDFL-Scal", {"pred_weight_mode": "2.0"},   "ext_full"))
    # New MOO method
    out.append(("CAGrad",                   "CAGrad",    {},                        "zero"))
    return out


def lambdas_for(mode: str) -> list[float]:
    if mode == "sweep":
        return list(LAMBDA_SWEEP)
    if mode == "sweep_short":
        return [0.0, 1.0]
    if mode == "ext_high":
        return list(LAMBDA_EXT_HIGH)
    if mode == "ext_full":
        return list(LAMBDA_EXT_FULL)
    return list(LAMBDA_BASELINE)


# ============================================================
# Task / training config builders
# ============================================================

def task_cfg(*, alpha_fair: float, fairness_type: str, imbalance: float) -> dict:
    p = imbalance_params(imbalance)
    return make_md_task_cfg(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
        n_features=5, n_resources=2,
        scenario="alpha_fair",
        alpha_fair=alpha_fair,
        poly_degree=2, snr=5.0,
        cost_mean=1.0, cost_std=0.2,
        budget_tightness=0.35,
        fairness_type=fairness_type,
        group_ratio=0.5,
        decision_mode="group",
        **p,
    )


def train_cfg(seeds: list[int], lambdas: list[float], *, early_stop: bool = False) -> dict:
    extra = {
        "force_lambda_path_all_methods": True,
    }
    if early_stop:
        # Turn early stopping on (val_regret-based, patience=10)
        extra.update({
            "early_stop_enabled": True,
            "early_stop_metric": "val_regret",
            "early_stop_patience": 10,
        })
    return make_train_cfg(
        seeds=seeds,
        lambdas=list(lambdas),
        steps=STEPS,
        lr=5e-4,
        hidden_dim=32, n_layers=2, arch="mlp",
        decision_grad_backend="cvxpylayers",
        eval_train=True,
        extra=extra,
    )


# ============================================================
# Main runner
# ============================================================

def run_one_method(
    *, method_tuple: tuple, alpha: float, fairness: str, imbalance: float,
    seeds: list[int], out_dir: Path, overwrite: bool,
) -> dict:
    """Run one method-config on one cell (alpha, fairness, imbalance level)."""
    label, base_method, overrides, lambda_mode = method_tuple
    early_stop = overrides.pop("_early_stop", False) if isinstance(overrides, dict) else False

    # Build base method config from ALL_METHOD_CONFIGS, then apply overrides.
    from fdfl_harness.configs import ALL_METHOD_CONFIGS
    base = copy.deepcopy(ALL_METHOD_CONFIGS[base_method])
    base.update(overrides)
    # Re-register as a custom-named method (run_one will look it up by label).
    ALL_METHOD_CONFIGS[label] = base

    lambdas = lambdas_for(lambda_mode)

    t0 = time.time()
    stage_df, _, _ = run_one(
        out_dir=out_dir,
        task_cfg=task_cfg(alpha_fair=alpha, fairness_type=fairness, imbalance=imbalance),
        train_cfg=train_cfg(seeds, lambdas, early_stop=early_stop),
        methods=[label],
        label=f"{label}_a{alpha}_imb{imbalance}_{fairness}",
        overwrite=overwrite,
    )
    return {
        "label": label,
        "n_rows": int(len(stage_df)),
        "wall_sec": float(time.time() - t0),
    }


def _exec_job(payload: tuple) -> dict:
    """Module-level pickleable wrapper for ProcessPoolExecutor.

    payload = (job_dict, seeds, overwrite)
    """
    job, seeds, overwrite = payload
    return run_one_method(
        seeds=seeds, out_dir=job["out_dir"], overwrite=overwrite,
        **{k: v for k, v in job.items() if k != "out_dir"},
    )


def build_jobs(mode: str, fairness_filter: list[str], alpha_filter: list[float],
               imbalance_filter: list[float], methods_filter: list[str] | None = None) -> list[dict]:
    if mode == "main":
        methods = method_configs_main()
        imbalances = [0.4]  # medium only
        alphas = alpha_filter or ALPHAS
    elif mode == "imbalance":
        methods = method_configs_imbalance()
        imbalances = imbalance_filter or IMBALANCE_LEVELS
        alphas = [2.0]  # alpha=2 only
    elif mode == "validation":
        # Validation mode is a focused diagnostic: test whether early-stop/val
        # changes training outcomes for PTO+DFL. Default to medium imbalance
        # and MAD fairness only to keep the test small (~20-30 min).
        methods = method_configs_validation()
        imbalances = imbalance_filter or [0.4]  # medium only
        alphas = alpha_filter or ALPHAS
    elif mode == "extension":
        # Extension to imbalance mode: higher lambdas + broader mu + CAGrad.
        methods = method_configs_extension()
        imbalances = imbalance_filter or IMBALANCE_LEVELS
        alphas = [2.0]  # alpha=2 only (matches imbalance)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if methods_filter:
        wanted = set(methods_filter)
        methods = [m for m in methods if m[0] in wanted]
        if not methods:
            raise ValueError(f"--methods filter matched nothing. Available: "
                             f"{[m[0] for m in methods]}")

    jobs = []
    for m, imb, a, f in product(methods, imbalances, alphas, fairness_filter):
        out_dir = OUT_BASE / mode / f / f"alpha_{a}" / f"imb_{imb}" / m[0]
        jobs.append({
            "method_tuple": m,
            "alpha": a,
            "fairness": f,
            "imbalance": imb,
            "out_dir": out_dir,
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "imbalance", "validation", "extension"],
                        required=True, help="Which experiment scope to run.")
    parser.add_argument("--fairness", nargs="+", default=FAIRNESS_TYPES,
                        help="Fairness measures to run.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Subset of alphas (defaults depend on mode).")
    parser.add_argument("--imbalance", type=float, nargs="+", default=None,
                        help="Subset of imbalance levels (defaults depend on mode).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Filter to specific method labels (e.g., --methods PTO).")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run one job for timing estimate.")
    args = parser.parse_args()

    jobs = build_jobs(args.mode, args.fairness, args.alphas or [], args.imbalance or [],
                      methods_filter=args.methods)
    print(f"=== MD-main-v2 ({args.mode}) ===")
    print(f"jobs={len(jobs)}  seeds={args.seeds}  max_workers={args.max_workers}")

    if args.benchmark:
        if not jobs:
            print("No jobs.")
            return
        j = jobs[0]
        t0 = time.time()
        row = run_one_method(seeds=args.seeds[:1], out_dir=j["out_dir"], overwrite=True,
                             **{k: v for k, v in j.items() if k != "out_dir"})
        per_seed_full = row["wall_sec"]
        full_wall_sequential = per_seed_full * len(args.seeds) * len(jobs)
        print(f"benchmark: 1 job 1 seed = {per_seed_full:.1f}s, "
              f"{row['n_rows']} rows")
        print(f"est full sweep: ~{full_wall_sequential/60:.1f}min sequential, "
              f"~{full_wall_sequential/60/args.max_workers:.1f}min with {args.max_workers} workers")
        return

    started = time.time()
    summary = []

    if args.max_workers <= 1:
        for i, j in enumerate(jobs, 1):
            try:
                row = _exec_job((j, args.seeds, args.overwrite))
                summary.append(row)
                print(f"  [{i}/{len(jobs)}] {row['label']}: {row['wall_sec']:.1f}s "
                      f"(cum={(time.time()-started)/60:.1f}m)", flush=True)
            except Exception as exc:
                print(f"  [{i}/{len(jobs)}] FAILED: {exc!r}", flush=True)
                summary.append({"label": j["method_tuple"][0], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            payloads = [(j, args.seeds, args.overwrite) for j in jobs]
            futs = {pool.submit(_exec_job, p): jobs[i] for i, p in enumerate(payloads)}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    row = fut.result()
                    summary.append(row)
                    print(f"  [{i}/{len(jobs)}] {row['label']}: {row['wall_sec']:.1f}s "
                          f"(cum={(time.time()-started)/60:.1f}m)", flush=True)
                except Exception as exc:
                    print(f"  [{i}/{len(jobs)}] {j['method_tuple'][0]} FAILED: {exc!r}", flush=True)

    grid_summary_path = OUT_BASE / args.mode / "grid_summary.json"
    grid_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grid_summary_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "fairness": args.fairness,
            "alphas": args.alphas or "default",
            "imbalance": args.imbalance or "default",
            "seeds": args.seeds,
            "summary": summary,
            "grand_total_sec": float(time.time() - started),
        }, f, indent=2)
    print(f"\n=== done in {(time.time()-started)/60:.1f} min, wrote {grid_summary_path} ===")


if __name__ == "__main__":
    main()
