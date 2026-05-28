"""MD-main-v2 at D=3: main-text method subset (2026-05-19).

Mirrors run_md_main_v2.py but with n_resources=3 and a locked main-text
subset; FDFL/FDFL-Scal carry a moderate lambda/mu grid.

  | Bucket                | Method            | Lambdas             |
  |-----------------------|-------------------|---------------------|
  | None                  | PTO               | {0}                 |
  | Prediction-focused    | FPTO              | {0, 0.5, 1}         |
  | Decision-only         | DFL               | {0}                 |
  | End-to-end (focal)    | FDFL              | {0, 0.5, 1, 2}      |
  | Scalarized            | FDFL-Scal-mu0.1   | {0, 0.5, 1, 2}      |
  | Scalarized            | FDFL-Scal-mu1     | {0, 0.5, 1, 2}      |
  | Prediction-anchor     | FPLG-kappa1       | {0, 1}              |
  | MOO                   | PCGrad            | {0}                 |
  | MOO                   | MGDA              | {0}                 |

  Lambda = 0 / 1 for FPLG-kappa1 captures the PLG-vs-FPLG contrast
  (no-fairness vs with-fairness) under the same anchor schedule.

Single-knob imbalance: same as v2, scalar parameter broadcasts per-resource
via groups[:, None]. D=2 numbers stay backward-comparable.

Output: results/md_main_v2_d3/<fairness>/alpha_<a>/imb_<level>/<method_label>/
            stage_results.csv, iter_logs.csv, config.json
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
from fdfl_harness.md_imbalance import (
    imbalance_params,
    IMBALANCE_LEVELS,
)

OUT_BASE = REPO_ROOT / "runs" / "md_d3"

N_RESOURCES = 3
ALPHA = 2.0
FAIRNESS_TYPES = ["mad"]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]
STEPS = 60
N_TRAIN = 200
N_VAL = 50
N_TEST = 100


def method_configs_subset() -> list[tuple]:
    """Locked main-text subset with explicit lambda list per method.

    Tuple = (label, base_method, overrides, lambdas).
    """
    return [
        ("PTO",              "PTO",       {},                                [0.0]),
        ("FPTO",             "FPTO",      {},                                [0.0, 0.5, 1.0]),
        ("DFL",              "DFL",       {},                                [0.0]),
        ("FDFL",             "FDFL",      {},                                [0.0, 0.5, 1.0, 2.0]),
        ("FDFL-Scal-mu0.1",  "FDFL-Scal", {"pred_weight_mode": "0.1"},       [0.0, 0.5, 1.0, 2.0]),
        ("FDFL-Scal-mu1",    "FDFL-Scal", {"pred_weight_mode": "fixed1"},    [0.0, 0.5, 1.0, 2.0]),
        ("FPLG-kappa1",      "FPLG",      {"mo_plg_kappa_decay": 0.01},      [0.0, 1.0]),
        ("PCGrad",           "PCGrad",    {},                                [0.0]),
        ("MGDA",             "MGDA",      {},                                [0.0]),
    ]


def task_cfg(*, fairness_type: str, imbalance: float) -> dict:
    p = imbalance_params(imbalance)
    return make_md_task_cfg(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
        n_features=5, n_resources=N_RESOURCES,
        scenario="alpha_fair",
        alpha_fair=ALPHA,
        poly_degree=2, snr=5.0,
        cost_mean=1.0, cost_std=0.2,
        budget_tightness=0.35,
        fairness_type=fairness_type,
        group_ratio=0.5,
        decision_mode="group",
        **p,
    )


def train_cfg(seeds: list[int], lambdas: list[float]) -> dict:
    return make_train_cfg(
        seeds=seeds,
        lambdas=list(lambdas),
        steps=STEPS,
        lr=5e-4,
        hidden_dim=32, n_layers=2, arch="mlp",
        decision_grad_backend="cvxpylayers",
        eval_train=True,
        extra={"force_lambda_path_all_methods": True},
    )


def run_one_method(
    *, method_tuple: tuple, fairness: str, imbalance: float,
    seeds: list[int], out_dir: Path, overwrite: bool,
) -> dict:
    label, base_method, overrides, lambdas = method_tuple

    from fdfl_harness.configs import ALL_METHOD_CONFIGS
    base = copy.deepcopy(ALL_METHOD_CONFIGS[base_method])
    base.update(overrides)
    ALL_METHOD_CONFIGS[label] = base

    t0 = time.time()
    stage_df, _, _ = run_one(
        out_dir=out_dir,
        task_cfg=task_cfg(fairness_type=fairness, imbalance=imbalance),
        train_cfg=train_cfg(seeds, lambdas),
        methods=[label],
        label=f"{label}_a{ALPHA}_imb{imbalance}_{fairness}",
        overwrite=overwrite,
    )
    return {
        "label": label,
        "n_rows": int(len(stage_df)),
        "wall_sec": float(time.time() - t0),
    }


def _exec_job(payload: tuple) -> dict:
    job, seeds, overwrite = payload
    return run_one_method(
        seeds=seeds, out_dir=job["out_dir"], overwrite=overwrite,
        **{k: v for k, v in job.items() if k != "out_dir"},
    )


def build_jobs(fairness_filter, imbalance_filter, methods_filter=None) -> list[dict]:
    methods = method_configs_subset()
    if methods_filter:
        wanted = set(methods_filter)
        methods = [m for m in methods if m[0] in wanted]
        if not methods:
            raise ValueError(f"--methods filter matched nothing.")
    imbalances = imbalance_filter or IMBALANCE_LEVELS

    jobs = []
    for m, imb, f in product(methods, imbalances, fairness_filter):
        out_dir = OUT_BASE / f / f"alpha_{ALPHA}" / f"imb_{imb}" / m[0]
        jobs.append({
            "method_tuple": m,
            "fairness": f,
            "imbalance": imb,
            "out_dir": out_dir,
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairness", nargs="+", default=FAIRNESS_TYPES)
    parser.add_argument("--imbalance", type=float, nargs="+", default=None,
                        help="Subset of imbalance levels (default: all 5).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Filter to specific method labels.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run one job, one seed for timing.")
    args = parser.parse_args()

    jobs = build_jobs(args.fairness, args.imbalance or [], methods_filter=args.methods)
    print(f"=== MD-main-v2 D=3 subset ===")
    print(f"n_resources : {N_RESOURCES}")
    print(f"alpha       : {ALPHA}")
    print(f"methods     : {[m[0] for m in method_configs_subset()]}")
    print(f"fairness    : {args.fairness}")
    print(f"imbalance   : {args.imbalance or IMBALANCE_LEVELS}")
    print(f"seeds       : {args.seeds}")
    print(f"jobs        : {len(jobs)}  max_workers={args.max_workers}")

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
        print(f"benchmark: 1 job 1 seed = {per_seed_full:.1f}s, {row['n_rows']} rows")
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
                print(f"  [{i}/{len(jobs)}] {row['label']} imb={j['imbalance']}: "
                      f"{row['wall_sec']:.1f}s "
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
                    print(f"  [{i}/{len(jobs)}] {row['label']} imb={j['imbalance']}: "
                          f"{row['wall_sec']:.1f}s "
                          f"(cum={(time.time()-started)/60:.1f}m)", flush=True)
                except Exception as exc:
                    print(f"  [{i}/{len(jobs)}] {j['method_tuple'][0]} FAILED: {exc!r}",
                          flush=True)

    grid_summary_path = OUT_BASE / "grid_summary.json"
    grid_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grid_summary_path, "w") as f:
        json.dump({
            "n_resources": N_RESOURCES,
            "alpha": ALPHA,
            "fairness": args.fairness,
            "imbalance": args.imbalance or "default",
            "seeds": args.seeds,
            "summary": summary,
            "grand_total_sec": float(time.time() - started),
        }, f, indent=2)
    print(f"\n=== done in {(time.time()-started)/60:.1f} min, wrote {grid_summary_path} ===")


if __name__ == "__main__":
    main()
