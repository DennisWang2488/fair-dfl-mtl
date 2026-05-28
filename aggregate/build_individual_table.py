"""Build the HC INDIVIDUAL alpha-fair appendix table (CSV only).

Reads the output tree produced by ``experiments/run_hc_individual.py``:
    runs/healthcare_individual/variant_a/<fairness>/alpha_<a>/seed_<s>/stage_results.csv

Drop convention (identical to the HC group main table): drop a
(method, cell, seed) if test_regret_normalized > 1000, nan_or_inf_steps > 0,
or exploding_steps > 0.

Output: <out>/individual_table_per_method.csv with one row per
(method, fairness, alpha, lambda): n, regret mean/std, fairness mean/std,
MSE mean/std.
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_IN = PKG_ROOT / "runs" / "healthcare_individual"
DEFAULT_OUT = PKG_ROOT / "runs" / "healthcare_individual_table"

FAIRNESS = ["mad", "dp", "wasserstein2_dp"]


def load_individual(in_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for fair in FAIRNESS:
        for p in sorted(in_root.glob(f"variant_a/{fair}/alpha_*/seed_*/stage_results.csv")):
            df = pd.read_csv(p)
            df["fairness"] = fair
            df["alpha"] = float(p.parts[-3].split("_")[1])
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)
    keep = (
        (full["test_regret_normalized"] <= 1000)
        & (full["nan_or_inf_steps"] == 0)
        & (full["exploding_steps"] == 0)
    )
    return full[keep].copy()


def agg(kept: pd.DataFrame) -> pd.DataFrame:
    table = (
        kept.groupby(["method", "fairness", "alpha", "lambda"])
        .agg(
            n=("seed", "count"),
            reg_norm_m=("test_regret_normalized", "mean"),
            reg_norm_s=("test_regret_normalized", "std"),
            fair_m=("test_fairness", "mean"),
            fair_s=("test_fairness", "std"),
            mse_m=("test_pred_mse", "mean"),
        )
        .reset_index()
    )
    # Match the published table's column order.
    return table[["fairness", "alpha", "method", "lambda", "n",
                  "reg_norm_m", "reg_norm_s", "fair_m", "fair_s", "mse_m"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-root", type=pathlib.Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    kept = load_individual(args.in_root)
    if len(kept) == 0:
        print(f"No data under {args.in_root}")
        return
    table = agg(kept)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "individual_table_per_method.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out}: {len(table)} cells, {table['method'].nunique()} methods, "
          f"fairness={sorted(table['fairness'].unique())}")


if __name__ == "__main__":
    main()
