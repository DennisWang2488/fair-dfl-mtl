"""Generate per-imbalance tables for the MD-main D=3 results (paper MD main).

Reads the output tree produced by ``experiments/run_md_d3.py``:
    runs/md_d3/<fairness>/alpha_<alpha>/imb_<imb>/<method>/stage_results.csv

Drop convention (per (method, seed, lambda)): drop if
    test_regret_normalized_true > 5  OR  test_fairness > 10.
``n`` / ``n_total`` record surviving / attempted seeds per cell.

Output: markdown to stdout + one CSV per imbalance level to
    <out>/v2_d3_table_<fairness>_a<alpha>_imb<level>.csv

Usage: python aggregate/build_md_d3_tables.py [fairness=mad] [alpha=2.0]
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_IN = PKG_ROOT / "runs" / "md_d3"
DEFAULT_OUT = PKG_ROOT / "runs" / "md_d3_table"

# D=3 9-method main-text subset.
CLASS_MAP = {
    "PTO": "Baseline (unfair)",
    "FPTO": "Baseline (fair)",
    "DFL": "Decision-focused",
    "FDFL": "Decision-focused + fair",
    "FDFL-Scal-mu0.1": "Scalarized",
    "FDFL-Scal-mu1": "Scalarized",
    "FPLG-kappa1": "Prediction-anchor (with fair)",
    "PCGrad": "MOO",
    "MGDA": "MOO",
}
ORDER = list(CLASS_MAP.keys())


def load_combined(in_root: pathlib.Path, fairness: str, alpha: float) -> pd.DataFrame:
    base = in_root / fairness / f"alpha_{alpha}"
    dfs = []
    if base.exists():
        # layout: <fairness>/alpha_<a>/imb_<imb>/<method>/stage_results.csv
        for p in sorted(base.rglob("stage_results.csv")):
            rel = p.relative_to(base)  # imb_<imb>/<method>/stage_results.csv
            imb = float(rel.parts[0].split("_")[1])
            label = rel.parts[1]
            d = pd.read_csv(p)
            d["imb"] = imb
            d["label"] = label
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["canon"] = df["label"].str.replace("-extlam", "", regex=False)
    return df


def filter_blowup(df: pd.DataFrame) -> pd.DataFrame:
    bad = (df["test_regret_normalized_true"] > 5) | (df["test_fairness"] > 10)
    return df[~bad].copy()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    clean = filter_blowup(df)
    agg = clean.groupby(["canon", "imb", "lambda"]).agg(
        n=("seed", "count"),
        reg_m=("test_regret_normalized_true", "mean"),
        reg_s=("test_regret_normalized_true", "std"),
        reg_med=("test_regret_normalized_true", "median"),
        mse_m=("test_pred_mse", "mean"),
        mse_s=("test_pred_mse", "std"),
        fair_m=("test_fairness", "mean"),
        fair_s=("test_fairness", "std"),
        time_m=("stage_wallclock_sec", "mean"),
    ).reset_index()
    n_total = df.groupby(["canon", "imb", "lambda"]).size().reset_index(name="n_total")
    return agg.merge(n_total, on=["canon", "imb", "lambda"])


def fmt(m, s):
    if pd.isna(m):
        return "--"
    if pd.isna(s):
        return f"{m:.3f}"
    return f"{m:.3f}+/-{s:.3f}"


def print_markdown_table(agg, imb, fairness, alpha):
    sub = agg[agg["imb"] == imb].copy()
    print(f"\n## D=3 alpha={alpha}, fairness={fairness}, imbalance={imb}\n")
    print("| Class | Method (lambda) | n_seeds | regret | MSE | MAD | time(s) |")
    print("|---|---|---|---|---|---|---|")
    seen_class = set()
    for canon in ORDER:
        rows = sub[sub["canon"] == canon].sort_values("lambda")
        if len(rows) == 0:
            continue
        cls = CLASS_MAP[canon]
        for _, r in rows.iterrows():
            cls_show = cls if cls not in seen_class else ""
            seen_class.add(cls)
            n_str = f"{int(r['n'])}/{int(r['n_total'])}"
            print(f"| {cls_show} | {canon} (lam={r['lambda']:.1f}) | {n_str} | "
                  f"{fmt(r['reg_m'], r['reg_s'])} | {fmt(r['mse_m'], r['mse_s'])} | "
                  f"{fmt(r['fair_m'], r['fair_s'])} | {r['time_m']:.1f} |")


def save_csv(agg, out_dir, imb, fairness, alpha):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"v2_d3_table_{fairness}_a{alpha}_imb{imb}.csv"
    sub = agg[agg["imb"] == imb].copy()
    sub["class"] = sub["canon"].map(CLASS_MAP)
    sub = sub.sort_values(["class", "canon", "lambda"])
    sub.to_csv(out, index=False)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fairness", nargs="?", default="mad")
    ap.add_argument("alpha", nargs="?", type=float, default=2.0)
    ap.add_argument("--in-root", type=pathlib.Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = load_combined(args.in_root, args.fairness, args.alpha)
    if len(df) == 0:
        print(f"No data for fairness={args.fairness}, alpha={args.alpha} under {args.in_root}")
        return
    agg = aggregate(df)
    for imb in sorted(agg["imb"].unique()):
        print_markdown_table(agg, imb, args.fairness, args.alpha)
        save_csv(agg, args.out, imb, args.fairness, args.alpha)


if __name__ == "__main__":
    main()
