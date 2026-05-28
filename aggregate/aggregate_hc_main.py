"""Build the HC group main per-(method, fairness, alpha, lambda) table.

Reads the single output tree produced by ``experiments/run_hc_group.py``:
    runs/healthcare_group/<fairness>/alpha_<a>/seed_<s>/stage_results.csv

and aggregates to mean +/- std over the 5 seeds.

Drop convention (per (method, cell, seed)): drop if
``test_regret_normalized > 1000``, ``nan_or_inf_steps > 0``, or
``exploding_steps > 0``.

DFL aliasing: ``fdfl`` at lambda=0 is decision-only equivalent (the fairness
gradient is scaled by lambda=0), so it is additionally surfaced as the ``dfl``
row — matching the published table, where DFL was never run as a separate
method.

Outputs (in --out, default results_reference comparison target):
  - main_table_per_method.csv
  - blowup_log.csv
  - SUMMARY.md
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PKG_ROOT / "runs" / "healthcare_group"
DEFAULT_OUT = PKG_ROOT / "runs" / "healthcare_group_table"


def parse_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.inf
    except (ValueError, TypeError):
        return math.inf


def parse_int(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def is_blowup(row):
    nis = parse_int(row.get("nan_or_inf_steps", "0"))
    es = parse_int(row.get("exploding_steps", "0"))
    rn = parse_float(row.get("test_regret_normalized", "nan"))
    if nis > 0:
        return True, f"nan_or_inf_steps={nis}"
    if es > 0:
        return True, f"exploding_steps={es}"
    if rn > 1000:
        return True, f"test_regret_normalized={rn:.2f}>1000"
    return False, ""


def load_rows(in_root: Path):
    """Load every stage_results.csv; tag fairness/alpha/seed from the path.

    Adds the dfl alias for fdfl@lambda=0.
    """
    rows = []
    for path in sorted(in_root.glob("*/alpha_*/seed_*/stage_results.csv")):
        parts = path.parts
        fairness = parts[-4]
        alpha = float(parts[-3].split("_")[1])
        seed = int(parts[-2].split("_")[1])
        with path.open() as fh:
            for r in csv.DictReader(fh):
                r["fairness_type"] = fairness
                r["alpha"] = alpha
                r["seed"] = seed
                method = r.get("method", "").lower()
                # fdfl @ lambda=0 IS the pure-DFL row (decision-only equivalent):
                # rename it to dfl, matching the published table where fdfl rows
                # start at lambda=0.5 and the lambda=0 point appears only as dfl.
                if method == "fdfl" and parse_float(r.get("lambda", "nan")) == 0.0:
                    r["_method"] = "dfl"
                else:
                    r["_method"] = method
                rows.append(r)
    return rows


def aggregate(rows, metrics=("test_regret_normalized", "test_fairness", "test_pred_mse")):
    grouped = defaultdict(list)
    blowups = []
    for r in rows:
        method = r["_method"]
        try:
            fairness = r["fairness_type"]
            alpha = float(r["alpha"])
            lam = float(r.get("lambda", 0))
            seed = int(r["seed"])
        except (KeyError, ValueError, TypeError):
            continue
        flag, reason = is_blowup(r)
        if flag:
            blowups.append({
                "method": method, "fairness": fairness, "alpha": alpha,
                "lambda": lam, "seed": seed, "reason": reason,
            })
            continue
        grouped[(method, fairness, alpha, lam)].append(r)

    out = []
    for (method, fairness, alpha, lam), rs in sorted(grouped.items()):
        row = {
            "method": method, "fairness": fairness,
            "alpha": alpha, "lambda": lam, "n_seeds": len(rs),
        }
        for m in metrics:
            vals = [parse_float(r.get(m, "nan")) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                row[f"{m}_mean"] = statistics.mean(vals)
                row[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            else:
                row[f"{m}_mean"] = math.nan
                row[f"{m}_std"] = math.nan
        out.append(row)
    return out, blowups


def write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {path.name})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=seen, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-root", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_rows(args.in_root)
    print(f"Loaded {len(rows)} stage rows (incl. dfl aliases) from {args.in_root}")
    agg, blow = aggregate(rows)
    print(f"Aggregated {len(agg)} cells; {len(blow)} blowup rows dropped")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "main_table_per_method.csv", agg)
    write_csv(args.out / "blowup_log.csv", blow)
    n_methods = len({r["method"] for r in agg})
    (args.out / "SUMMARY.md").write_text(
        f"# HC group main table\n\n"
        f"- cells: {len(agg)}\n- methods: {n_methods}\n- blowups dropped: {len(blow)}\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}/main_table_per_method.csv ({n_methods} methods)")


if __name__ == "__main__":
    main()
