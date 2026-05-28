#!/usr/bin/env bash
# End-to-end reproduction driver for the FDFL experiments.
#
#   bash reproduce.sh smoke   # fast end-to-end check (subsampled / few steps)
#   bash reproduce.sh full    # full paper grids (long-running)
#
# Each track: run drivers -> raw stage_results.csv under runs/, then aggregate
# into result tables. Compare against results_reference/.
set -euo pipefail

MODE="${1:-smoke}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")"

echo "=== FDFL reproduction (mode: $MODE) ==="

# Healthcare needs the processed cohort.
if [ ! -f data/data_processed.csv ]; then
    echo "[reproduce] building data/data_processed.csv ..."
    "$PY" data/prepare_data.py
fi

if [ "$MODE" = "smoke" ]; then
    echo "--- HC group (smoke) ---"
    "$PY" experiments/run_hc_group.py --smoke --overwrite
    "$PY" aggregate/aggregate_hc_main.py

    echo "--- MD D=3 (smoke: 2 methods, 1 imbalance, 1 seed) ---"
    "$PY" experiments/run_md_d3.py --methods FPTO FDFL --imbalance 0.0 --seeds 11 --overwrite
    "$PY" aggregate/build_md_d3_tables.py mad 2.0

    echo "--- HC individual (smoke) ---"
    "$PY" experiments/run_hc_individual.py --smoke --overwrite
    "$PY" aggregate/build_individual_table.py

    echo "--- MD D=2 (smoke: imbalance mode, 1 imbalance, 1 seed) ---"
    "$PY" experiments/run_md_d2.py --mode imbalance --imbalance 0.0 --seeds 11 --overwrite
    "$PY" aggregate/md_d2_tables.py mad 2.0

elif [ "$MODE" = "full" ]; then
    echo "--- HC group (full: 5 fairness x 2 alpha x 5 seeds) ---"
    "$PY" experiments/run_hc_group.py
    "$PY" aggregate/aggregate_hc_main.py

    echo "--- MD D=3 (full: 5 imbalance x 5 seeds) ---"
    "$PY" experiments/run_md_d3.py
    "$PY" aggregate/build_md_d3_tables.py mad 2.0

    echo "--- HC individual (full: 3 fairness x 2 alpha x 5 seeds) ---"
    "$PY" experiments/run_hc_individual.py
    "$PY" aggregate/build_individual_table.py

    echo "--- MD D=2 archive (full: imbalance + extension modes) ---"
    "$PY" experiments/run_md_d2.py --mode imbalance
    "$PY" experiments/run_md_d2.py --mode extension
    "$PY" aggregate/md_d2_tables.py mad 2.0

else
    echo "Unknown mode: $MODE (use 'smoke' or 'full')"; exit 1
fi

echo ""
echo "=== done. Aggregated tables are under runs/*_table/. ==="
echo "Compare against results_reference/ ."
