# Fair Decision-Focused Learning (FDFL) — reproduction package

This package reproduces the experimental results in the paper. It contains the
experiment drivers, the aggregation scripts that build the result tables, the
data-preparation script, and reference copies of the published tables for
comparison. The reusable algorithm code is shipped as a prebuilt wheel
(`wheels/fair_dfl_moo-*.whl`).

## Contents

```
ijoc_reproduce/
├── wheels/                  fair_dfl algorithm package (prebuilt wheel)
├── data/
│   ├── prepare_data.py      download public cohort + rebuild data_processed.csv
│   └── README.md            data source, license, derived-column formulas
├── fdfl_harness/            shared harness: method registry, runner, configs
├── experiments/             experiment drivers (one per result set)
│   ├── run_hc_group.py        healthcare, group alpha-fair  (main text)
│   ├── run_md_d3.py           MD knapsack D=3              (main text)
│   ├── run_hc_individual.py   healthcare, individual       (appendix)
│   └── run_md_d2.py           MD knapsack D=2              (archive/reference)
├── aggregate/               raw runs -> result tables (CSV + markdown)
│   ├── aggregate_hc_main.py
│   ├── build_md_d3_tables.py
│   ├── build_individual_table.py
│   └── md_d2_tables.py
├── results_reference/       published tables, for comparison
├── runs/                    raw + aggregated outputs land here (gitignored)
├── reproduce.sh             end-to-end driver (smoke + full modes)
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip            # Windows: .venv\Scripts\python.exe
# 1) install the algorithm package from the bundled wheel
.venv/bin/python -m pip install wheels/fair_dfl_moo-0.1.0-py3-none-any.whl
# 2) install this package + its dependencies
.venv/bin/python -m pip install -e .
```

`cvxpylayers` is a required dependency (the MD-knapsack decision-gradient
backend). The healthcare experiments use an analytic backend and do not need it.

### Solver note

The decision layers are solved with `cvxpy`. The default open-source solvers
(OSQP/SCS/ECOS) are sufficient to reproduce the results. Commercial solvers
(MOSEK/Gurobi) are **not** required and **not** included; if you have academic
licenses installed, `cvxpy` may pick them up automatically, which does not
change the reported numbers.

## Data

The healthcare experiment uses the public Obermeyer et al. cohort. Build the
processed file (downloads ~18 MB, no raw data is redistributed):

```bash
python data/prepare_data.py
```

See [data/README.md](data/README.md) for the source, license, and the exact
(verified) derived-column formulas.

## Reproduce

`reproduce.sh` runs the full pipeline. A fast smoke mode confirms the toolchain
end-to-end in a couple of minutes:

```bash
bash reproduce.sh smoke     # tiny: subsampled / few steps, ~minutes
bash reproduce.sh full      # full grids (long; see timing below)
```

Or run any track manually — each driver writes raw `stage_results.csv` files
under `runs/`, and each aggregator turns them into the result tables:

| Result set | Run | Aggregate | Reference |
|---|---|---|---|
| HC group (main) | `python experiments/run_hc_group.py` | `python aggregate/aggregate_hc_main.py` | `results_reference/healthcare/` |
| MD D=3 (main) | `python experiments/run_md_d3.py` | `python aggregate/build_md_d3_tables.py mad 2.0` | `results_reference/md_d3/` |
| HC individual (appendix) | `python experiments/run_hc_individual.py` | `python aggregate/build_individual_table.py` | `results_reference/healthcare_individual/` |
| MD D=2 (archive) | `python experiments/run_md_d2.py --mode imbalance` and `--mode extension` | `python aggregate/md_d2_tables.py mad 2.0` | `results_reference/md_d2/` |

All drivers accept `--help`. Common flags: `--seeds`, `--overwrite`, and
(HC group) `--fairness`, `--alpha`, plus `--smoke`/`--n-sample`/`--steps`.

## Experiment configuration (canonical)

- **HC group / individual**: full 48,784-patient cohort, 50% test, per-seed
  train/test split, analytic decision-gradient backend, `budget_rho=0.30`,
  `lr=1e-3`, 64-wide 2-layer MLP, 70 steps, `lambda in {0, 0.5, 1, 2}`,
  seeds `{11, 22, 33, 44, 55}`, `alpha in {0.5, 2.0}`. Each method's lambda grid
  is implied by its objective flags (fair scalarized methods sweep all four
  lambdas; MOO and non-fair methods use a single `lambda=0`).
- **MD D=3 / D=2**: synthetic knapsack, `n_train=200`, cvxpylayers backend,
  single-knob imbalance `{0.0, 0.2, 0.4, 0.6, 0.8}`, `alpha=2.0`, MAD, 5 seeds.

## Notes on faithful reproduction

- **Unified runs.** The HC group table historically accreted from several
  separate runs; here all methods run in one pass per `(fairness, alpha, seed)`
  cell. This is numerically identical because `fair_dfl` seeds every
  `(method, seed)` combination independently of which other methods share the
  invocation (verified: running methods together vs. separately gives
  bit-identical results).
- **DFL row.** The published `dfl` row is `fdfl` at `lambda=0` (decision-only
  equivalent); the HC aggregator renames it rather than running DFL separately.
- **`source` column.** `results_reference/healthcare/main_table_per_method.csv`
  carries a historical `source` provenance column; the reproduced table omits
  it (single source). All other columns match.
- **Benefit construction.** The healthcare benefit score uses the raw released
  `cost_avoidable_t` (no nearest-neighbor imputation); see
  [data/README.md](data/README.md).
