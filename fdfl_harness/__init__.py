"""Shared experiment harness for the FDFL reproduction package.

Modules:
  - configs:      method registry (ALL_METHOD_CONFIGS) + training defaults.
  - runner:       thin wrapper over ``fair_dfl.runner.run_experiment_unified``
                  that persists stage_results.csv / iter_logs.csv / config.json.
  - hc_v2:        healthcare (group/individual) task + train config factories.
  - md_imbalance: the single-knob imbalance design for the MD-knapsack task.

The reusable algorithm code lives in the ``fair_dfl`` package, installed from
the bundled wheel (see ../wheels and ../pyproject.toml).
"""
