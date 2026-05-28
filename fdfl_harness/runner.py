"""Thin wrapper that drives ``run_experiment_unified`` and writes a
``config.json`` next to the stage / iter CSVs so each result directory is
self-describing.

Used by the Step 2-4 advisor-review experiments. Designed to be called from
short driver scripts in ``experiments/advisor_review/`` rather than from the
command line — config dicts are easier to pass than CLI strings.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# fair_dfl is installed from the bundled wheel (see wheels/ and pyproject.toml),
# so no source-path manipulation is needed.
from fdfl_harness.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG
from fair_dfl.runner import run_experiment_unified


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy / non-JSON types to plain Python."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def make_train_cfg(
    *,
    seeds: list[int],
    lambdas: list[float],
    steps: int = 30,
    lr: float = 5e-4,
    hidden_dim: int = 32,
    n_layers: int = 2,
    arch: str = "mlp",
    decision_grad_backend: str = "spsa",
    spsa_eps: float = 5e-3,
    spsa_n_dirs: int = 4,
    batch_size: int = -1,
    device: str = "cpu",
    eval_train: bool = True,
    extra: dict | None = None,
) -> dict:
    cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    cfg.update(
        {
            "seeds": list(seeds),
            "lambdas": list(lambdas),
            "steps_per_lambda": int(steps),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "decision_grad_backend": str(decision_grad_backend),
            "decision_grad_spsa_eps": float(spsa_eps),
            "decision_grad_spsa_n_dirs": int(spsa_n_dirs),
            "device": str(device),
            "eval_train": bool(eval_train),
            "log_every": 5,
        }
    )
    cfg["model"] = {
        "arch": arch,
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": False,
        "init_mode": "default",
    }
    if extra:
        cfg.update(extra)
    return cfg


def make_md_task_cfg(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int = 5,
    n_resources: int = 2,
    scenario: str = "alpha_fair",
    alpha_fair: float = 2.0,
    poly_degree: int = 2,
    snr: float = 5.0,
    benefit_group_bias: float = 0.3,
    benefit_noise_ratio: float = 1.0,
    cost_group_bias: float = 0.0,
    cost_noise_ratio: float = 1.0,
    cost_mean: float = 1.0,
    cost_std: float = 0.2,
    budget_tightness: float = 0.5,
    fairness_type: str = "mad",
    group_ratio: float = 0.5,
    decision_mode: str = "group",
    data_seed: int = 42,
) -> dict:
    return {
        "name": "md_knapsack",
        "n_samples_train": int(n_train),
        "n_samples_val": int(n_val),
        "n_samples_test": int(n_test),
        "n_features": int(n_features),
        "n_resources": int(n_resources),
        "scenario": scenario,
        "alpha_fair": float(alpha_fair),
        "poly_degree": int(poly_degree),
        "snr": float(snr),
        "benefit_group_bias": float(benefit_group_bias),
        "benefit_noise_ratio": float(benefit_noise_ratio),
        "cost_group_bias": float(cost_group_bias),
        "cost_noise_ratio": float(cost_noise_ratio),
        "cost_mean": float(cost_mean),
        "cost_std": float(cost_std),
        "budget_tightness": float(budget_tightness),
        "fairness_type": fairness_type,
        "group_ratio": float(group_ratio),
        "decision_mode": decision_mode,
        "data_seed": int(data_seed),
    }


def make_healthcare_task_cfg(
    *,
    data_csv: str = "data/data_processed.csv",
    n_sample: int = 5000,
    alpha_fair: float = 2.0,
    fairness_type: str = "mad",
    val_fraction: float = 0.2,
    test_fraction: float = 0.5,
    decision_mode: str = "group",
    budget_rho: float = 0.35,
    data_seed: int = 42,
    split_seed: int = 2,
) -> dict:
    return {
        "name": "medical_resource_allocation",
        "data_csv": data_csv,
        "n_sample": int(n_sample),
        "data_seed": int(data_seed),
        "split_seed": int(split_seed),
        "test_fraction": float(test_fraction),
        "val_fraction": float(val_fraction),
        "alpha_fair": float(alpha_fair),
        "budget": -1,
        "budget_rho": float(budget_rho),
        "decision_mode": decision_mode,
        "fairness_type": fairness_type,
    }


def run_one(
    *,
    out_dir: Path | str,
    task_cfg: dict,
    train_cfg: dict,
    methods: list[str],
    label: str = "",
    overwrite: bool = False,
    append: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run a single experiment, persist CSVs + config.json.

    Mutually exclusive flags:
      - overwrite=True: replace existing stage_results.csv / iter_logs.csv.
      - append=True: run the requested methods and append rows to existing
        CSVs (useful for adding methods after an earlier pass).
      - default (both False): skip if stage_results.csv exists.
    """
    if overwrite and append:
        raise ValueError("run_one: pass at most one of overwrite, append")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_csv = out_dir / "stage_results.csv"
    iter_csv = out_dir / "iter_logs.csv"

    if stage_csv.exists() and not (overwrite or append):
        print(f"  [skip] {stage_csv} exists; pass overwrite=True or append=True to redo.")
        return pd.read_csv(stage_csv), pd.DataFrame(), 0.0

    method_configs = {name: copy.deepcopy(ALL_METHOD_CONFIGS[name]) for name in methods}
    cfg = {"task": dict(task_cfg), "training": dict(train_cfg)}

    t0 = time.time()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs=method_configs)
    elapsed = time.time() - t0

    if append:
        if stage_csv.exists() and not stage_df.empty:
            existing = pd.read_csv(stage_csv)
            stage_df = pd.concat([existing, stage_df], ignore_index=True)
        if iter_csv.exists() and not iter_df.empty:
            existing_iter = pd.read_csv(iter_csv)
            iter_df = pd.concat([existing_iter, iter_df], ignore_index=True)

    if not stage_df.empty:
        stage_df.to_csv(stage_csv, index=False)
    if not iter_df.empty:
        iter_df.to_csv(iter_csv, index=False)

    config_payload = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": float(elapsed),
        "task_cfg": _serialise(task_cfg),
        "train_cfg": _serialise(train_cfg),
        "methods": list(methods),
        "n_stage_rows": int(len(stage_df)),
        "n_iter_rows": int(len(iter_df)),
        "append_mode": bool(append),
    }
    config_path = out_dir / "config.json"
    if append and config_path.exists():
        # Preserve prior pass's config under a sibling filename.
        suffix = label or "appended"
        try:
            prior = json.loads(config_path.read_text())
            (out_dir / f"config_{prior.get('label', 'pass1')}.json").write_text(
                json.dumps(prior, indent=2)
            )
        except Exception:
            pass
        config_path = out_dir / f"config_{suffix}.json"
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=2)

    print(f"  [{label or out_dir.name}] {len(stage_df)} stages in {elapsed:.1f}s -> {stage_csv}")
    return stage_df, iter_df, elapsed


def run_grid(
    *,
    base_dir: Path | str,
    sweeps: list[dict],
    base_task_cfg: dict,
    base_train_cfg: dict,
    methods: list[str],
    overwrite: bool = False,
) -> list[dict]:
    """Run a list of sweep variants, each is a dict of {task: ..., train: ..., subdir: ...}."""
    base_dir = Path(base_dir)
    summary: list[dict] = []
    for sw in sweeps:
        sub = base_dir / sw["subdir"]
        task_cfg = dict(base_task_cfg)
        task_cfg.update(sw.get("task", {}))
        train_cfg = dict(base_train_cfg)
        train_cfg.update(sw.get("train", {}))
        stage_df, _, elapsed = run_one(
            out_dir=sub,
            task_cfg=task_cfg,
            train_cfg=train_cfg,
            methods=methods,
            label=sw.get("label", sw["subdir"]),
            overwrite=overwrite,
        )
        summary.append({
            "subdir": sw["subdir"],
            "label": sw.get("label", sw["subdir"]),
            "elapsed_sec": float(elapsed),
            "n_rows": int(len(stage_df)),
        })
    return summary
