"""Healthcare follow-up v2: revised HPs + validation/early-stop variant.

Two variants of the v2 experiment:

- **Variant A** (3 seeds, no validation): reduced budget_rho=0.30,
  lambdas=[0, 0.5, 1, 2], lr=1e-3, per-seed split, no early stopping.
  Baseline for the updated setup.

- **Variant B** (2 seeds, validation + LR decay + early stopping): same
  revised HPs, plus val_fraction=0.2, stronger lr_decay, linear warmup,
  and per-K-step val evaluation with best-val snapshot restore. Two
  additional seeds [44, 55] disjoint from Variant A's [11, 22, 33].

Both variants couple ``split_seed`` to the training seed, so each seed
runs on a different train/test partition of the 48,784-patient cohort.

See ``results/advisor_review/healthcare_followup_v2/setup.md`` for
formulas, justifications, and interpretation notes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

from .runner import make_healthcare_task_cfg, make_train_cfg, run_one

REPO_ROOT = Path(__file__).resolve().parents[1]
HC_V2_DIR = REPO_ROOT / "runs" / "healthcare_group"

# ---------------------------------------------------------------------------
# v2 shared constants
# ---------------------------------------------------------------------------

HC_V2_METHODS: List[str] = [
    "FPTO",
    "FDFL-Scal",
    "FPLG",
    "PCGrad",
    "MGDA",
    "SAA",
    "WDRO",
]

HC_V2_FAIRNESS_TYPES: List[str] = ["mad", "dp", "atkinson", "bias_parity"]
HC_V2_ALPHAS: List[float] = [0.5, 2.0]

# Per-seed train/test split via split_seed = seed coupling.
# Variant A originally ran [11, 22, 33]; seeds 44 and 55 were added in a
# follow-up run to reach 5 seeds total. The loader auto-discovers all
# seed_* directories so both original and extra seeds load together.
HC_V2_SEEDS_A: List[int] = [11, 22, 33, 44, 55]
# Variant B was originally run on [44, 55]. Deprecated — the v2 paper
# numbers use Variant A's 5 seeds only.
HC_V2_SEEDS_B: List[int] = [44, 55]

HC_V2_LAMBDAS: List[float] = [0.0, 0.5, 1.0, 2.0]

# Shared HPs
HC_V2_BUDGET_RHO: float = 0.30
HC_V2_LR: float = 1e-3
HC_V2_HIDDEN_DIM: int = 64
HC_V2_N_LAYERS: int = 2
HC_V2_TEST_FRACTION: float = 0.5

# Variant A specifics
HC_V2_A_STEPS: int = 70
HC_V2_A_LR_DECAY: float = 5e-4   # v1 default; gentle reciprocal decay
HC_V2_A_VAL_FRACTION: float = 0.0

# Variant B specifics
HC_V2_B_STEPS: int = 150          # longer to give early-stop room
HC_V2_B_LR_DECAY: float = 5e-3    # 10x stronger reciprocal decay
HC_V2_B_LR_WARMUP_STEPS: int = 5  # linear warmup
HC_V2_B_VAL_FRACTION: float = 0.2
HC_V2_B_EVAL_VAL_K: int = 10      # evaluate val every 10 training steps
HC_V2_B_EARLY_STOP_METRIC: str = "val_regret"
HC_V2_B_EARLY_STOP_MIN_STEPS: int = 20  # don't stop before step 20


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def hc_v2_task_cfg(
    *,
    fairness_type: str,
    alpha_fair: float,
    split_seed: int,
    val_fraction: float = HC_V2_A_VAL_FRACTION,
    decision_mode: str = "group",
) -> dict:
    """Task cfg for v2 — couples split_seed per seed, tighter budget."""
    return make_healthcare_task_cfg(
        n_sample=0,
        val_fraction=val_fraction,
        test_fraction=HC_V2_TEST_FRACTION,
        alpha_fair=alpha_fair,
        fairness_type=fairness_type,
        decision_mode=decision_mode,
        budget_rho=HC_V2_BUDGET_RHO,
        split_seed=int(split_seed),
        data_seed=42,  # fixed; only split_seed varies per seed
    )


def hc_v2_train_cfg_a(seeds: Iterable[int] = HC_V2_SEEDS_A) -> dict:
    """Variant A: baseline (3 seeds, no val, no early stop, gentle decay).

    Uses the same lr_decay=5e-4 as v1 so the comparison is 'what does a
    tighter budget + shorter lambda range buy us?' without introducing
    confounds from a different training schedule.
    """
    return make_train_cfg(
        seeds=list(seeds),
        lambdas=list(HC_V2_LAMBDAS),
        steps=HC_V2_A_STEPS,
        lr=HC_V2_LR,
        hidden_dim=HC_V2_HIDDEN_DIM,
        n_layers=HC_V2_N_LAYERS,
        arch="mlp",
        decision_grad_backend="analytic",
        eval_train=True,
        extra={
            "lr_decay": HC_V2_A_LR_DECAY,
        },
    )


def hc_v2_train_cfg_b(seeds: Iterable[int] = HC_V2_SEEDS_B) -> dict:
    """Variant B: validation + LR schedule + early stopping (2 seeds).

    Changes from A:
    - ``steps_per_lambda=150`` (was 70) to give early stopping room
    - ``lr_decay=5e-3`` (10x stronger reciprocal decay; halves LR by ~200 steps)
    - ``lr_warmup_steps=5`` linear warmup
    - ``eval_val_every_k_steps=10`` val check every 10 steps
    - ``early_stop_metric="val_regret"`` select best by val regret
    - ``early_stop_min_steps=20`` floor to avoid stopping inside warmstart
    """
    return make_train_cfg(
        seeds=list(seeds),
        lambdas=list(HC_V2_LAMBDAS),
        steps=HC_V2_B_STEPS,
        lr=HC_V2_LR,
        hidden_dim=HC_V2_HIDDEN_DIM,
        n_layers=HC_V2_N_LAYERS,
        arch="mlp",
        decision_grad_backend="analytic",
        eval_train=True,
        extra={
            "lr_decay": HC_V2_B_LR_DECAY,
            "lr_warmup_steps": HC_V2_B_LR_WARMUP_STEPS,
            "eval_val_every_k_steps": HC_V2_B_EVAL_VAL_K,
            "early_stop_metric": HC_V2_B_EARLY_STOP_METRIC,
            "early_stop_min_steps": HC_V2_B_EARLY_STOP_MIN_STEPS,
        },
    )


# ---------------------------------------------------------------------------
# Cell / grid runners
# ---------------------------------------------------------------------------


def run_healthcare_cell_v2(
    *,
    variant: str,
    fairness_type: str,
    alpha_fair: float,
    seed: int,
    methods: Iterable[str] = HC_V2_METHODS,
    out_root: Path = HC_V2_DIR,
    overwrite: bool = False,
) -> tuple:
    """Run a single (variant, fairness_type, alpha_fair, seed) cell.

    Each seed gets its own split_seed (coupled), so each run uses a
    different train/test partition. The cell output lives under
    ``<out_root>/variant_<v>/<fairness_type>/alpha_<a>/seed_<s>``.
    """
    variant = str(variant).lower().strip()
    if variant not in {"a", "b"}:
        raise ValueError(f"variant must be 'a' or 'b', got: {variant!r}")

    val_fraction = HC_V2_A_VAL_FRACTION if variant == "a" else HC_V2_B_VAL_FRACTION
    task_cfg = hc_v2_task_cfg(
        fairness_type=fairness_type,
        alpha_fair=alpha_fair,
        split_seed=int(seed),
        val_fraction=val_fraction,
    )
    if variant == "a":
        train_cfg = hc_v2_train_cfg_a(seeds=[int(seed)])
    else:
        train_cfg = hc_v2_train_cfg_b(seeds=[int(seed)])

    sub = out_root / f"variant_{variant}" / fairness_type / f"alpha_{alpha_fair}" / f"seed_{seed}"
    return run_one(
        out_dir=sub,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=list(methods),
        label=f"hc_v2_{variant}_{fairness_type}_a{alpha_fair}_s{seed}",
        overwrite=overwrite,
    )


def run_healthcare_grid_v2(
    *,
    variant: str,
    fairness_types: Iterable[str] = HC_V2_FAIRNESS_TYPES,
    alphas: Iterable[float] = HC_V2_ALPHAS,
    seeds: Iterable[int] | None = None,
    methods: Iterable[str] = HC_V2_METHODS,
    out_root: Path = HC_V2_DIR,
    overwrite: bool = False,
) -> list[dict]:
    """Run the full v2 grid for one variant, seed by seed.

    Unlike v1 where a single run_one call handled all seeds via the
    runner's internal loop, v2 runs each seed separately so that each
    seed can use its own ``split_seed``. This means more ``run_one``
    invocations but cleaner split-seed semantics.
    """
    variant = str(variant).lower().strip()
    if seeds is None:
        seeds = HC_V2_SEEDS_A if variant == "a" else HC_V2_SEEDS_B
    summary: list[dict] = []
    for ft in fairness_types:
        for a in alphas:
            for s in seeds:
                t0 = time.time()
                stage_df, _, elapsed = run_healthcare_cell_v2(
                    variant=variant,
                    fairness_type=ft,
                    alpha_fair=a,
                    seed=int(s),
                    methods=methods,
                    out_root=out_root,
                    overwrite=overwrite,
                )
                summary.append(
                    {
                        "variant": variant,
                        "fairness_type": ft,
                        "alpha": float(a),
                        "seed": int(s),
                        "elapsed_sec": float(elapsed),
                        "n_rows": int(len(stage_df)),
                    }
                )
                print(
                    f"[v2-{variant}] {ft} alpha={a} seed={s}: "
                    f"{elapsed:.1f}s, {len(stage_df)} rows"
                )
    return summary
