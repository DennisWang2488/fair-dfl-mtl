"""Shared experiment configuration — method registry, training defaults, and plot styling."""

from __future__ import annotations

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
ALPHA_VALUES = [0.5, 2.0]
N_SAMPLE_SMALL = 500  # quick verification
N_SAMPLE_FULL = 0     # 0 = use all patients (48,784)

# ---------------------------------------------------------------------------
# Method registry
#
# Each method config declares its objective flags explicitly:
#   use_dec:  use decision regret gradient
#   use_pred: use prediction (MSE) gradient
#   use_fair: use fairness gradient
#
# The "method" key maps to the training loop backend:
#   - Core backends: fpto, dfl, fdfl, moo, fair_moo, saa, var_dro, wdro
#
# MOO methods set "mo_method" to override the default gradient combination.
#
# ---------------------------------------------------------------------------
# Method taxonomy (grouped by training strategy)
# ---------------------------------------------------------------------------
#   1. Predict-then-Optimize (PTO):
#        PTO, FPTO, SAA, WDRO, VarDRO
#      No decision gradient during training; predictor is fit to outcomes
#      (MSE, or a distributionally-robust surrogate) and the solver is
#      applied post-hoc at evaluation time.
#
#   2. Static decision-focused (constant prediction weight):
#        DFL, FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal
#      The decision-regret gradient is combined with a fixed-weight
#      prediction / fairness term.  FDFL-0.1 / FDFL-0.5 use mu ∈ {0.1, 0.5}
#      as a small prediction-anchor term on top of FDFL's dec+fair; FDFL-Scal
#      uses mu=1 (i.e. standard weighted-sum scalarization).
#
#   3. Dynamic decision-focused (adaptive per-step combination):
#        PLG, FPLG, PCGrad, MGDA, CAGrad, FAMO, WS-*
#      The gradient combination rule changes every step — either through
#      PLG's alpha_t schedule or through a multi-objective handler that
#      resolves conflicts between per-objective gradients online.
# ---------------------------------------------------------------------------
#
# FDFL loss (new):
#     L_FDFL = L_regret + mu * L_pred + lambda * F
# where mu = pred_weight (static, see pred_weight_mode) and
# lambda = fairness penalty weight (via the lambda sweep).
# ---------------------------------------------------------------------------
ALL_METHOD_CONFIGS = {
    # ================================================================
    # Predict-then-optimize (PTO) — no decision gradient during training
    # ================================================================
    "PTO":    {"method": "fpto",    "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1",
               "lambdas": [0.0], "force_lambda_path_all_methods": False},
    "FPTO":   {"method": "fpto",    "use_dec": False, "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "fixed1"},
    "SAA":    {"method": "saa",     "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1"},
    "WDRO":   {"method": "wdro",    "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1", "wdro_epsilon": 0.1},
    "VarDRO": {"method": "var_dro", "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1", "dro_epsilon": 0.1},

    # ================================================================
    # Static decision-focused — constant prediction weight (mu)
    # ================================================================
    # DFL:  dec only (mu = 0, no fairness)
    # FDFL: dec + fair, mu = 0 (pure decision-focused, no prediction anchor)
    # FDFL-0.1 / FDFL-0.5: dec + mu * pred + lambda * fair, mu ∈ {0.1, 0.5}
    # FDFL-Scal: mu = 1 (standard weighted-sum scalarization of dec+pred+fair)
    "DFL":       {"method": "dfl",  "use_dec": True,  "use_pred": False, "use_fair": False,
                  "pred_weight_mode": "zero"},
    "FDFL":      {"method": "fdfl", "use_dec": True,  "use_pred": False, "use_fair": True,
                  "pred_weight_mode": "zero"},
    "FDFL-0.1":  {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "0.1", "gradient_merge": "raw"},
    "FDFL-0.5":  {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "0.5", "gradient_merge": "raw"},
    "FDFL-Scal": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "fixed1", "gradient_merge": "raw"},
    "FDFL-Scal-mu0.01": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                         "pred_weight_mode": "0.01", "gradient_merge": "raw"},
    # FDFL-Scal-mu2: mu = 2 (heavy prediction anchor). Matches the MD knapsack
    # mu-extension at FDFL-Scal-mu2 used in `tab:md-v2-imb*`.
    "FDFL-Scal-mu2": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                      "pred_weight_mode": "2.0", "gradient_merge": "raw"},

    # ================================================================
    # Dynamic decision-focused — adaptive per-step combination
    # ================================================================
    # PLG / FPLG use the alpha_t schedule from the PLG paper.
    # MOO handlers (PCGrad, MGDA, CAGrad, FAMO, WS-*) resolve
    # per-objective gradient conflicts online.  pred_weight_mode is
    # ignored in the MOO code path (the handler receives raw per-
    # objective gradients); we set "fixed1" here for clarity.
    "PLG":    {"method": "moo",   "use_dec": True,  "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "schedule"},
    # PLG-kappa1: PLG with the kappa=1 schedule (mirrors FPLG-kappa1 but without the
    # fairness regularizer). The "with-prediction-no-fairness" MD baseline.
    "PLG-kappa1": {"method": "moo", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "schedule", "mo_plg_kappa_decay": 0.01},
    "FPLG":   {"method": "fair_moo",  "use_dec": True,  "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True},
    "PCGrad": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "pcgrad", "mo_pcgrad_normalize": True},
    "MGDA":   {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "mgda"},
    "CAGrad": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "cagrad"},
    "FAMO":   {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "famo"},
    "AlignMO": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
                "mo_method": "alignmo",
                "mo_alignmo_tau_conflict": -0.1,
                "mo_alignmo_tau_scale": 2.0,
                "mo_alignmo_mu_floor": 0.1,
                "mo_alignmo_beta_ema": 0.9,
                "mo_alignmo_T_warmup": 10,
                "mo_alignmo_projection_primitive": "pcgrad"},
    "AlignMO-S": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
                  "mo_method": "alignmo_smooth",
                  "mo_alignmo_tau_conflict": -0.1,
                  "mo_alignmo_tau_scale": 2.0,
                  "mo_alignmo_mu_floor": 0.1,
                  "mo_alignmo_beta_ema": 0.9,
                  "mo_alignmo_T_warmup": 10,
                  "mo_alignmo_sharpness_scale": 0.5,
                  "mo_alignmo_sharpness_conflict": 0.5,
                  "mo_alignmo_projection_primitive": "pcgrad"},
    # ---- R7 intervention-ablation variants (all use mo_method="alignmo") ----
    "AlignMO-no-routing": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "alignmo",
        "mo_alignmo_force_no_project": True,
        "mo_alignmo_force_no_normalize": True,
        "mo_alignmo_projection_primitive": "pcgrad"},
    "AlignMO-scale-only": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "alignmo",
        "mo_alignmo_force_no_project": True,
        "mo_alignmo_projection_primitive": "pcgrad"},
    "AlignMO-conflict-only": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "alignmo",
        "mo_alignmo_force_no_normalize": True,
        "mo_alignmo_projection_primitive": "pcgrad"},
    "AlignMO-random": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "alignmo",
        "mo_alignmo_routing_mode": "random",
        "mo_alignmo_random_routing_p": 0.5,
        "mo_alignmo_projection_primitive": "pcgrad"},
    "AlignMO-frozen": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "alignmo",
        "mo_alignmo_routing_mode": "frozen",
        "mo_alignmo_projection_primitive": "pcgrad"},
    "NashMTL": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
                "mo_method": "nashmtl",
                "mo_nashmtl_n_iters": 20,
                "mo_nashmtl_normalize": True,
                "mo_nashmtl_eps": 1e-8},
    "WS-equal": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "weighted_sum",
        "mo_weights": {"decision_regret": 0.333, "pred_loss": 0.333, "pred_fairness": 0.333},
    },
    "WS-dec": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "weighted_sum",
        "mo_weights": {"decision_regret": 0.6, "pred_loss": 0.2, "pred_fairness": 0.2},
    },
    "WS-fair": {
        "method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "fixed1", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "weighted_sum",
        "mo_weights": {"decision_regret": 0.2, "pred_loss": 0.2, "pred_fairness": 0.6},
    },

    # ================================================================
    # No-fairness MOO variants (2-objective: dec + pred)
    # ================================================================
    "PCGrad-nf":  {"method": "moo", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "fixed1", "mo_method": "pcgrad",
                   "mo_pcgrad_normalize": True},
    "MGDA-nf":    {"method": "moo", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "fixed1", "mo_method": "mgda"},
    "CAGrad-nf":  {"method": "moo", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "fixed1", "mo_method": "cagrad"},
}


def describe_method(name: str, spec: dict) -> str:
    """Return a human-readable description of a method's objectives and handler."""
    objectives = []
    if spec.get("use_dec", False):
        objectives.append("dec")
    if spec.get("use_pred", False):
        objectives.append("pred")
    if spec.get("use_fair", False):
        objectives.append("fair")
    obj_str = "+".join(objectives) if objectives else "none"
    parts = [f"objectives={obj_str}"]
    mo = spec.get("mo_method", "")
    if mo:
        parts.append(f"mo={mo}")
    dgb = spec.get("decision_grad_backend", "")
    if dgb:
        parts.append(f"dec_grad={dgb}")
    return ", ".join(parts)

# ---------------------------------------------------------------------------
# Default training config
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_CFG = {
    "lambdas": [0.0, 0.5],
    "seeds": [11, 22, 33],
    "steps_per_lambda": 70,
    "batch_size": -1,
    "lr": 0.0005,
    "lr_decay": 0.0005,
    "optimizer": "sgd",             # "sgd" or "adam"
    "weight_decay": 0.0,            # L2 regularization (set to 1e-4 for regularization)
    "alpha_schedule": {"type": "inv_sqrt", "alpha0": 1.0, "alpha_min": 0.0},
    "warmstart_fraction": 0.0,
    "force_lambda_path_all_methods": False,
    "grad_clip_norm": 10000.0,
    "explode_threshold": 1000000.0,
    "fairness_smoothing": 1e-6,
    "log_every": 5,
    "pareto_sweep_mode": True,
    # NOTE: lambda_train is only used when pareto_sweep_mode=False
    # (single-lambda training). In sweep mode (our default), the
    # "lambdas" list is used instead. Kept for backward compatibility.
    "lambda_train": 0.0,
    "model": {
        "arch": "mlp",
        "hidden_dim": 64,
        "n_layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": False,
        "init_mode": "default",     # "default", "best_practice" (Kaiming He), "legacy_core"
    },
    "device": DEVICE,
}


# ---------------------------------------------------------------------------
# Task config builder
# ---------------------------------------------------------------------------
def make_task_cfg(
    data_csv: str,
    n_sample: int,
    alpha_fair: float,
    fairness_type: str = "mad",
    val_fraction: float = 0.2,
) -> dict:
    return {
        "name": "medical_resource_allocation",
        "data_csv": data_csv,
        "n_sample": n_sample,
        "data_seed": 42,
        "split_seed": 2,
        "test_fraction": 0.5,
        "val_fraction": val_fraction,
        "alpha_fair": alpha_fair,
        "budget": -1,
        "budget_rho": 0.35,
        "decision_mode": "group",
        "fairness_type": fairness_type,
    }


def compute_full_batch_size(data_csv: str, n_sample: int,
                            test_fraction: float = 0.5,
                            val_fraction: float = 0.2) -> int:
    """Compute the full training set size for use as batch_size.

    Full-batch training is required because the allocation solver needs to see
    all patients simultaneously to respect the global budget constraint.
    """
    df = pd.read_csv(data_csv)
    n_total = n_sample if (n_sample > 0 and n_sample < len(df)) else len(df)
    n_test = int(round(test_fraction * n_total))
    n_remaining = n_total - n_test
    n_val = int(round(val_fraction * n_remaining))
    n_train = n_remaining - n_val
    return n_train


# ---------------------------------------------------------------------------
# Plot styling — shared across all plots
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "FPTO": "#1f77b4", "FDFL": "#ff7f0e",
    "FDFL-0.1": "#ffa64d", "FDFL-0.5": "#ff9800", "FDFL-Scal": "#d95f02",
    "FDFL-Scal-mu0.01": "#fdae6b", "FDFL-Scal-mu2": "#a63603",
    "WS-equal": "#9467bd", "WS-dec": "#8c564b", "WS-fair": "#e377c2", "WS-balanced": "#7f7f7f",
    "MGDA": "#bcbd22", "PCGrad": "#17becf",
    "CAGrad": "#98df8a", "FAMO": "#ff9896", "AlignMO": "#2ca02c", "AlignMO-S": "#5dab50", "NashMTL": "#d62728",
    "DFL": "#c5b0d5", "PLG": "#c49c94", "PLG-kappa1": "#9c6151", "FPLG": "#f7b6d2",
    "SAA": "#e6550d", "VarDRO": "#756bb1", "WDRO": "#393b79",
    "PTO": "#636363", "PCGrad-nf": "#17becf", "MGDA-nf": "#bcbd22", "CAGrad-nf": "#98df8a",
}

MARKER_MAP = {
    "FPTO": "o", "FDFL": "s",
    "FDFL-0.1": "s", "FDFL-0.5": "s", "FDFL-Scal": "s",
    "FDFL-Scal-mu0.01": "s", "FDFL-Scal-mu2": "s",
    "WS-equal": "v", "WS-dec": "<", "WS-fair": ">", "WS-balanced": "p",
    "MGDA": "h", "PCGrad": "*",
    "CAGrad": "d", "FAMO": "H", "AlignMO": "P", "AlignMO-S": "X", "NashMTL": "X",
    "DFL": "8", "PLG": "+", "PLG-kappa1": "+", "FPLG": "x",
    "SAA": "D", "VarDRO": "p", "WDRO": "2",
    "PTO": "o", "PCGrad-nf": "*", "MGDA-nf": "h", "CAGrad-nf": "d",
}
