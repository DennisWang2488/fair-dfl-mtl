"""Single-knob imbalance design for the multi-dimensional knapsack task.

A single scalar ``imbalance in {0.0, 0.2, 0.4, 0.6, 0.8}`` scales all four
group-distortion parameters together. Extracted verbatim from the original
``run_md_main_v2.py`` so the MD-D2 archive and MD-D3 main tracks share one
definition.
"""
from __future__ import annotations

IMBALANCE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
LEVEL_NAMES = {0.0: "level0", 0.2: "level1", 0.4: "level2", 0.6: "level3", 0.8: "level4"}


def imbalance_params(level: float) -> dict:
    """Single-knob imbalance scales all 4 distortion params together."""
    return {
        "benefit_group_bias": level * 0.9,
        "benefit_noise_ratio": 1.0 + level * 1.0,
        "cost_group_bias": level * 0.9,
        "cost_noise_ratio": 1.0 + level * 1.0,
    }
