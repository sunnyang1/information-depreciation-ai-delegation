"""
Experiment 17: Memory-Augmented Chains (MACLA-style)

Show that memory raises effective eta_bar but finite depth persists.
"""

from __future__ import annotations

import math

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import register
from exp_framework import TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K, KAPPA, PHI, PSI

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_COST = 0.02


def _value_fn(tau: float) -> float:
    return PSI * (1.0 - math.exp(-tau / 200.0))


@register(
    id="exp17",
    name="Memory-Augmented Chains",
    description="MACLA-style: higher preservation, still finite depth.",
    category="v6_architecture",
)
def run_experiment_17() -> dict:
    print("\n[Experiment 17] Memory-Augmented Chains (MACLA-style)")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_17", "skipped": True}


    budget_tokens = 40_000
    eff_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 8
    K = DEFAULT_K

    configs = {
        "No Memory": {"eta_bar": 0.85, "eta_underline": 0.65, "memory_boost": 0.0},
        "Moderate Memory": {"eta_bar": 0.90, "eta_underline": 0.70, "memory_boost": 0.05},
        "Large Memory": {"eta_bar": 0.95, "eta_underline": 0.75, "memory_boost": 0.10},
    }

    depths = list(range(1, max_depth + 1))
    results = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    for (name, cfg), color in zip(configs.items(), colors):
        chain = TheoreticalLLMChain(eta_bar=cfg["eta_bar"], eta_underline=cfg["eta_underline"])
        precisions = []
        profits = []
        for L in depths:
            budgets = [eff_budget] * L
            states = chain.precision_path(budgets, [K] * L)
            tau = states[-1].tau_post
            tau += cfg["memory_boost"] * tau * L
            precisions.append(tau)
            value = _value_fn(tau)
            cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets) + L * FIXED_COST
            profits.append(value - cost)

        ax.plot(depths, precisions, "o-", label=name, color=color)
        L_star = depths[np.argmax(profits)]
        results[name] = {
            "precisions": precisions,
            "profits": profits,
            "optimal_depth": L_star,
        }
        print(f"  {name:20s}: L* = {L_star}")

    ax.set_xlabel("Chain Depth $L$")
    ax.set_ylabel("Final Precision $\\tau^*_L$")
    ax.set_title("Memory-Augmented Chains: Higher Preservation, Still Finite Depth")
    ax.legend()
    ax.set_xticks(depths)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_memory_augmented.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_memory_augmented.png"}")

    return {
        "experiment_id": "exp17",
        "results": results,
    }
