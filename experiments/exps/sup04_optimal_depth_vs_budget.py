"""
Supplementary Figure 4: Optimal Chain Depth L* vs. Attention Budget A
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
from exp_framework import (
    TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K, KAPPA, PHI,
)

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_COST_PER_LAYER = 0.02


def _value_fn(tau: float) -> float:
    return 1.0 * (1.0 - math.exp(-tau / 200.0))


@register(
    id="sup04",
    name="Optimal Depth vs. Attention Budget",
    description="L* increases with context window size.",
    category="supplementary",
)
def run_supplementary_04() -> dict:
    print("\n[Supplementary 04] Optimal Chain Depth vs. Attention Budget")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_supplementary_04", "skipped": True}


    chain = TheoreticalLLMChain()
    budget_list = np.linspace(4_096, 512_000, 30)
    max_depth = 12

    L_stars = []
    for budget_tokens in budget_list:
        effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
        profits = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            K_list = [DEFAULT_K] * L
            states = chain.precision_path(budgets, K_list)
            final_tau = states[-1].tau_post if states else 0.0
            value = _value_fn(final_tau)
            variable_cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets)
            cost = variable_cost + L * FIXED_COST_PER_LAYER
            profits.append(value - cost)
        L_star = int(np.argmax(profits) + 1)
        L_stars.append(L_star)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budget_list / 1000, L_stars, "o-", color="#1f77b4")
    ax.axvline(x=32, color="gray", linestyle=":", alpha=0.7, label="GPT-3.5 (32K)")
    ax.axvline(x=128, color="orange", linestyle=":", alpha=0.7, label="GPT-4 (128K)")
    ax.set_xlabel("Context Window $A$ (thousands of tokens)")
    ax.set_ylabel("Optimal Depth $L^*$")
    ax.set_title("Optimal Chain Depth vs. Attention Budget")
    ax.legend()
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig4_optimal_depth_vs_budget.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig4_optimal_depth_vs_budget.png"}")

    return {
        "experiment_id": "sup04",
        "budgets_tokens": budget_list.tolist(),
        "optimal_depths": L_stars,
    }
