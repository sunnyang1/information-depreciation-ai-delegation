"""
Supplementary Figure 3: Profit Function Pi(L) and Interior Optimum L*
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
    TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K, KAPPA, PHI, PSI,
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
    return PSI * (1.0 - math.exp(-tau / 200.0))


@register(
    id="sup03",
    name="Profit Function and Interior Optimum",
    description="Profit Pi(L) = V(tau*_L) - sum c(A_l) with interior optimum.",
    category="supplementary",
)
def run_supplementary_03() -> dict:
    print("\n[Supplementary 03] Profit Function and Interior Optimum")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_supplementary_03", "skipped": True}


    chain = TheoreticalLLMChain()
    budget_tokens = 20_000
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 12

    depths = list(range(1, max_depth + 1))
    profits = []

    for L in depths:
        budgets = [effective_budget] * L
        K_list = [DEFAULT_K] * L
        states = chain.precision_path(budgets, K_list)
        final_tau = states[-1].tau_post if states else 0.0

        value = _value_fn(final_tau)
        variable_cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets)
        cost = variable_cost + L * FIXED_COST_PER_LAYER
        profit = value - cost
        profits.append(profit)

    L_star = depths[np.argmax(profits)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, profits, "o-", color="#1f77b4", label="Profit $\\Pi(L)$")
    ax.axvline(x=L_star, color="red", linestyle="--",
               label=f"Optimum $L^* = {L_star}$")
    ax.set_xlabel("Chain Depth $L$")
    ax.set_ylabel("Profit $\\Pi(L)$")
    ax.set_title("Profit Function and Interior Optimum")
    ax.legend()
    ax.set_xticks(depths)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig3_profit_function.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig3_profit_function.png"}")
    print(f"  Optimal depth L* = {L_star}")

    return {
        "experiment_id": "sup03",
        "depths": depths,
        "profits": profits,
        "optimal_depth": L_star,
    }
