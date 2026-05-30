"""
Experiment 8: Task Complexity and Preservation Rate

Information depreciation varies with task complexity.
Simple tasks: eta_bar close to 1. Complex tasks: eta_bar drops sharply.
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
    TheoreticalLLMChain,
    TOKENS_PER_ATTENTION_UNIT,
    DEFAULT_K,
    KAPPA,
    PHI,
    PSI,
)

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@register(
    id="exp08",
    name="Task Complexity and Preservation Rate",
    description="Complex tasks reduce eta_bar and limit optimal depth.",
    category="reviewer",
)
def run_experiment_8() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Task Complexity and Preservation Rate")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_8", "skipped": True}

    print("=" * 70)

    complexities = np.linspace(0.0, 1.0, 6)
    budget_tokens = 50_000
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 12
    K = DEFAULT_K
    fixed_cost_per_layer = 0.02

    results = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    for t in complexities:
        eta_bar_t = max(0.60, 0.99 - 0.25 * t)
        eta_underline_t = max(0.50, eta_bar_t - 0.20)
        chain = TheoreticalLLMChain(eta_bar=eta_bar_t, eta_underline=eta_underline_t)

        retentions = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            states = chain.precision_path(budgets, [K] * L)
            cum_rho = 1.0
            for s in states:
                cum_rho *= s.rho
            retentions.append(cum_rho)

        label = f"t={t:.1f} (eta_bar={eta_bar_t:.2f})"
        ax.plot(range(1, max_depth + 1), retentions, "o-", label=label, markersize=4)
        results[f"t_{t:.1f}"] = {
            "complexity": float(t),
            "eta_bar": float(eta_bar_t),
            "eta_underline": float(eta_underline_t),
            "retention_path": [float(x) for x in retentions],
        }

    ax.set_xlabel("Chain Depth $L$")
    ax.set_ylabel("Normalized Precision $\\tau^*_L / \\tau^*_1$")
    ax.set_title("Task Complexity and Information Depreciation")
    ax.legend(loc="upper right", title="Task Complexity")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "exp8_task_complexity.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "exp8_task_complexity.png"}")

    print("\n  Optimal Depth by Task Complexity:")
    for t in complexities:
        eta_bar_t = max(0.60, 0.99 - 0.25 * t)
        eta_underline_t = max(0.50, eta_bar_t - 0.20)
        chain = TheoreticalLLMChain(eta_bar=eta_bar_t, eta_underline=eta_underline_t)
        profits = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            states = chain.precision_path(budgets, [K] * L)
            tau = states[-1].tau_post
            value = PSI * (1.0 - math.exp(-tau / 200.0))
            cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets) + L * fixed_cost_per_layer
            profits.append(value - cost)
        L_star = int(np.argmax(profits) + 1)
        results[f"t_{t:.1f}"]["optimal_depth"] = L_star
        print(f"    t={t:.1f}: L* = {L_star}")

    return {
        "experiment_id": "exp08",
        "results": results,
    }
