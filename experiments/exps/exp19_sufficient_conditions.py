"""
Experiment 19: Sufficient Conditions for Front-Loading

Visualize sufficient conditions: front-loading optimal when V''(tau) <= 0.
"""

from __future__ import annotations

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
from exp_framework import TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K, PSI

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@register(
    id="exp19",
    name="Sufficient Conditions for Front-Loading",
    description="Front-loading dominance under concave value.",
    category="v6_frontloading",
)
def run_experiment_19() -> dict:
    print("\n[Experiment 19] Sufficient Conditions for Front-Loading")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_19", "skipped": True}


    budget_tokens = 40_000
    eff_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    L = 3
    K = DEFAULT_K
    chain = TheoreticalLLMChain()

    total_budgets = np.linspace(eff_budget * 2, eff_budget * 6, 20)

    front_loaded_vals = []
    uniform_vals = []
    back_loaded_vals = []

    for B_total in total_budgets:
        r = 0.6
        budgets_front = [B_total * (1 - r) / (1 - r**L) * r**i for i in range(L)]
        states_front = chain.precision_path(budgets_front, [K] * L)
        tau_front = states_front[-1].tau_post

        budgets_uniform = [B_total / L] * L
        states_uniform = chain.precision_path(budgets_uniform, [K] * L)
        tau_uniform = states_uniform[-1].tau_post

        budgets_back = list(reversed(budgets_front))
        states_back = chain.precision_path(budgets_back, [K] * L)
        tau_back = states_back[-1].tau_post

        V_front = -PSI / max(tau_front, 1.0)
        V_uniform = -PSI / max(tau_uniform, 1.0)
        V_back = -PSI / max(tau_back, 1.0)

        front_loaded_vals.append(V_front)
        uniform_vals.append(V_uniform)
        back_loaded_vals.append(V_back)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(total_budgets, front_loaded_vals, "b-o", label="Front-loaded", markersize=4)
    ax.plot(total_budgets, uniform_vals, "g-s", label="Uniform", markersize=4)
    ax.plot(total_budgets, back_loaded_vals, "r-^", label="Back-loaded", markersize=4)
    ax.set_xlabel("Total Budget $B$")
    ax.set_ylabel("Value $V(\\tau^*_L) = -\\psi / \\tau$")
    ax.set_title("Front-Loading Dominance Under Concave Value (Quadratic Loss)")
    ax.legend()
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_sufficient_conditions.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_sufficient_conditions.png"}")

    return {
        "experiment_id": "exp19",
        "total_budgets": total_budgets.tolist(),
        "front_loaded": front_loaded_vals,
        "uniform": uniform_vals,
        "back_loaded": back_loaded_vals,
    }
