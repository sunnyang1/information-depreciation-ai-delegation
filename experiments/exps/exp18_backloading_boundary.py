"""
Experiment 18: Back-Loading Boundary

Find parameter regions where back-loading outperforms front-loading.
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
from exp_framework import TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@register(
    id="exp18",
    name="Back-Loading Boundary",
    description="Convex late payoffs favor back-loading.",
    category="v6_frontloading",
)
def run_experiment_18() -> dict:
    print("\n[Experiment 18] Back-Loading Boundary")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_18", "skipped": True}


    budget_tokens = 30_000
    eff_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    L = 3
    K = DEFAULT_K
    chain = TheoreticalLLMChain()

    p_values = np.linspace(0.3, 3.0, 30)
    front_loaded_wins = []
    back_loaded_wins = []

    for p in p_values:
        budgets_front = [eff_budget * 1.5, eff_budget * 0.8, eff_budget * 0.7]
        states_front = chain.precision_path(budgets_front, [K] * L)
        tau_front = states_front[-1].tau_post

        budgets_back = [eff_budget * 0.7, eff_budget * 0.8, eff_budget * 1.5]
        states_back = chain.precision_path(budgets_back, [K] * L)
        tau_back = states_back[-1].tau_post

        c = 100.0
        V_front = (tau_front ** p) / (tau_front ** p + c ** p)
        V_back = (tau_back ** p) / (tau_back ** p + c ** p)

        front_loaded_wins.append(V_front)
        back_loaded_wins.append(V_back)

    front_loaded_wins = np.array(front_loaded_wins)
    back_loaded_wins = np.array(back_loaded_wins)

    backloading_better = back_loaded_wins > front_loaded_wins
    boundary_idx = np.where(backloading_better)[0]
    boundary_p = p_values[boundary_idx[0]] if len(boundary_idx) > 0 else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, front_loaded_wins, "b-", linewidth=2, label="Front-loaded")
    ax.plot(p_values, back_loaded_wins, "r-", linewidth=2, label="Back-loaded")
    if boundary_p is not None:
        ax.axvline(x=boundary_p, color="gray", linestyle="--", alpha=0.7,
                   label=f"Boundary $p \\approx {boundary_p:.2f}$")
    ax.set_xlabel("Late-Stage Convexity Parameter $p$")
    ax.set_ylabel("Final Value $V(\\tau^*_L)$")
    ax.set_title("Back-Loading Boundary: Convex Late Payoffs Favor Back-Loading")
    ax.legend()
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_backloading_boundary.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_backloading_boundary.png"}")
    if boundary_p is not None:
        print(f"  Boundary at p ≈ {boundary_p:.2f}")

    return {
        "experiment_id": "exp18",
        "p_values": p_values.tolist(),
        "front_loaded_value": front_loaded_wins.tolist(),
        "back_loaded_value": back_loaded_wins.tolist(),
        "boundary_p": float(boundary_p) if boundary_p is not None else None,
    }
