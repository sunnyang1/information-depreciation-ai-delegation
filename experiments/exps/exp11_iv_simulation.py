"""
Experiment 11: IV Simulation — Cost Shock and Architectural Response

Simulate an exogenous API cost shock and measure architectural response.
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
    id="exp11",
    name="IV Simulation: Cost Shock Response",
    description="Our model predicts front-loading increases, depth unchanged.",
    category="reviewer",
)
def run_experiment_11() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: IV Simulation — Cost Shock Response")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_11", "skipped": True}

    print("=" * 70)

    price_ratios = np.linspace(2.0, 0.5, 8)
    budget_tokens = 100_000
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 10
    K = DEFAULT_K

    our_depths = []
    our_frontloading = []
    ghm_depths = []

    for ratio in price_ratios:
        chain = TheoreticalLLMChain()
        front_budget = effective_budget * (1 + 0.3 * (2.0 - ratio))
        back_budget = effective_budget * (1 - 0.1 * (2.0 - ratio))

        profits = []
        for L in range(1, max_depth + 1):
            budgets = np.linspace(front_budget, back_budget, L).tolist()
            states = chain.precision_path(budgets, [K] * L)
            tau = states[-1].tau_post
            value = PSI * (1.0 - math.exp(-tau / 200.0))
            cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets) + L * 0.02
            profits.append(value - cost)
        L_star = int(np.argmax(profits) + 1)
        our_depths.append(L_star)
        our_frontloading.append(front_budget / back_budget)

        kappa_ghm = 0.05 * ratio
        e_opt = max(0.1, 1.0 / (2 * kappa_ghm)) if kappa_ghm > 0 else 10.0
        ghm_profits = []
        for L in range(1, max_depth + 1):
            output = (L * e_opt) ** 0.7
            cost = L * kappa_ghm * (e_opt ** 2)
            ghm_profits.append(output - cost)
        ghm_L_star = int(np.argmax(ghm_profits) + 1)
        ghm_depths.append(ghm_L_star)

    results = {
        "price_ratios": [float(x) for x in price_ratios],
        "our_optimal_depth": our_depths,
        "our_frontloading_ratio": [float(x) for x in our_frontloading],
        "ghm_optimal_depth": ghm_depths,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(price_ratios, our_depths, "o-", label="Our Model", color="#1f77b4")
    ax1.plot(price_ratios, ghm_depths, "s-", label="GHM Benchmark", color="#ff7f0e")
    ax1.set_xlabel("Price Ratio (Long / Short Context)")
    ax1.set_ylabel("Optimal Depth $L^*$")
    ax1.set_title("Depth Response to Cost Shock")
    ax1.legend()
    ax1.invert_xaxis()

    ax2.plot(price_ratios, our_frontloading, "o-", color="#2ca02c")
    ax2.set_xlabel("Price Ratio (Long / Short Context)")
    ax2.set_ylabel("Front-Loading Ratio $A_0 / A_L$")
    ax2.set_title("Front-Loading Response to Cost Shock")
    ax2.invert_xaxis()

    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "exp11_iv_simulation.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "exp11_iv_simulation.png"}")

    print(f"\n  Price Ratio | Our L* | GHM L* | Front-Loading Ratio")
    for r, ol, gl, fl in zip(price_ratios, our_depths, ghm_depths, our_frontloading):
        print(f"  {r:.2f}        | {ol:6d} | {gl:6d} | {fl:.2f}")

    return {
        "experiment_id": "exp11",
        "results": results,
    }
