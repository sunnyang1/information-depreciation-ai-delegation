"""
Experiment 10: Human-AI Hybrid Chains

Optimal depth of human-AI chains is U-shaped in context-window size.
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
    id="exp10",
    name="Human-AI Hybrid Chains",
    description="Human-AI hybrid chain optimal depth vs pure AI.",
    category="reviewer",
)
def run_experiment_10() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Human-AI Hybrid Chains")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_10", "skipped": True}

    print("=" * 70)

    def human_layer_precision(ai_output_tau: float, context_window: int) -> float:
        human_capacity = 4000
        comprehension_rate = min(1.0, human_capacity / max(context_window, 1))
        human_noise = 0.15
        return ai_output_tau * comprehension_rate * (1.0 - human_noise)

    context_windows = [2048, 4096, 8192, 16384, 32768, 65536, 128000]
    K = DEFAULT_K

    pure_ai_depths = []
    hybrid_depths = []

    for ctx in context_windows:
        effective_budget = ctx / TOKENS_PER_ATTENTION_UNIT
        chain = TheoreticalLLMChain()

        profits_ai = []
        for L in range(1, 8):
            budgets = [effective_budget] * L
            states = chain.precision_path(budgets, [K] * L)
            tau = states[-1].tau_post
            value = PSI * (1.0 - math.exp(-tau / 200.0))
            cost = L * 0.02
            profits_ai.append(value - cost)
        L_star_ai = int(np.argmax(profits_ai) + 1)
        pure_ai_depths.append(L_star_ai)

        states_hybrid = chain.precision_path([effective_budget, effective_budget], [K, K])
        tau_after_ai = states_hybrid[-1].tau_post
        tau_after_human = human_layer_precision(tau_after_ai, ctx)

        profits_hybrid = []
        for L_ai_after in range(0, 5):
            tau = tau_after_human
            for _ in range(L_ai_after):
                budgets = [effective_budget]
                states = chain.precision_path(budgets, [K])
                tau = states[-1].tau_post
            value = PSI * (1.0 - math.exp(-tau / 200.0))
            cost = (2 + L_ai_after) * 0.02
            profits_hybrid.append(value - cost)
        L_star_hybrid = int(np.argmax(profits_hybrid) + 1) + 1
        hybrid_depths.append(L_star_hybrid)

    results = {
        "context_windows": context_windows,
        "pure_ai_optimal_depth": pure_ai_depths,
        "hybrid_optimal_depth": hybrid_depths,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(context_windows) / 1000
    ax.plot(x, pure_ai_depths, "o-", label="Pure AI Chain", color="#1f77b4")
    ax.plot(x, hybrid_depths, "s-", label="Human-AI Hybrid", color="#ff7f0e")
    ax.set_xlabel("Context Window $A$ (thousands of tokens)")
    ax.set_ylabel("Optimal Depth $L^*$")
    ax.set_title("Optimal Depth: Pure AI vs. Human-AI Hybrid")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "exp10_human_ai_hybrid.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "exp10_human_ai_hybrid.png"}")

    print(f"\n  Context Window | Pure-AI L* | Hybrid L*")
    for ctx, lai, lhy in zip(context_windows, pure_ai_depths, hybrid_depths):
        print(f"  {ctx:10,}      | {lai:10d} | {lhy:9d}")

    return {
        "experiment_id": "exp10",
        "results": results,
    }
