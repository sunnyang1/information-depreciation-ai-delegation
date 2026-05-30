"""
Experiment 9: Heterogeneous Agent Types

Compare chains with Transformer, Rule-based, and Retrieval agents.
Prediction: Optimal placement is Transformer early, Rule-based late.
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
from exp_framework import TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K, DEFAULT_TAU0

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@register(
    id="exp09",
    name="Heterogeneous Agent Types",
    description="Optimal placement varies by agent type preservation rate.",
    category="reviewer",
)
def run_experiment_9() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Heterogeneous Agent Types")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_9", "skipped": True}

    print("=" * 70)

    budget_tokens = 40_000
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 5
    K = DEFAULT_K

    agent_configs = {
        "transformer": {"eta_bar": 0.95, "eta_underline": 0.75, "label": "Transformer"},
        "rule_based": {"eta_bar": 0.999, "eta_underline": 0.99, "label": "Rule-based"},
        "retrieval": {"eta_bar": 0.85, "eta_underline": 0.65, "label": "Retrieval"},
    }

    architectures = {
        "All Transformer": ["transformer"] * max_depth,
        "All Rule-based": ["rule_based"] * max_depth,
        "Mixed (Optimal)": ["transformer", "transformer", "retrieval", "rule_based", "rule_based"],
        "Reversed": ["rule_based", "rule_based", "retrieval", "transformer", "transformer"],
    }

    results = {}
    accuracies = {}

    for arch_name, agent_types in architectures.items():
        # Layer-by-layer recursion with heterogeneous agent types
        tau_prior = 0.0
        cum_depreciation = 1.0
        for l, agent_type in enumerate(agent_types):
            cfg = agent_configs[agent_type]
            chain = TheoreticalLLMChain(eta_bar=cfg["eta_bar"], eta_underline=cfg["eta_underline"])
            A = effective_budget
            eta = chain.compute_eta(A, K)
            cum_depreciation *= eta
            tau0_list = [DEFAULT_TAU0 * cum_depreciation] * K
            _, _, tau_fresh = chain.solve_attention_allocation(A, tau0_list)
            tau_post = tau_prior + tau_fresh
            tau_fb = tau_prior + sum(tau0_list)
            rho = tau_post / tau_fb if tau_fb > 0 else 1.0
            tau_prior = rho * tau_post

        final_tau = tau_prior
        accuracy = chain.accuracy_from_precision(final_tau, add_noise=False)
        results[arch_name] = {
            "architecture": agent_types,
            "final_tau": float(final_tau),
            "accuracy": float(accuracy),
        }
        accuracies[arch_name] = accuracy
        print(f"  {arch_name:25s}: Final tau = {final_tau:.2f}, Accuracy = {accuracy:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(accuracies.keys())
    vals = list(accuracies.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("End-to-End Accuracy")
    ax.set_title("Heterogeneous Agent Architectures ($L=5$)")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "exp9_heterogeneous_agents.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "exp9_heterogeneous_agents.png"}")

    return {
        "experiment_id": "exp09",
        "results": results,
    }
