"""
Experiment 16: Parallel Branches (Trees vs. Chains)

Compare chain vs. tree architecture with parallel branches.
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
from exp_framework import TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@register(
    id="exp16",
    name="Parallel Branches (Trees vs. Chains)",
    description="Parallel branches improve precision but still exhibit decay.",
    category="v6_architecture",
)
def run_experiment_16() -> dict:
    print("\n[Experiment 16] Parallel Branches (Trees vs. Chains)")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_16", "skipped": True}


    budget_tokens = 40_000
    total_K = 100
    max_depth = 5
    chain = TheoreticalLLMChain()
    eff_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT

    chain_precisions = []
    for L in range(1, max_depth + 1):
        budgets = [eff_budget] * L
        states = chain.precision_path(budgets, [total_K] * L)
        chain_precisions.append(states[-1].tau_post)

    tree_precisions = {}
    for M in [2, 4]:
        K_per_branch = total_K // M
        budget_per_branch = eff_budget / M
        precisions = []
        for L in range(1, max_depth + 1):
            states = chain.precision_path([budget_per_branch] * L, [K_per_branch] * L)
            tau_branch = states[-1].tau_post
            tau_agg = M * tau_branch
            precisions.append(tau_agg)
        tree_precisions[M] = precisions

    fig, ax = plt.subplots(figsize=(8, 5))
    depths = list(range(1, max_depth + 1))
    ax.plot(depths, chain_precisions, "o-", label="Chain (M=1)", color="#1f77b4")
    for M, precisions in tree_precisions.items():
        ax.plot(depths, precisions, "s-", label=f"Tree (M={M} branches)", alpha=0.8)

    ax.set_xlabel("Chain Depth $L$")
    ax.set_ylabel("Aggregate Precision $\\tau^*_L$")
    ax.set_title("Parallel Branches Improve Precision but Still Exhibit Decay")
    ax.legend()
    ax.set_xticks(depths)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_parallel_branches.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_parallel_branches.png"}")

    return {
        "experiment_id": "exp16",
        "depths": depths,
        "chain_precision": chain_precisions,
        "tree_precisions": {str(k): v for k, v in tree_precisions.items()},
    }
