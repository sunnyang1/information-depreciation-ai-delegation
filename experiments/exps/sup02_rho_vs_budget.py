"""
Supplementary Figure 2: Endogenous Transmission Factor rho vs. Attention Budget A
"""

from __future__ import annotations

from typing import List

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
from exp_framework import TheoreticalLLMChain

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _pareto_precisions(K: int, alpha: float) -> List[float]:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(K)]
    total = sum(weights)
    return [w / total * K for w in weights]


@register(
    id="sup02",
    name="Transmission Factor vs. Attention Budget",
    description="Rho vs budget for different heterogeneity profiles.",
    category="supplementary",
)
def run_supplementary_02() -> dict:
    print("\n[Supplementary 02] Endogenous Transmission Factor vs. Attention Budget")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_supplementary_02", "skipped": True}


    budgets = np.logspace(2, 5, 50)
    chain = TheoreticalLLMChain()

    profiles = {
        "High heterogeneity (Pareto)": lambda K: _pareto_precisions(K, alpha=1.5),
        "Moderate heterogeneity": lambda K: _pareto_precisions(K, alpha=0.8),
        "Homogeneous": lambda K: [1.0] * K,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    all_results = {}

    for (name, prec_fn), color in zip(profiles.items(), colors):
        K = 50
        precisions = prec_fn(K)
        rhos = []
        for A in budgets:
            _, _, tau_star = chain.solve_attention_allocation(A, precisions)
            tau_fb = sum(precisions)
            rho = tau_star / tau_fb if tau_fb > 0 else 1.0
            rhos.append(min(rho, 1.0))
        ax.plot(budgets, rhos, color=color, label=name, linewidth=2)
        all_results[name] = {
            "budgets": budgets.tolist(),
            "rhos": rhos,
        }

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Attention Budget $A_\\ell$ (attention units)")
    ax.set_ylabel("Transmission Factor $\\rho_\\ell$")
    ax.set_title("Endogenous Transmission Factor vs. Attention Budget")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig2_rho_vs_budget.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig2_rho_vs_budget.png"}")

    return {
        "experiment_id": "sup02",
        "results": all_results,
    }
