"""
Supplementary Figure 1: Information Precision Decay Across Hierarchy Levels
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
    id="sup01",
    name="Precision Decay Across Hierarchy Levels",
    description="Normalized effective precision vs layer depth for three eta_bar values.",
    category="supplementary",
)
def run_supplementary_01() -> dict:
    print("\n[Supplementary 01] Information Precision Decay Across Hierarchy Levels")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_supplementary_01", "skipped": True}


    depths = np.arange(0, 11)
    eta_bars = [0.95, 0.85, 0.70]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    labels = [
        r"$\bar{\eta}=0.95$ (High-fidelity)",
        r"$\bar{\eta}=0.85$ (Medium)",
        r"$\bar{\eta}=0.70$ (Low-fidelity)",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    all_results = {}

    for eta_bar, color, label in zip(eta_bars, colors, labels):
        chain = TheoreticalLLMChain(eta_bar=eta_bar, eta_underline=0.60)
        budget_tokens = 50_000
        effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
        budgets = [effective_budget] * 10
        K_list = [DEFAULT_K] * 10

        states = chain.precision_path(budgets, K_list)
        cum_rho = 1.0
        retentions = [1.0]
        for s in states:
            cum_rho *= s.rho
            retentions.append(cum_rho)

        ax.plot(depths[:len(retentions)], retentions, "o-", color=color,
                label=label, markersize=5)
        all_results[label] = {
            "eta_bar": eta_bar,
            "retentions": [float(x) for x in retentions],
        }

    ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=1,
               label="Effective precision threshold")
    ax.set_xlabel("Hierarchy Depth $\\ell$")
    ax.set_ylabel("Normalized Effective Precision $\\tau^*_\\ell / \\tau^*_0$")
    ax.set_title("Information Precision Decay Across Hierarchy Levels")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(depths)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig1_precision_decay.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig1_precision_decay.png"}")

    return {
        "experiment_id": "sup01",
        "results": all_results,
    }
