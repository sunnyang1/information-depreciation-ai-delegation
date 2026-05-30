"""
Supplementary Figure 5: Information Retention Rate and Marginal Loss vs. Chain Depth
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
    id="sup05",
    name="Retention Rate and Marginal Loss vs. Depth",
    description="Cumulative retention and marginal loss as functions of depth.",
    category="supplementary",
)
def run_supplementary_05() -> dict:
    print("\n[Supplementary 05] Information Retention and Marginal Loss vs. Depth")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_supplementary_05", "skipped": True}


    chain = TheoreticalLLMChain()
    budget_tokens = 128_000
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 10

    budgets = [effective_budget] * max_depth
    K_list = [DEFAULT_K] * max_depth
    states = chain.precision_path(budgets, K_list)

    taus = [s.tau_post for s in states]
    retention = np.array(taus) / taus[0] if taus[0] > 0 else np.ones(len(taus))
    marginal_loss = np.diff(-retention, prepend=1.0)

    depths = np.arange(1, max_depth + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(depths, retention, "o-", color="#1f77b4")
    ax1.set_xlabel("Chain Depth $L$")
    ax1.set_ylabel("Cumulative Retention Rate")
    ax1.set_title("Information Retention Rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(depths)

    ax2.bar(depths, marginal_loss, color="#d62728", alpha=0.7)
    ax2.set_xlabel("Chain Depth $L$")
    ax2.set_ylabel("Marginal Loss")
    ax2.set_title("Marginal Information Loss per Layer")
    ax2.set_xticks(depths)

    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig5_retention_marginal_loss.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig5_retention_marginal_loss.png"}")

    return {
        "experiment_id": "sup05",
        "depths": depths.tolist(),
        "retention": retention.tolist(),
        "marginal_loss": marginal_loss.tolist(),
    }
