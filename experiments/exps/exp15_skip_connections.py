"""
Experiment 15: Skip Connections vs. Serial Chains

Compare serial chain vs. skip-connection architecture.
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
    id="exp15",
    name="Skip Connections vs. Serial Chains",
    description="Skip connections raise precision but preserve finite-depth logic.",
    category="v6_architecture",
)
def run_experiment_15() -> dict:
    print("\n[Experiment 15] Skip Connections vs. Serial Chains")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_15", "skipped": True}


    budget_tokens = 40_000
    eff_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 8
    K = DEFAULT_K

    chain_serial = TheoreticalLLMChain()
    chain_skip = TheoreticalLLMChain(eta_bar=0.95, eta_underline=0.80)

    depths = list(range(1, max_depth + 1))

    serial_precisions = []
    for L in depths:
        budgets = [eff_budget] * L
        states = chain_serial.precision_path(budgets, [K] * L)
        serial_precisions.append(states[-1].tau_post)

    skip_precisions = []
    alpha_skip = 0.3
    for L in depths:
        budgets = [eff_budget] * L
        states_serial = chain_serial.precision_path(budgets, [K] * L)
        tau_skip = states_serial[0].tau_post * (chain_skip.compute_eta(eff_budget, K) ** (L - 1))
        tau_eff = states_serial[-1].tau_post + alpha_skip * tau_skip
        skip_precisions.append(tau_eff)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, serial_precisions, "o-", label="Serial Chain", color="#1f77b4")
    ax.plot(depths, skip_precisions, "s-", label="Skip Connections", color="#ff7f0e")
    ax.set_xlabel("Chain Depth $L$")
    ax.set_ylabel("Final Precision $\\tau^*_L$")
    ax.set_title("Skip Connections Raise Precision but Preserve Finite-Depth Logic")
    ax.legend()
    ax.set_xticks(depths)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_skip_connections.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_skip_connections.png"}")

    return {
        "experiment_id": "exp15",
        "depths": depths,
        "serial_precision": serial_precisions,
        "skip_precision": skip_precisions,
    }
