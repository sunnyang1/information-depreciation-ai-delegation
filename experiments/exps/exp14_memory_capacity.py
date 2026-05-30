"""
Experiment 14: Memory Capacity vs. Effective Preservation

Show that external memory raises effective eta_bar but cannot eliminate eta_bar < 1.
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

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "figure.figsize": (8, 5), "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@register(
    id="exp14",
    name="Memory Capacity vs. Effective Preservation",
    description="Memory raises preservation but cannot eliminate the bound.",
    category="v6_microfoundation",
)
def run_experiment_14() -> dict:
    print("\n[Experiment 14] Memory Capacity vs. Effective Preservation")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_14", "skipped": True}


    M_values = np.linspace(0, 100, 100)
    eta_bar_base = 0.85
    lambdas = [0.01, 0.03, 0.05]

    fig, ax = plt.subplots(figsize=(8, 5))
    results = {"base_eta_bar": eta_bar_base, "curves": []}

    for lam in lambdas:
        eta_eff = 1.0 - (1.0 - eta_bar_base) * np.exp(-lam * M_values)
        ax.plot(M_values, eta_eff, linewidth=2, label=f"$\\lambda = {lam:.2f}$")
        results["curves"].append({
            "lambda": lam,
            "eta_eff": eta_eff.tolist(),
        })

    ax.axhline(y=eta_bar_base, color="gray", linestyle="--", alpha=0.5,
               label=f"Base $\\bar{{\\eta}} = {eta_bar_base}$ (no memory)")
    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.5,
               label="Perfect preservation (unattainable)")
    ax.set_xlabel("External Memory Capacity $M$")
    ax.set_ylabel("Effective Preservation Bound $\\bar{\\eta}_{\\text{eff}}(M)$")
    ax.set_title("Memory Raises Preservation but Cannot Eliminate the Bound")
    ax.legend(loc="lower right")
    ax.set_ylim(0.75, 1.02)
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_memory_capacity.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_memory_capacity.png"}")

    return {
        "experiment_id": "exp14",
        "results": results,
    }
