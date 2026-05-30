"""
Experiment 13: Rate-Distortion Simulation

Simulate rate-distortion for parallel Gaussian sources.
Show reverse water-filling correctness.
"""

from __future__ import annotations

from typing import Tuple

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
    id="exp13",
    name="Rate-Distortion Simulation",
    description="Reverse water-filling for parallel Gaussian sources.",
    category="v6_microfoundation",
)
def run_experiment_13() -> dict:
    print("\n[Experiment 13] Rate-Distortion Simulation")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_13", "skipped": True}


    d = 20
    eigenvalues = np.array([1.0 / (k ** 0.8) for k in range(1, d + 1)])
    eigenvalues = eigenvalues / eigenvalues.sum() * d

    def reverse_water_filling(eigvals: np.ndarray, R: float) -> Tuple[float, float]:
        sorted_eig = np.sort(eigvals)[::-1]
        best_theta = sorted_eig[-1]
        for m in range(1, len(sorted_eig) + 1):
            active = sorted_eig[:m]
            theta = (active.sum() - R) / m
            if theta >= 0 and (m == len(sorted_eig) or theta <= sorted_eig[m]):
                best_theta = theta
                break
        else:
            best_theta = max(0.0, (sorted_eig.sum() - R) / len(sorted_eig))
        total_distortion = np.sum(np.minimum(eigvals, best_theta))
        return total_distortion, best_theta

    R_max = eigenvalues.sum()
    R_values = np.linspace(0.1, R_max * 2, 100)
    distortions = []
    thetas = []
    active_modes_counts = []

    for R in R_values:
        D, theta = reverse_water_filling(eigenvalues, R)
        distortions.append(D)
        thetas.append(theta)
        active = np.sum(eigenvalues > theta)
        active_modes_counts.append(active)

    distortions = np.array(distortions)
    thetas = np.array(thetas)
    active_modes_counts = np.array(active_modes_counts)

    D_max = eigenvalues.sum()
    preservation_rates = 1.0 - distortions / D_max

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(R_values, distortions, "b-", linewidth=2, label="$D(R) = \\sum_k \\min\\{\\lambda_k, \\theta\\}$")
    naive_D = d * thetas
    ax.plot(R_values, naive_D, "r--", linewidth=1.5, alpha=0.7, label="Naive: $K \\cdot \\theta$ (incorrect)")
    ax.set_xlabel("Rate Budget $R$")
    ax.set_ylabel("Total Distortion $D(R)$")
    ax.set_title("(a) Reverse Water-Filling: Correct vs. Naive")
    ax.legend()
    ax.set_ylim(0, D_max * 1.1)

    ax = axes[1]
    ax.plot(R_values, preservation_rates, "g-", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Rate Budget $R$")
    ax.set_ylabel("Preservation Rate $\\eta(R) = 1 - D(R)/D_{\\max}$")
    ax.set_title("(b) Preservation Rate from Rate-Distortion")
    ax.set_ylim(0, 1.05)

    ax = axes[2]
    ax.plot(R_values, active_modes_counts, "m-", linewidth=2)
    ax.set_xlabel("Rate Budget $R$")
    ax.set_ylabel("Number of Active Modes")
    ax.set_title("(c) Active Modes vs. Rate Budget")
    ax.set_ylim(0, d + 1)

    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "fig_v6_rate_distortion.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "fig_v6_rate_distortion.png"}")

    return {
        "experiment_id": "exp13",
        "eigenvalues": eigenvalues.tolist(),
        "rate_values": R_values.tolist(),
        "distortions": distortions.tolist(),
        "preservation_rates": preservation_rates.tolist(),
        "active_modes": active_modes_counts.tolist(),
    }
