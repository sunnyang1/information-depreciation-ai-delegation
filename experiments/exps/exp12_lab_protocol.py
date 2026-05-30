"""
Experiment 12: Lab Experiment Protocol

Simulated lab experiment with three budget-allocation architectures.
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

from scipy import stats as spystats

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
    id="exp12",
    name="Lab Experiment Protocol",
    description="Simulated lab experiment: Front-loaded > Uniform > Back-loaded.",
    category="reviewer",
)
def run_experiment_12() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: Lab Experiment Protocol (Simulation)")
    if not HAS_MATPLOTLIB:
        print("  [SKIP] Matplotlib not installed, skipping figure generation.")
        return {"experiment_id": "run_experiment_12", "skipped": True}

    print("=" * 70)

    N_per_group = 40
    n_questions = 50
    depth = 3
    K = DEFAULT_K

    architectures = {
        "uniform": [8000, 8000, 8000],
        "front_loaded": [12000, 8000, 4000],
        "back_loaded": [4000, 8000, 12000],
    }

    results = {}

    for name, budgets_tokens in architectures.items():
        chain = TheoreticalLLMChain()
        budgets = [b / TOKENS_PER_ATTENTION_UNIT for b in budgets_tokens]
        states = chain.precision_path(budgets, [K] * depth)
        final_tau = states[-1].tau_post

        base_accuracy = chain.accuracy_from_precision(final_tau, add_noise=False)
        participant_scores = np.random.normal(base_accuracy, 0.05, N_per_group)
        participant_scores = np.clip(participant_scores, 0.0, 1.0)

        mean_f1 = float(np.mean(participant_scores))
        std_f1 = float(np.std(participant_scores))

        results[name] = {
            "architecture": budgets_tokens,
            "mean_f1": round(mean_f1, 4),
            "std_f1": round(std_f1, 4),
            "n_participants": N_per_group,
            "n_questions": n_questions,
            "final_tau": float(final_tau),
        }
        print(f"  {name:15s}: F1 = {mean_f1:.4f} (±{std_f1:.4f}), tau* = {final_tau:.2f}")

    effect_size = 0.5
    power = 0.80
    alpha = 0.05
    z_alpha = spystats.norm.ppf(1 - alpha / 2)
    z_beta = spystats.norm.ppf(power)
    n_required = int(2 * ((z_alpha + z_beta) / effect_size) ** 2) + 1

    print(f"\n  Power Analysis:")
    print(f"    Expected effect size d = {effect_size}")
    print(f"    Required N per group = {n_required} (actual = {N_per_group})")
    print(f"    Total N = {N_per_group * 3}")

    results["power_analysis"] = {
        "effect_size": effect_size,
        "required_n_per_group": n_required,
        "actual_n_per_group": N_per_group,
        "alpha": alpha,
        "power": power,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [k.replace("_", " ").title() for k in results.keys() if k != "power_analysis"]
    means = [results[k]["mean_f1"] for k in results.keys() if k != "power_analysis"]
    stds = [results[k]["std_f1"] for k in results.keys() if k != "power_analysis"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Lab Experiment: Budget Allocation Strategies ($N=40$ per group)")
    ax.set_ylim(0, 1)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    if HAS_MATPLOTLIB:
        fig.savefig(OUTPUT_DIR / "exp12_lab_experiment.png")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / "exp12_lab_experiment.png"}")

    return {
        "experiment_id": "exp12",
        "results": results,
    }
