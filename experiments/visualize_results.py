"""
Visualization script for experimental results.

Generates publication-quality figures for all 7 experiments:
  1. Depth-Accuracy Tradeoff
  2. Front-Loading Comparison
  3. Exponential Decay of Retention
  4. Signal Overload (rho vs K)
  5. Heterogeneity and Distortion
  6. Cost Irrelevance for Depth
  7. Budget Expansion and Optimal Depth

Usage:
    python visualize_results.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Paper-style matplotlib settings
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def plot_experiment_1(data: Dict):
    """Plot Depth-Accuracy Tradeoff."""
    results = data.get("experiment_1", [])
    if not results:
        return

    depths = [r["depth"] for r in results]
    accs = [r["accuracy"] for r in results]
    stds = [r["metadata"]["accuracy_std"] for r in results]
    taus = [r["metadata"]["final_tau"] for r in results]

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Chain Depth $L$")
    ax1.set_ylabel("Accuracy", color=color)
    ax1.plot(depths, accs, "o-", color=color, label="Accuracy")
    ax1.fill_between(depths, np.array(accs) - np.array(stds), np.array(accs) + np.array(stds), alpha=0.2, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Final Precision $\\tau^*_L$", color=color)
    ax2.plot(depths, taus, "s--", color=color, label="Precision")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp1_depth_accuracy.png")
    plt.close(fig)
    print("Saved: exp1_depth_accuracy.png")


def plot_experiment_2(data: Dict):
    """Plot Front-Loading Comparison."""
    results = data.get("experiment_2", {})
    if not results:
        return

    strategies = list(results.keys())
    accs = [results[s]["accuracy"] for s in strategies]
    stds = [results[s]["metadata"]["accuracy_std"] for s in strategies]
    labels = [s.replace("_", " ").title() for s in strategies]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, accs, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Experiment 2: Budget Allocation Strategies ($L=3$)")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.3f}", ha="center", va="bottom")
    fig.savefig(OUTPUT_DIR / "exp2_front_loading.png")
    plt.close(fig)
    print("Saved: exp2_front_loading.png")


def plot_experiment_3(data: Dict):
    """Plot Exponential Decay of Retention."""
    results = data.get("experiment_3", [])
    if not results:
        return

    layers = [e["layer"] for e in results]
    retention = [e["retention_rate"] for e in results]
    ci_lower = [e["ci_lower"] for e in results]
    ci_upper = [e["ci_upper"] for e in results]
    theoretical = [e["theoretical"] for e in results]

    fig, ax = plt.subplots()
    ax.plot(layers, retention, "o-", label="Observed", color="tab:blue")
    ax.fill_between(layers, ci_lower, ci_upper, alpha=0.2, color="tab:blue", label="95% CI")
    ax.plot(layers, theoretical, "s--", label="Theoretical $\\bar{\\eta}^\\ell$", color="tab:red")
    ax.set_xlabel("Layer $\\ell$")
    ax.set_ylabel("Retention Rate")
    ax.set_title("Experiment 3: Per-Layer Information Retention")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig(OUTPUT_DIR / "exp3_eta_decay.png")
    plt.close(fig)
    print("Saved: exp3_eta_decay.png")


def plot_experiment_4(data: Dict):
    """Plot Signal Overload."""
    results = data.get("experiment_4_signal_overload", {})
    if not results:
        return

    fig, ax = plt.subplots()
    for key, sub in results.items():
        Ks = sub["n_signals"]
        rhos = sub["rhos"]
        ax.plot(Ks, rhos, "o-", label=key.replace("depth_", "$L=") + "$")

    ax.set_xlabel("Number of Signals $K$")
    ax.set_ylabel("Transmission Factor $\\rho$")
    ax.set_title("Experiment 4: Signal Overload")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig(OUTPUT_DIR / "exp4_signal_overload.png")
    plt.close(fig)
    print("Saved: exp4_signal_overload.png")


def plot_experiment_5(data: Dict):
    """Plot Heterogeneity and Distortion."""
    results = data.get("experiment_5_heterogeneity", {})
    if not results:
        return

    labels = [k.replace("_", " ").title() for k in results.keys()]
    deltas = [v["distortion_delta"] for v in results.values()]
    rhos = [v["rho"] for v in results.values()]
    ginis = [v["gini"] for v in results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(labels, deltas, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.set_ylabel("Aggregate Distortion $\\Delta$")
    ax1.set_title("Distortion vs Heterogeneity")
    ax1.set_ylim(0, max(deltas) * 1.2)

    ax2.bar(labels, rhos, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax2.set_ylabel("Transmission Factor $\\rho$")
    ax2.set_title("Transmission Factor vs Heterogeneity")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp5_heterogeneity.png")
    plt.close(fig)
    print("Saved: exp5_heterogeneity.png")


def plot_experiment_6(data: Dict):
    """Plot Cost Irrelevance for Depth."""
    results = data.get("experiment_6_cost_irrelevance", {})
    if not results:
        return

    kappas = [v["kappa"] for v in results.values()]
    L_stars = [v["optimal_depth"] for v in results.values()]

    fig, ax = plt.subplots()
    ax.plot(kappas, L_stars, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Cost Parameter $\\kappa$")
    ax.set_ylabel("Optimal Depth $L^*$")
    ax.set_title("Experiment 6: Cost Irrelevance for Depth")
    ax.axhline(y=L_stars[-1], color="red", linestyle="--", label=f"Limit = {L_stars[-1]}")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "exp6_cost_irrelevance.png")
    plt.close(fig)
    print("Saved: exp6_cost_irrelevance.png")


def plot_experiment_7(data: Dict):
    """Plot Budget Expansion and Optimal Depth."""
    results = data.get("experiment_7_budget_depth", {})
    if not results:
        return

    budgets = [v["budget_tokens"] / 1000 for v in results.values()]  # in thousands
    L_stars = [v["optimal_depth"] for v in results.values()]

    fig, ax = plt.subplots()
    ax.plot(budgets, L_stars, "o-")
    ax.set_xlabel("Context Window $A$ (thousands of tokens)")
    ax.set_ylabel("Optimal Depth $L^*$")
    ax.set_title("Experiment 7: Budget Expansion Increases Depth")
    fig.savefig(OUTPUT_DIR / "exp7_budget_depth.png")
    plt.close(fig)
    print("Saved: exp7_budget_depth.png")


def main():
    print("Generating figures from experimental results...")
    print(f"Reading from: {RESULTS_DIR}")
    print(f"Saving to: {OUTPUT_DIR}")

    # Baseline experiments
    baseline_path = RESULTS_DIR / "results.json"
    if baseline_path.exists():
        data = load_json(baseline_path)
        plot_experiment_1(data)
        plot_experiment_2(data)
        plot_experiment_3(data)
    else:
        print(f"Warning: {baseline_path} not found. Run exp_framework.py first.")

    # Advanced experiments
    advanced_path = RESULTS_DIR / "advanced_results.json"
    if advanced_path.exists():
        data = load_json(advanced_path)
        plot_experiment_4(data)
        plot_experiment_5(data)
        plot_experiment_6(data)
        plot_experiment_7(data)
    else:
        print(f"Warning: {advanced_path} not found. Run exp_advanced.py first.")

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
