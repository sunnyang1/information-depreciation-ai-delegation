"""
Experiment 5: Heterogeneity Reduces Distortion (Prediction 7.3)

Holding total precision constant, heterogeneous signal distributions
yield lower aggregate distortion.
"""

from __future__ import annotations

from typing import List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import register
from exp_framework import TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K


@register(
    id="exp05",
    name="Heterogeneity Reduces Distortion",
    description="Heterogeneous signals yield lower aggregate distortion.",
    category="advanced",
)
def run_experiment_5(
    budget_tokens: int = 20_000,
    depth: int = 3,
    n_signals: int = 100,
    total_precision: float = 100.0,
    n_trials: int = 100,
) -> dict:
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Heterogeneity Reduces Distortion")
    print("=" * 70)
    print(f"\nSetup: Depth={depth}, K={n_signals}, Total precision={total_precision}")
    print(f"Effective budget={effective_budget:.1f}")
    print("Compare: Homogeneous vs Heterogeneous signal distributions")
    print()

    chain = TheoreticalLLMChain()

    homog_prec = [total_precision / n_signals] * n_signals

    heterog_prec = []
    alpha_pareto = 1.5
    for i in range(n_signals):
        rank = i + 1
        weight = 1.0 / (rank ** alpha_pareto)
        heterog_prec.append(weight)
    scale = total_precision / sum(heterog_prec)
    heterog_prec = [p * scale for p in heterog_prec]

    moderate_prec = []
    alpha_mod = 0.8
    for i in range(n_signals):
        rank = i + 1
        weight = 1.0 / (rank ** alpha_mod)
        moderate_prec.append(weight)
    scale_mod = total_precision / sum(moderate_prec)
    moderate_prec = [p * scale_mod for p in moderate_prec]

    configs = {
        "homogeneous": homog_prec,
        "moderate_heterogeneity": moderate_prec,
        "high_heterogeneity": heterog_prec,
    }

    results = {}
    for name, precisions in configs.items():
        budgets = [effective_budget] * depth
        trial_accs = []
        trial_deltas = []
        trial_rhos = []
        for _ in range(n_trials):
            states = chain.precision_path(budgets, [n_signals] * depth, heterogeneity=precisions)
            final_state = states[-1]
            acc = chain.accuracy_from_precision(final_state.tau_post)
            trial_accs.append(acc)
            trial_deltas.append(1.0 - final_state.rho)
            trial_rhos.append(final_state.rho)

        mean_acc = float(np.mean(trial_accs))
        std_acc = float(np.std(trial_accs))
        mean_delta = float(np.mean(trial_deltas))
        mean_rho = float(np.mean(trial_rhos))
        gini = compute_gini(precisions)

        results[name] = {
            "accuracy": mean_acc,
            "accuracy_std": std_acc,
            "distortion_delta": mean_delta,
            "rho": mean_rho,
            "gini": gini,
        }

        print(
            f"  {name:25s}: Accuracy={mean_acc:.4f} (±{std_acc:.4f}), "
            f"Delta={mean_delta:.4f}, Rho={mean_rho:.4f}, Gini={gini:.3f}"
        )

    names = list(results.keys())
    ginis = [results[n]["gini"] for n in names]
    deltas = [results[n]["distortion_delta"] for n in names]
    rhos = [results[n]["rho"] for n in names]

    def _pearson_r(x, y):
        x, y = np.array(x), np.array(y)
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=0), np.std(y, ddof=0)
        if sx == 0 or sy == 0:
            return 0.0
        return float(np.mean((x - mx) * (y - my)) / (sx * sy))

    corr_delta = _pearson_r(ginis, deltas)
    corr_rho = _pearson_r(ginis, rhos)
    confirmed = (corr_delta < 0.0) and (corr_rho > 0.0)
    print(f"\n  Gini vs Delta correlation = {corr_delta:+.3f} (Expected: negative)")
    print(f"  Gini vs Rho   correlation = {corr_rho:+.3f} (Expected: positive)")
    print(f"  Prediction confirmed: {confirmed}")

    return {
        "experiment_id": "exp05",
        "results": results,
        "correlations": {
            "gini_vs_delta": corr_delta,
            "gini_vs_rho": corr_rho,
        },
        "prediction_confirmed": bool(confirmed),
    }


def compute_gini(values: List[float]) -> float:
    arr = np.array(sorted(values))
    n = len(arr)
    cumsum = np.cumsum(arr)
    return (2.0 * np.sum((np.arange(1, n + 1) * arr)) / (n * cumsum[-1])) - (n + 1.0) / n
