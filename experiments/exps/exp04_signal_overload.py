"""
Experiment 4: Signal Overload (Prediction 7.2)

For a fixed context window, increasing K decreases per-signal precision.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import register
from exp_framework import (
    TheoreticalLLMChain,
    TOKENS_PER_ATTENTION_UNIT,
    DEFAULT_K,
)


@register(
    id="exp04",
    name="Signal Overload",
    description="Fixed budget, varying K decreases per-signal precision.",
    category="advanced",
)
def run_experiment_4(
    budget_tokens: int = 30_000,
    depths: Optional[List[int]] = None,
    n_signals_list: Optional[List[int]] = None,
    n_trials: int = 100,
) -> dict:
    depths = depths or [1, 3, 5]
    n_signals_list = n_signals_list or [20, 50, 100, 200, 500]
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Signal Overload")
    print("=" * 70)
    print(f"\nSetup: Fixed effective budget = {effective_budget:.1f} attention units")
    print(f"Vary K = {n_signals_list}")
    print(f"Depths tested: {depths}")
    print("Metric: Per-signal transmission factor (rho)")
    print()

    chain = TheoreticalLLMChain()
    results = {}

    for depth in depths:
        rhos = []
        per_signal_taus = []
        for K in n_signals_list:
            budgets = [effective_budget] * depth
            K_list = [K] * depth
            # precision_path is deterministic; no need for repeated trials
            states = chain.precision_path(budgets, K_list)
            final_state = states[-1]
            mean_rho = float(final_state.rho)
            mean_ps = float(final_state.tau_post / K)
            rhos.append(mean_rho)
            per_signal_taus.append(mean_ps)
            print(
                f"  Depth={depth}, K={K:4d}: Rho={mean_rho:.4f}, "
                f"Per-signal tau={mean_ps:.3f}"
            )

        K_arr = np.array(n_signals_list, dtype=float)
        fit = {}
        if len(rhos) >= 3:
            X = np.column_stack((K_arr, K_arr ** 2, np.ones(len(K_arr))))
            y = np.array(rhos)
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            fit["beta_K"] = float(beta[0])
            fit["beta_K2"] = float(beta[1])
            fit["decreasing"] = bool(beta[0] < 0)
            fit["concave"] = bool(beta[1] < 0)
            print(
                f"    -> Quadratic fit beta_K={beta[0]:.6f}, beta_K2={beta[1]:.6f}, "
                f"decreasing={fit['decreasing']}, concave={fit['concave']}"
            )

        results[f"depth_{depth}"] = {
            "n_signals": n_signals_list,
            "rhos": rhos,
            "per_signal_taus": per_signal_taus,
        }

    return {
        "experiment_id": "exp04",
        "budget_tokens": budget_tokens,
        "depths": depths,
        "results": results,
        "fit": fit,
    }
