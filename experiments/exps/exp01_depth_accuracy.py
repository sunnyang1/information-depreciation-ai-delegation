"""
Experiment 1: Depth-Accuracy Tradeoff (Prediction 7.1)

Measure accuracy as a function of chain depth L.
Prediction: Accuracy is decreasing and concave in L.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import register
from exp_framework import (
    TheoreticalLLMChain,
    BudgetStrategy,
    ExperimentResult,
    DEFAULT_TOTAL_BUDGET,
    DEFAULT_N_TRIALS,
    GAMMA,
    ETA_UNDERLINE,
    ETA_BAR,
)


@register(
    id="exp01",
    name="Depth-Accuracy Tradeoff",
    description="Accuracy degrades with chain depth (concave).",
    category="baseline",
)
def run_experiment_1(
    depths: Optional[List[int]] = None,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    n_trials: int = DEFAULT_N_TRIALS,
) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Depth-Accuracy Tradeoff")
    print("=" * 70)
    print(f"\nSetup: Fix total budget B = {total_budget:,} tokens, uniform allocation")
    print(f"Vary depth L = {depths or list(range(1, 6))}")
    print(f"Model: gamma={GAMMA}, eta in [{ETA_UNDERLINE}, {ETA_BAR}]")
    print()

    chain = TheoreticalLLMChain()
    results = []
    depths = depths or list(range(1, 6))

    for depth in depths:
        result = chain.run_chain(
            depth=depth,
            strategy=BudgetStrategy.UNIFORM,
            total_budget=total_budget,
            n_trials=n_trials,
        )
        results.append(result)
        print(
            f"  Depth L={depth}: Accuracy = {result.accuracy:.4f} "
            f"(±{result.metadata['accuracy_std']:.4f}), "
            f"Final tau = {result.metadata['final_tau']:.2f}"
        )

    # Statistical test for concavity
    accs = [r.accuracy for r in results]
    depths_arr = np.array(depths, dtype=float)
    if len(accs) >= 3:
        X = np.column_stack((depths_arr, depths_arr ** 2, np.ones(len(depths_arr))))
        y = np.array(accs)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        concave = beta[1] < 0
        print(f"\n  Quadratic fit: beta_L = {beta[0]:.4f}, beta_L2 = {beta[1]:.4f}")
        print(f"  Concave (beta_L2 < 0): {concave}")

    monotonic = all(accs[i] >= accs[i + 1] for i in range(len(accs) - 1))
    print(f"  Monotonically decreasing: {monotonic}")

    return {
        "experiment_id": "exp01",
        "depths": depths,
        "results": [
            {
                "depth": r.depth,
                "accuracy": r.accuracy,
                "accuracy_std": r.metadata["accuracy_std"],
                "final_tau": r.metadata["final_tau"],
                "precision_retained": r.precision_retained,
            }
            for r in results
        ],
        "concave": bool(concave) if len(accs) >= 3 else None,
        "monotonic": bool(monotonic),
    }
