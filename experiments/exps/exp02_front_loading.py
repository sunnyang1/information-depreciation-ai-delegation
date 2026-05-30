"""
Experiment 2: Front-Loading Validation (Proposition 4)

Compare budget allocation strategies.
Prediction: Front-loaded > Uniform > Back-loaded.
"""

from __future__ import annotations

import math
from typing import Dict

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
)


@register(
    id="exp02",
    name="Front-Loading Validation",
    description="Front-loaded budget allocation outperforms uniform and back-loaded.",
    category="baseline",
)
def run_experiment_2(
    depth: int = 3,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    n_trials: int = DEFAULT_N_TRIALS,
) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Front-Loading Validation")
    print("=" * 70)
    print(f"\nSetup: Fix total budget B = {total_budget:,} tokens, depth L = {depth}")
    print("Compare: Uniform | Front-loaded | Back-loaded")
    print()

    chain = TheoreticalLLMChain()
    results: Dict[str, ExperimentResult] = {}

    for strategy in [
        BudgetStrategy.UNIFORM,
        BudgetStrategy.FRONT_LOADED,
        BudgetStrategy.BACK_LOADED,
    ]:
        result = chain.run_chain(
            depth=depth,
            strategy=strategy,
            total_budget=total_budget,
            n_trials=n_trials,
        )
        results[strategy.value] = result
        print(
            f"  Strategy: {strategy.value:18s} | "
            f"Accuracy = {result.accuracy:.4f} (±{result.metadata['accuracy_std']:.4f}) | "
            f"Budgets = {result.metadata['budgets']}"
        )

    ranking = sorted(results.items(), key=lambda x: -x[1].accuracy)
    print(f"\n  Ranking: {' > '.join([f'{s} ({r.accuracy:.4f})' for s, r in ranking])}")
    front_loaded_best = ranking[0][0] in ("front_loaded", "geometric_front")
    print(f"  Front-loading confirmed best: {front_loaded_best}")

    print("\n  Pairwise comparisons (approximate t-tests):")
    strategies = list(results.keys())
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            r1, r2 = results[s1], results[s2]
            se = math.sqrt(
                (r1.metadata["accuracy_std"] ** 2 + r2.metadata["accuracy_std"] ** 2) / n_trials
            )
            if se > 0:
                t_stat = (r1.accuracy - r2.accuracy) / se
                print(f"    {s1} vs {s2}: t = {t_stat:+.3f}")

    return {
        "experiment_id": "exp02",
        "depth": depth,
        "results": {
            s: {
                "strategy": s,
                "accuracy": r.accuracy,
                "accuracy_std": r.metadata["accuracy_std"],
                "final_tau": r.metadata["final_tau"],
                "budgets": r.metadata["budgets"],
            }
            for s, r in results.items()
        },
        "ranking": [s for s, _ in ranking],
        "front_loaded_best": bool(front_loaded_best),
    }
