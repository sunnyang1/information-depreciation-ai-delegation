"""
Experiment 6: Cost Irrelevance for Depth (Prediction 7.4)

Reducing API cost does not increase optimal depth beyond a finite limit.
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
    PHI,
    PSI,
)


@register(
    id="exp06",
    name="Cost Irrelevance for Depth",
    description="Reducing cost kappa does not increase optimal depth indefinitely.",
    category="advanced",
)
def run_experiment_6(
    budget_tokens: int = 80_000,
    kappa_values: Optional[List[float]] = None,
    max_depth: int = 10,
) -> dict:
    kappa_values = kappa_values or [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0]
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Cost Irrelevance for Depth")
    print("=" * 70)
    print(f"\nSetup: Effective budget={effective_budget:.1f}, max_depth={max_depth}")
    print("Vary cost parameter kappa and compute optimal depth L*")
    print()

    chain = TheoreticalLLMChain()
    results = {}

    for kappa in kappa_values:
        net_values = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            K_list = [DEFAULT_K] * L
            states = chain.precision_path(budgets, K_list)
            final_tau = states[-1].tau_post if states else 0.0
            value = -chain.psi / max(final_tau, 1e-6)
            cost = sum(kappa * A + 0.5 * PHI * (A ** 2) for A in budgets)
            net_value = value - cost
            net_values.append(net_value)

        L_star = int(np.argmax(net_values) + 1)
        results[f"kappa_{kappa}"] = {
            "kappa": kappa,
            "optimal_depth": L_star,
            "net_values": [float(v) for v in net_values],
        }
        print(f"  kappa = {kappa:>10.2e}: L* = {L_star}, NetValue[L*] = {net_values[L_star-1]:.4f}")

    depths = [results[k]["optimal_depth"] for k in results]
    converged = len(set(depths[-3:])) == 1
    print(f"\n  Optimal depth converges as kappa -> 0: {converged}")
    print(f"  Final L* = {depths[-1]}")

    return {
        "experiment_id": "exp06",
        "budget_tokens": budget_tokens,
        "max_depth": max_depth,
        "results": results,
        "converged": bool(converged),
        "final_L_star": int(depths[-1]),
    }
