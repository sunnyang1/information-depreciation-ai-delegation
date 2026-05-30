"""
Experiment 7: Budget Expansion Increases Depth (Prediction 7.5)

Larger context window increases optimal depth.
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
    KAPPA,
    PHI,
)


@register(
    id="exp07",
    name="Budget Expansion Increases Depth",
    description="Larger context windows permit deeper optimal chains.",
    category="advanced",
)
def run_experiment_7(
    budget_list_tokens: Optional[List[int]] = None,
    max_depth: int = 10,
) -> dict:
    budget_list_tokens = budget_list_tokens or [4_096, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000]

    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Budget Expansion Increases Depth")
    print("=" * 70)
    print(f"\nVary context-window budget A and compute optimal depth L*")
    print(f"Max depth considered: {max_depth}")
    print()

    chain = TheoreticalLLMChain()
    results = {}
    prev_L_star = 0
    monotonic = True

    for budget_tokens in budget_list_tokens:
        effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
        net_values = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            K_list = [DEFAULT_K] * L
            states = chain.precision_path(budgets, K_list)
            final_tau = states[-1].tau_post if states else 0.0
            value = -chain.psi / max(final_tau, 1e-6)
            cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets)
            net_values.append(value - cost)

        L_star = int(np.argmax(net_values) + 1)
        results[f"budget_{budget_tokens}"] = {
            "budget_tokens": budget_tokens,
            "effective_budget": effective_budget,
            "optimal_depth": L_star,
            "net_values": [float(v) for v in net_values],
        }

        if L_star < prev_L_star:
            monotonic = False
        prev_L_star = L_star

        print(
            f"  Budget = {budget_tokens:10,} tokens ({effective_budget:8.1f} units): "
            f"L* = {L_star}"
        )

    print(f"\n  Monotonic increase confirmed: {monotonic}")

    return {
        "experiment_id": "exp07",
        "budget_list_tokens": budget_list_tokens,
        "max_depth": max_depth,
        "results": results,
        "monotonic": bool(monotonic),
    }
