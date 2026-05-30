"""
Supplementary Tables 6 & 9: Sensitivity Analysis
"""

from __future__ import annotations

import math

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import register
from exp_framework import (
    TheoreticalLLMChain, TOKENS_PER_ATTENTION_UNIT, DEFAULT_K, DEFAULT_TAU0, KAPPA, PHI,
)

FIXED_COST_PER_LAYER = 0.02


def _value_fn(tau: float) -> float:
    return 1.0 * (1.0 - math.exp(-tau / 200.0))


@register(
    id="sup07",
    name="Sensitivity Analysis",
    description="Sensitivity of predictions to gamma and optimal depth grid.",
    category="supplementary",
)
def run_supplementary_07() -> dict:
    print("\n[Supplementary 07] Sensitivity Analysis")

    gamma_values = [0.29, 0.35, 0.41]
    eta_values = [0.70, 0.80, 0.85, 0.90, 0.95]
    budget_tokens = 128_000
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    max_depth = 15

    table6 = []
    for gamma in gamma_values:
        chain = TheoreticalLLMChain(gamma=gamma)
        profits = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            states = chain.precision_path(budgets, [DEFAULT_K] * L)
            tau = states[-1].tau_post
            value = _value_fn(tau)
            variable_cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets)
            cost = variable_cost + L * FIXED_COST_PER_LAYER
            profits.append(value - cost)
        L_star = int(np.argmax(profits) + 1)

        budget_5 = 30_000
        eff_5 = budget_5 / TOKENS_PER_ATTENTION_UNIT
        budgets_5 = [eff_5] * 5
        states_5 = chain.precision_path(budgets_5, [DEFAULT_K] * 5)
        rho_at_5 = states_5[-1].rho if states_5 else 0.0
        tau_at_5 = states_5[-1].tau_post if states_5 else 0.0

        K = DEFAULT_K
        eta_5 = chain.compute_eta(eff_5, K)
        tau_fb = K * DEFAULT_TAU0
        for l in range(5):
            tau0_l = K * DEFAULT_TAU0 * (eta_5 ** l)
            tau_fb += tau0_l
        pct_fb = (tau_at_5 / tau_fb * 100) if tau_fb > 0 else 0.0

        table6.append({
            "gamma": gamma,
            "optimal_depth": L_star,
            "rho_at_L5": round(rho_at_5, 2),
            "pct_of_first_best": f"{pct_fb:.0f}%",
        })
        print(f"  gamma={gamma:.2f}: L*={L_star}, rho(L=5)={rho_at_5:.2f}")

    table9 = []
    gamma_grid = [0.03, 0.05, 0.10, 0.15, 0.20]
    for gamma in gamma_grid:
        row = {"gamma": gamma}
        for eta in eta_values:
            chain = TheoreticalLLMChain(gamma=gamma, eta_underline=eta,
                                        eta_bar=min(eta + 0.15, 0.99))
            profits = []
            for L in range(1, max_depth + 1):
                budgets = [effective_budget] * L
                states = chain.precision_path(budgets, [DEFAULT_K] * L)
                tau = states[-1].tau_post
                value = _value_fn(tau)
                variable_cost = sum(KAPPA * A + 0.5 * PHI * (A ** 2) for A in budgets)
                cost = variable_cost + L * FIXED_COST_PER_LAYER
                profits.append(value - cost)
            L_star = int(np.argmax(profits) + 1)
            row[f"eta_{eta}"] = L_star
        table9.append(row)

    print("\n  Table 9 (Optimal Depth L* grid):")
    header = "gamma      | " + " | ".join([f"eta={e:.2f}" for e in eta_values])
    print(f"  {header}")
    for row in table9:
        vals = " | ".join([f"{row[f'eta_{e}']:3d}" for e in eta_values])
        print(f"  {row['gamma']:.2f}   | {vals}")

    return {
        "experiment_id": "sup07",
        "table6": table6,
        "table9": table9,
    }
