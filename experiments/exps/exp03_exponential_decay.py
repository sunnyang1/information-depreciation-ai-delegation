"""
Experiment 3: Exponential Decay of Information Retention (Proposition 5)

Estimate the per-layer information depreciation factor eta.
Expected: Exponential decay ~ prod rho_j across layers.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import register
from exp_framework import (
    TheoreticalLLMChain,
    DepreciationEstimate,
    TOKENS_PER_ATTENTION_UNIT,
    DEFAULT_TAU0,
    DEFAULT_K,
)


@register(
    id="exp03",
    name="Exponential Decay of Retention",
    description="Information retention follows exponential decay ~ eta^l.",
    category="baseline",
)
def run_experiment_3(
    max_depth: int = 5,
    n_facts: int = 100,
    budget: int = 30_000,
) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Information Depreciation Rate Estimation")
    print("=" * 70)
    print(f"\nMethod: Insert {n_facts} unit-precision signals at layer 0")
    print("Measure: Cumulative transmission factor after each layer")
    print("Expected: Exponential decay ~ (product of rho_j)")
    print()

    chain = TheoreticalLLMChain()
    estimates = []

    for l in range(1, max_depth + 1):
        effective_budget = int(budget / TOKENS_PER_ATTENTION_UNIT)
        budgets = [effective_budget] * l
        K_list = [n_facts] * l

        states = chain.precision_path(budgets, K_list)
        cumulative_rho = 1.0
        for s in states:
            cumulative_rho *= s.rho

        retention = float(np.clip(cumulative_rho, 0.0, 1.0))
        se = math.sqrt(retention * (1.0 - retention) / n_facts)
        ci_lower = max(0.0, retention - 1.96 * se)
        ci_upper = min(1.0, retention + 1.96 * se)
        avg_eta = float(np.mean([s.eta for s in states]))
        theoretical = avg_eta ** l

        estimates.append({
            "layer": l,
            "retention_rate": round(retention, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n_samples": n_facts,
            "theoretical": round(theoretical, 4),
        })

        print(
            f"  Layer {l}: Retention = {retention:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] | "
            f"Avg rho = {np.mean([s.rho for s in states]):.4f} | "
            f"Theoretical avg_eta^{l} = {theoretical:.4f}"
        )

    layers = np.arange(1, max_depth + 1, dtype=float)
    retentions = np.array([e["retention_rate"] for e in estimates])
    mask = (retentions > 0.01) & (retentions < 1.0)
    fit = {}
    if mask.sum() >= 2:
        log_ret = np.log(retentions[mask])
        X = np.column_stack((layers[mask], np.ones(mask.sum())))
        coef, _, _, _ = np.linalg.lstsq(X, log_ret, rcond=None)
        eta_estimated = math.exp(coef[0])
        r_squared = 1.0 - np.var(log_ret - X @ coef) / np.var(log_ret)
        fit["eta_estimated"] = round(eta_estimated, 4)
        fit["r_squared"] = round(r_squared, 4)
        print(f"\n  Estimated eta = {eta_estimated:.4f}")
        print(f"  R-squared of log-linear fit: {r_squared:.4f}")
    else:
        print("\n  [Not enough variation to fit exponential model]")

    return {
        "experiment_id": "exp03",
        "estimates": estimates,
        "fit": fit,
    }
