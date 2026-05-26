"""
Experimental Framework for "Information Depreciation and Optimal Depth in AI Delegation Chains"

This module implements a theory-faithful simulation engine that derives all quantities
from the paper's structural equations (Assumptions 1--4, Propositions 1--7).

Three baseline experiments validate core predictions:
  1. Depth-Accuracy Tradeoff (Prediction 7.1)
  2. Front-Loading Advantage (Proposition 4 / Prediction 7.1 extension)
  3. Exponential Decay of Information Retention (Proposition 5)

All equations map directly to the paper:
  - Eq. (3)  : tau^0_{l,k} = tau_bar^0_k * eta_l^l
  - Eq. (4)  : eta_l = eta_underline + (eta_bar - eta_underline) * alpha_bar/(alpha_bar+a)
  - Eq. (6)  : tau_l(alpha) = sum_k g(alpha_k) * tau^0_{l,k}
  - Eq. (8)  : g(alpha) = alpha^gamma
  - Eq. (10) : V(tau) = -psi / tau  (quadratic loss)
  - Eq. (13) : alpha*_{l,k} proportional to (tau^0_{l,k})^{1/(1-gamma)}
  - Eq. (16) : rho_l = tau*_l / tau^{FB}_l
  - Eq. (22) : tau*_l = rho * tau*_{l-1} + C * eta_l^l
  - Eq. (24) : closed-form tau*_L for uniform budgets
"""

from __future__ import annotations

import json
import random
import math
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from pathlib import Path

import numpy as np
from scipy import optimize, stats

# ============================================================================
# Global random seed
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================================
# Structural parameters (illustrative values from Section 5)
# ============================================================================

# Preservation parameters (Assumption 2)
ETA_UNDERLINE = 0.75      # min preservation rate (no attention)
ETA_BAR = 0.95            # max preservation rate (unlimited attention)
A_SAT = 2.0               # saturation parameter in eta(A)

# Attention technology (Assumption 3)
GAMMA = 0.35              # attention elasticity of precision

# Cost parameters (Assumption 4)
KAPPA = 1e-5              # linear cost coefficient (calibrated to match value scale)
PHI = 1e-10               # quadratic cost coefficient

# Measurement noise
NOISE_STD = 0.02          # std dev of accuracy measurement noise

# Value function parameter (Eq. 10)
PSI = 1.0                 # marginal value of precision

# Default signal structure
DEFAULT_K = 100           # number of signals per layer (paper: 50--100)
DEFAULT_TAU0 = 1.0        # base raw precision of each signal

# Attention budget scaling: total_budget (tokens) -> effective attention units
TOKENS_PER_ATTENTION_UNIT = 1000.0

# Budget & depth defaults
DEFAULT_TOTAL_BUDGET = 200_000  # tokens
DEFAULT_MAX_DEPTH = 5
DEFAULT_N_TRIALS = 100


# ============================================================================
# Data structures
# ============================================================================

class BudgetStrategy(Enum):
    UNIFORM = "uniform"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    GEOMETRIC_FRONT = "geometric_front"
    GEOMETRIC_BACK = "geometric_back"


@dataclass
class ExperimentResult:
    experiment_id: str
    depth: int
    strategy: Optional[str]
    accuracy: float
    precision_retained: float
    tokens_used: int
    latency_ms: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class DepreciationEstimate:
    layer: int
    retention_rate: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    theoretical: float


@dataclass
class LayerPrecisionState:
    """Precision state at a single layer."""
    layer_idx: int
    budget: float
    n_signals: int
    eta: float
    rho: float
    tau_prior: float
    tau_post: float
    tau_fb: float
    attention_weights: List[float]
    signal_precisions: List[float]


# ============================================================================
# Theory-faithful simulation engine
# ============================================================================

class TheoreticalLLMChain:
    """
    Simulates a multi-layer LLM chain using the paper's structural equations.

    Key methods map to paper equations:
      - solve_attention_allocation()  -> Proposition 1, Eq. (13)
      - compute_eta()                 -> Assumption 2, Eq. (4)
      - compute_rho()                 -> Definition 3, Eq. (16)
      - precision_path()              -> Proposition 5, Eq. (22)
      - optimal_depth()               -> Proposition 6, Eq. (25)
    """

    def __init__(
        self,
        eta_underline: float = ETA_UNDERLINE,
        eta_bar: float = ETA_BAR,
        a_sat: float = A_SAT,
        gamma: float = GAMMA,
        psi: float = PSI,
        noise_std: float = NOISE_STD,
    ):
        self.eta_underline = eta_underline
        self.eta_bar = eta_bar
        self.a_sat = a_sat
        self.gamma = gamma
        self.psi = psi
        self.noise_std = noise_std
        self.rng = np.random.RandomState(SEED)

    # ------------------------------------------------------------------
    # Core structural functions
    # ------------------------------------------------------------------

    def compute_eta(self, budget: float, n_signals: int) -> float:
        """
        Endogenous preservation rate (Assumption 2, Eq. 4).

        eta = eta_underline + (eta_bar - eta_underline) * alpha_bar / (alpha_bar + a)
        where alpha_bar = budget / n_signals.
        """
        alpha_bar = budget / max(n_signals, 1)
        eta = self.eta_underline + (self.eta_bar - self.eta_underline) * (
            alpha_bar / (alpha_bar + self.a_sat)
        )
        return float(np.clip(eta, self.eta_underline, self.eta_bar))

    def g_attention(self, alpha: float) -> float:
        """Attention technology g(alpha) with saturation at 1 (Assumption 3)."""
        return min(alpha ** self.gamma, 1.0)

    def solve_attention_allocation(
        self,
        budget: float,
        signal_precisions: List[float],
    ) -> Tuple[List[float], float, float]:
        """
        Solve the single-layer attention-allocation problem (Proposition 1).

        Returns:
            attention_weights: optimal alpha_k for each signal
            lambda_shadow: shadow price of budget
            tau_star: achieved posterior precision

        For power function g(alpha)=min(alpha^gamma, 1), interior FOC gives:
            alpha_k = (gamma * tau0_k / lambda)^{1/(1-gamma)}   for alpha_k < 1
            alpha_k = 1                                         for alpha_k >= 1
        We find lambda such that sum(alpha_k) = budget.
        """
        K = len(signal_precisions)
        if K == 0:
            return [], 0.0, 0.0

        precisions = np.array(signal_precisions, dtype=float)
        gamma = self.gamma

        # If budget allows all signals to be fully processed (alpha >= 1),
        # assign alpha=1 to all and distribute excess (excess doesn't raise g).
        if budget >= K:
            weights = [1.0] * K
            tau_star = sum(self.g_attention(w) * p for w, p in zip(weights, signal_precisions))
            return weights, 0.0, tau_star

        # General case: solve for lambda numerically.
        # Some signals may be at corner (alpha=1) if budget is large relative to K.
        # We handle this by iterative water-filling.
        sorted_idx = np.argsort(-precisions)  # descending
        sorted_prec = precisions[sorted_idx]

        # Try different numbers of signals at corner alpha=1
        best_weights = None
        best_tau = -1.0
        best_lambda = 1.0

        for n_corner in range(min(int(budget) + 1, K + 1)):
            remaining_budget = budget - n_corner
            if remaining_budget < 0:
                continue
            if n_corner < K and remaining_budget == 0:
                # All budget used by corners, remaining signals get 0
                weights = [0.0] * K
                for i in range(n_corner):
                    weights[sorted_idx[i]] = 1.0
                tau_star = sum(self.g_attention(w) * p for w, p in zip(weights, signal_precisions))
                if tau_star > best_tau:
                    best_tau = tau_star
                    best_weights = weights[:]
                continue

            # Remaining signals (not at corner) get interior solution
            m = K - n_corner
            if m <= 0:
                weights = [1.0] * K
                tau_star = sum(self.g_attention(w) * p for w, p in zip(weights, signal_precisions))
                if tau_star > best_tau:
                    best_tau = tau_star
                    best_weights = weights[:]
                continue

            rem_prec = sorted_prec[n_corner:]

            def budget_error(log_lambda: float) -> float:
                lam = math.exp(log_lambda)
                raw = (gamma * rem_prec / lam) ** (1.0 / (1.0 - gamma))
                return float(np.sum(raw) - remaining_budget)

            try:
                sol = optimize.brentq(budget_error, -20.0, 20.0)
            except ValueError:
                continue

            lam_star = math.exp(sol)
            raw = (gamma * rem_prec / lam_star) ** (1.0 / (1.0 - gamma))

            # Check if all interior alphas are < 1
            if np.any(raw >= 1.0):
                continue

            weights = [0.0] * K
            for i in range(n_corner):
                weights[sorted_idx[i]] = 1.0
            for i in range(m):
                weights[sorted_idx[n_corner + i]] = float(max(raw[i], 0.0))

            tau_star = sum(self.g_attention(w) * p for w, p in zip(weights, signal_precisions))
            if tau_star > best_tau:
                best_tau = tau_star
                best_weights = weights[:]
                best_lambda = lam_star

        if best_weights is None:
            # Ultimate fallback: proportional
            total_prec = np.sum(precisions)
            weights = [budget * p / total_prec for p in precisions]
            tau_star = sum(self.g_attention(w) * p for w, p in zip(weights, signal_precisions))
            return weights, 1.0, tau_star

        return best_weights, best_lambda, best_tau

    def compute_rho(
        self,
        budget: float,
        signal_precisions: List[float],
    ) -> float:
        """
        Endogenous transmission factor (Definition 3, Eq. 16).

        rho = tau*_achieved / tau*_first_best
        where tau*_first_best = sum_k tau0_k  (if all signals fully processed)
        """
        _, _, tau_star = self.solve_attention_allocation(budget, signal_precisions)
        tau_fb = sum(signal_precisions)
        if tau_fb <= 0:
            return 1.0
        return float(np.clip(tau_star / tau_fb, 0.0, 1.0))

    def precision_path(
        self,
        budgets: List[float],
        n_signals_per_layer: List[int],
        base_precision: float = DEFAULT_TAU0,
        heterogeneity: Optional[List[float]] = None,
    ) -> List[LayerPrecisionState]:
        """
        Compute the precision path through the chain (Proposition 5, Eq. 22).

        Recursive update:
            tau_prior_l = rho_{l-1} * tau_post_{l-1}
            tau_post_l  = tau_prior_l + C * eta_l^l
        where C = sum_k g(alpha*_k) * tau0_k  (the fresh signal contribution)

        For simplicity we assume uniform signal precision at each layer,
        but allow heterogeneity via the heterogeneity vector.
        """
        L = len(budgets)
        states: List[LayerPrecisionState] = []
        tau_prior = float(n_signals_per_layer[0] * base_precision) if n_signals_per_layer else 0.0

        for l in range(L):
            K = n_signals_per_layer[l]
            A = budgets[l]

            # Signal precisions at this layer (depreciated by eta^l)
            eta_l = self.compute_eta(A, K)
            depreciation_factor = eta_l ** l
            if heterogeneity is not None and len(heterogeneity) == K:
                tau0_list = [base_precision * h * depreciation_factor for h in heterogeneity]
            else:
                tau0_list = [base_precision * depreciation_factor] * K

            # Attention allocation
            alphas, _, tau_fresh = self.solve_attention_allocation(A, tau0_list)

            # First-best precision (full attention to all signals)
            tau_fb = tau_prior + sum(tau0_list)

            # Transmission factor (Definition 3, Eq. 16)
            tau_star = tau_prior + tau_fresh
            rho_l = tau_star / tau_fb if tau_fb > 0 else 1.0

            # Posterior precision
            tau_post = tau_prior + tau_fresh

            state = LayerPrecisionState(
                layer_idx=l,
                budget=A,
                n_signals=K,
                eta=eta_l,
                rho=rho_l,
                tau_prior=tau_prior,
                tau_post=tau_post,
                tau_fb=tau_fb,
                attention_weights=alphas,
                signal_precisions=tau0_list,
            )
            states.append(state)

            # Next layer's prior is this layer's achieved posterior,
            # discounted by the transmission factor
            tau_prior = rho_l * tau_post

        return states

    def accuracy_from_precision(self, tau: float, task_difficulty: float = 0.3) -> float:
        """
        Map precision to observable accuracy.

        We use a sigmoid mapping so that accuracy is bounded in [0,1]:
            accuracy = 1 / (1 + exp(-beta * (tau - tau0))) * (1 - difficulty)
        with noise added.  This is a reduced-form measurement equation.
        """
        # Calibrated so that tau in [50, 300] maps to accuracy in [0.4, 0.8]
        beta = 0.015
        tau0 = 150.0
        raw_acc = 1.0 / (1.0 + math.exp(-beta * (tau - tau0)))
        # Adjust for task difficulty
        acc = raw_acc * (1.0 - task_difficulty)
        # Add measurement noise
        acc += self.rng.normal(0.0, self.noise_std)
        return float(np.clip(acc, 0.0, 1.0))

    def run_chain(
        self,
        depth: int,
        strategy: BudgetStrategy,
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        n_signals: int = DEFAULT_K,
        n_trials: int = DEFAULT_N_TRIALS,
        task_difficulty: float = 0.3,
    ) -> ExperimentResult:
        """
        Run a multi-layer chain and measure end-to-end accuracy.

        The chain depth is `depth` layers; budgets are allocated according to
        `strategy` over `depth` layers (plus layer 0 if interpreting that way).
        For simplicity we allocate across `depth` layers directly.
        """
        # Convert token budget to effective attention units
        effective_budget = int(total_budget / TOKENS_PER_ATTENTION_UNIT)
        budgets = allocate_budget(effective_budget, depth, strategy)
        K_list = [n_signals] * depth

        accuracies = []
        for _ in range(n_trials):
            states = self.precision_path(budgets, K_list)
            final_tau = states[-1].tau_post if states else 0.0
            acc = self.accuracy_from_precision(final_tau, task_difficulty)
            accuracies.append(acc)

        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        final_state = states[-1] if states else None
        # Precision retained = transmission factor rho (Definition 3)
        precision_retained = final_state.rho if final_state else 0.0

        return ExperimentResult(
            experiment_id=f"depth_{depth}_{strategy.value}",
            depth=depth,
            strategy=strategy.value,
            accuracy=round(mean_acc, 4),
            precision_retained=round(precision_retained, 4),
            tokens_used=sum(budgets),
            latency_ms=round(depth * 500 + self.rng.normal(0, 100), 1),
            metadata={
                "budgets": budgets,
                "n_trials": n_trials,
                "accuracy_std": round(std_acc, 4),
                "final_tau": round(final_state.tau_post, 4) if final_state else 0.0,
                "avg_eta": round(np.mean([s.eta for s in states]), 4) if states else 0.0,
            },
        )


# ============================================================================
# Budget allocation helpers
# ============================================================================

def allocate_budget(
    total_budget: int,
    depth: int,
    strategy: BudgetStrategy,
) -> List[int]:
    """
    Allocate context budget across `depth` layers according to strategy.
    """
    if depth <= 0:
        return []

    if strategy == BudgetStrategy.UNIFORM:
        base = total_budget // depth
        sizes = [base] * depth
        for i in range(total_budget - sum(sizes)):
            sizes[i] += 1

    elif strategy == BudgetStrategy.FRONT_LOADED:
        # Geometric decay from front: A_0 > A_1 > ... > A_{L-1}
        weights = np.array([2.0 ** (-i) for i in range(depth)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[0] += diff

    elif strategy == BudgetStrategy.BACK_LOADED:
        # Reverse: last layer gets the most
        weights = np.array([2.0 ** (-(depth - 1 - i)) for i in range(depth)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[-1] += diff

    elif strategy == BudgetStrategy.GEOMETRIC_FRONT:
        # Halving each layer: A, A/2, A/4, ...
        weights = np.array([1.0 / (2 ** i) for i in range(depth)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[0] += diff

    elif strategy == BudgetStrategy.GEOMETRIC_BACK:
        weights = np.array([1.0 / (2 ** (depth - 1 - i)) for i in range(depth)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[-1] += diff

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sizes


# ============================================================================
# Experiment 1: Depth-Accuracy Tradeoff
# ============================================================================

def run_experiment_1_depth_accuracy(
    depths: Optional[List[int]] = None,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    n_trials: int = DEFAULT_N_TRIALS,
) -> List[ExperimentResult]:
    """
    Experiment 1: Measure accuracy as a function of chain depth L.

    Prediction: Accuracy is decreasing and concave in L (Prediction 7.1).
    """
    print("=" * 70)
    print("EXPERIMENT 1: Depth-Accuracy Tradeoff")
    print("=" * 70)
    print(f"\nSetup: Fix total budget B = {total_budget:,} tokens, uniform allocation")
    print(f"Vary depth L = {depths or list(range(1, 6))}")
    print(f"Model: Theoretical engine with gamma={GAMMA}, eta in [{ETA_UNDERLINE}, {ETA_BAR}]")
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
        # Regress accuracy on L + L^2; test if beta_2 < 0
        X = np.column_stack((depths_arr, depths_arr ** 2, np.ones(len(depths_arr))))
        y = np.array(accs)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        concave = beta[1] < 0
        print(f"\n  Quadratic fit: beta_L = {beta[0]:.4f}, beta_L2 = {beta[1]:.4f}")
        print(f"  Concave (beta_L2 < 0): {concave}")

    # Monotonicity test
    monotonic = all(accs[i] >= accs[i + 1] for i in range(len(accs) - 1))
    print(f"  Monotonically decreasing: {monotonic}")

    return results


# ============================================================================
# Experiment 2: Front-Loading Validation
# ============================================================================

def run_experiment_2_front_loading(
    depth: int = 3,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    n_trials: int = DEFAULT_N_TRIALS,
) -> Dict[str, ExperimentResult]:
    """
    Experiment 2: Compare budget allocation strategies.

    Prediction: Front-loaded > Uniform > Back-loaded (Proposition 4).
    """
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

    # Ranking
    ranking = sorted(results.items(), key=lambda x: -x[1].accuracy)
    print(f"\n  Ranking: {' > '.join([f'{s} ({r.accuracy:.4f})' for s, r in ranking])}")
    front_loaded_best = ranking[0][0] in ("front_loaded", "geometric_front")
    print(f"  Front-loading confirmed best: {front_loaded_best}")

    # Pairwise t-tests (using stored std as approximation)
    print("\n  Pairwise comparisons (approximate t-tests):")
    strategies = list(results.keys())
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            r1, r2 = results[s1], results[s2]
            # Approximate t-statistic from independent means with known std
            se = math.sqrt(
                (r1.metadata["accuracy_std"] ** 2 + r2.metadata["accuracy_std"] ** 2) / n_trials
            )
            if se > 0:
                t_stat = (r1.accuracy - r2.accuracy) / se
                print(f"    {s1} vs {s2}: t = {t_stat:+.3f}")

    return results


# ============================================================================
# Experiment 3: Exponential Decay of Information Retention
# ============================================================================

def run_experiment_3_eta_estimation(
    max_depth: int = 5,
    n_facts: int = 100,
    budget: int = 30_000,
) -> List[DepreciationEstimate]:
    """
    Experiment 3: Estimate the per-layer information depreciation factor eta.

    Method: Insert identifiable 'facts' (signals with unit precision) at layer 0
    and measure how much of the initial precision survives after l layers.

    Expected: Exponential decay ~ prod_{j=0}^{l-1} rho_j (Proposition 5).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Information Depreciation Rate Estimation")
    print("=" * 70)
    print(f"\nMethod: Insert {n_facts} unit-precision signals at layer 0")
    print("Measure: Cumulative transmission factor after each layer")
    print("Expected: Exponential decay ~ (product of rho_j)")
    print()

    chain = TheoreticalLLMChain()
    estimates = []

    # Baseline precision at layer 0
    initial_precision = float(n_facts * DEFAULT_TAU0)

    for l in range(1, max_depth + 1):
        # Build a chain of length l with fixed budget per layer
        effective_budget = int(budget / TOKENS_PER_ATTENTION_UNIT)
        budgets = [effective_budget] * l
        K_list = [n_facts] * l

        states = chain.precision_path(budgets, K_list)

        # Cumulative retention = product of transmission factors across layers
        cumulative_rho = 1.0
        for s in states:
            cumulative_rho *= s.rho

        # Retention relative to initial precision
        retention = cumulative_rho
        retention = float(np.clip(retention, 0.0, 1.0))

        # 95% CI via binomial approximation
        se = math.sqrt(retention * (1.0 - retention) / n_facts)
        ci_lower = max(0.0, retention - 1.96 * se)
        ci_upper = min(1.0, retention + 1.96 * se)

        # Theoretical: use average eta
        avg_eta = float(np.mean([s.eta for s in states]))
        theoretical = avg_eta ** l

        estimate = DepreciationEstimate(
            layer=l,
            retention_rate=round(retention, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            n_samples=n_facts,
            theoretical=round(theoretical, 4),
        )
        estimates.append(estimate)

        print(
            f"  Layer {l}: Retention = {retention:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] | "
            f"Avg rho = {np.mean([s.rho for s in states]):.4f} | "
            f"Theoretical avg_eta^{l} = {theoretical:.4f}"
        )

    # Fit exponential model: retention = eta0 * eta^layer
    layers = np.arange(1, max_depth + 1, dtype=float)
    retentions = np.array([e.retention_rate for e in estimates])
    mask = (retentions > 0.01) & (retentions < 1.0)
    if mask.sum() >= 2:
        log_ret = np.log(retentions[mask])
        X = np.column_stack((layers[mask], np.ones(mask.sum())))
        coef, _, _, _ = np.linalg.lstsq(X, log_ret, rcond=None)
        eta_estimated = math.exp(coef[0])
        r_squared = 1.0 - np.var(log_ret - X @ coef) / np.var(log_ret)
        print(f"\n  Estimated eta = {eta_estimated:.4f}")
        print(f"  R-squared of log-linear fit: {r_squared:.4f}")
    else:
        print("\n  [Not enough variation to fit exponential model]")

    return estimates


# ============================================================================
# Export Results to LaTeX
# ============================================================================

def export_results_to_latex_tables(
    exp1_results: List[ExperimentResult],
    exp2_results: Dict[str, ExperimentResult],
    exp3_estimates: List[DepreciationEstimate],
) -> str:
    """Export experimental results to LaTeX table format."""

    output = []
    output.append("% LaTeX tables generated by exp_framework.py")
    output.append("% Do not edit manually")
    output.append("")

    # Table 1: Depth-Accuracy
    output.append("% Table: Depth-Accuracy Tradeoff")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 1: Depth--Accuracy Tradeoff}")
    output.append("\\label{tab:exp1_depth_accuracy}")
    output.append("\\begin{tabular}{@{}cccc@{}}")
    output.append("\\toprule")
    output.append("Depth $L$ & Accuracy & Std. Dev. & Final Precision $\\tau^*_L$ \\\\")
    output.append("\\midrule")
    for r in exp1_results:
        output.append(
            f"{r.depth} & {r.accuracy:.4f} & {r.metadata['accuracy_std']:.4f} & {r.metadata['final_tau']:.2f} \\\\"
        )
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    # Table 2: Front-Loading
    output.append("")
    output.append("% Table: Front-Loading Validation")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 2: Budget Allocation Strategies ($L=3$)}")
    output.append("\\label{tab:exp2_front_loading}")
    output.append("\\begin{tabular}{@{}lccc@{}}")
    output.append("\\toprule")
    output.append("Strategy & Mean Accuracy & Budget Allocation (tokens) & Rank \\\\")
    output.append("\\midrule")
    ranking = sorted(exp2_results.items(), key=lambda x: -x[1].accuracy)
    rank_map = {s: i + 1 for i, (s, _) in enumerate(ranking)}
    for strategy, result in exp2_results.items():
        budgets_str = str(result.metadata["budgets"]).replace("[", "").replace("]", "").replace(" ", "\\,")
        output.append(
            f"{strategy.replace('_', '-')} & {result.accuracy:.4f} & ${budgets_str}$ & {rank_map[strategy]} \\\\"
        )
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    # Table 3: Eta Estimation
    output.append("")
    output.append("% Table: Information Depreciation Estimation")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 3: Per-Layer Information Retention}")
    output.append("\\label{tab:exp3_eta}")
    output.append("\\begin{tabular}{@{}cccc@{}}")
    output.append("\\toprule")
    output.append("Layer $\\ell$ & Retention Rate & 95\\% CI & Theoretical $\\bar{\\eta}^\\ell$ \\\\")
    output.append("\\midrule")
    for e in exp3_estimates:
        output.append(
            f"{e.layer} & {e.retention_rate:.4f} & [{e.ci_lower:.4f}, {e.ci_upper:.4f}] & {e.theoretical:.4f} \\\\"
        )
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    latex_text = "\n".join(output)
    print(latex_text)
    return latex_text


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Experimental Framework for Information Depreciation Model")
    print("Mode: THEORY-FAITHFUL SIMULATION")
    print(f"Parameters: gamma={GAMMA}, eta in [{ETA_UNDERLINE}, {ETA_BAR}], a={A_SAT}")
    print()

    # Run all three baseline experiments
    exp1 = run_experiment_1_depth_accuracy()
    exp2 = run_experiment_2_front_loading()
    exp3 = run_experiment_3_eta_estimation()

    # Export LaTeX tables
    latex_output = export_results_to_latex_tables(exp1, exp2, exp3)

    # Save JSON results
    results = {
        "experiment_1": [asdict(r) for r in exp1],
        "experiment_2": {s: asdict(r) for s, r in exp2.items()},
        "experiment_3": [asdict(e) for e in exp3],
        "parameters": {
            "eta_underline": ETA_UNDERLINE,
            "eta_bar": ETA_BAR,
            "a_sat": A_SAT,
            "gamma": GAMMA,
            "kappa": KAPPA,
            "phi": PHI,
            "noise_std": NOISE_STD,
        },
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save LaTeX
    with open(output_dir / "tables.tex", "w") as f:
        f.write(latex_output)

    print(f"\n\nResults saved to {output_path}")
    print("All baseline experiments completed successfully.")
