"""
Advanced Experiments for "Information Depreciation and Optimal Depth in AI Delegation Chains"

Four additional experiments validate Predictions 7.2--7.5 from Section 6:
  4. Signal Overload                (Prediction 7.2)
  5. Heterogeneity Reduces Distortion (Prediction 7.3)
  6. Cost Irrelevance for Depth      (Prediction 7.4)
  7. Budget Expansion Increases Depth (Prediction 7.5)

All simulations use the theory-faithful engine from exp_framework.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
from scipy import optimize

from exp_framework import (
    TheoreticalLLMChain,
    BudgetStrategy,
    allocate_budget,
    TOKENS_PER_ATTENTION_UNIT,
    DEFAULT_K,
    DEFAULT_TAU0,
    ETA_UNDERLINE,
    ETA_BAR,
    A_SAT,
    GAMMA,
    KAPPA,
    PHI,
    SEED,
)


# ============================================================================
# Experiment 4: Signal Overload (Prediction 7.2)
# ============================================================================

def run_experiment_4_signal_overload(
    budget_tokens: int = 30_000,
    depths: List[int] = None,
    n_signals_list: List[int] = None,
    n_trials: int = 100,
) -> Dict:
    """
    Experiment 4: For a fixed context window, increasing K decreases per-signal accuracy.

    Prediction: The transmission factor rho (per-signal precision) is decreasing
    in K for fixed A (Prediction 7.2 / Proposition 2).
    """
    depths = depths or [1, 3, 5]
    n_signals_list = n_signals_list or [20, 50, 100, 200, 500]
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Signal Overload")
    print("=" * 70)
    print(f"\nSetup: Fixed effective budget = {effective_budget:.1f} attention units")
    print(f"Vary number of signals K = {n_signals_list}")
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
            trial_rhos = []
            trial_per_signal = []
            for _ in range(n_trials):
                states = chain.precision_path(budgets, K_list)
                final_state = states[-1]
                trial_rhos.append(final_state.rho)
                # Per-signal precision = tau_post / K
                trial_per_signal.append(final_state.tau_post / K)
            mean_rho = float(np.mean(trial_rhos))
            std_rho = float(np.std(trial_rhos))
            mean_ps = float(np.mean(trial_per_signal))
            rhos.append(mean_rho)
            per_signal_taus.append(mean_ps)
            print(f"  Depth={depth}, K={K:4d}: Rho={mean_rho:.4f} (±{std_rho:.4f}), Per-signal tau={mean_ps:.3f}")

        # Test decreasing in K: regress rho on K
        K_arr = np.array(n_signals_list, dtype=float)
        if len(rhos) >= 3:
            X = np.column_stack((K_arr, K_arr ** 2, np.ones(len(K_arr))))
            y = np.array(rhos)
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            concave = beta[1] < 0
            decreasing = beta[0] < 0
            print(f"    -> Quadratic fit beta_K={beta[0]:.6f}, beta_K2={beta[1]:.6f}, decreasing={decreasing}, concave={concave}")

        results[f"depth_{depth}"] = {
            "n_signals": n_signals_list,
            "rhos": rhos,
            "per_signal_taus": per_signal_taus,
        }

    return results


# ============================================================================
# Experiment 5: Heterogeneity Reduces Distortion (Prediction 7.3)
# ============================================================================

def run_experiment_5_heterogeneity(
    budget_tokens: int = 20_000,
    depth: int = 3,
    n_signals: int = 100,
    total_precision: float = 100.0,
    n_trials: int = 100,
) -> Dict:
    """
    Experiment 5: Heterogeneous signal precision reduces aggregate distortion.

    Prediction: Holding total precision constant, a more heterogeneous
    precision distribution yields lower aggregate distortion Δ (Prediction 7.3).
    """
    effective_budget = budget_tokens / TOKENS_PER_ATTENTION_UNIT
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Heterogeneity Reduces Distortion")
    print("=" * 70)
    print(f"\nSetup: Depth={depth}, K={n_signals}, Total precision={total_precision}")
    print(f"Effective budget={effective_budget:.1f} (budget < K, so selective attention matters)")
    print("Compare: Homogeneous vs Heterogeneous signal distributions")
    print()

    chain = TheoreticalLLMChain()

    # Homogeneous: all signals have equal precision
    homog_prec = [total_precision / n_signals] * n_signals

    # Heterogeneous: few dominant signals (Pareto-like)
    # 20% of signals carry 80% of precision
    heterog_prec = []
    alpha_pareto = 1.5  # shape parameter
    for i in range(n_signals):
        rank = i + 1
        weight = 1.0 / (rank ** alpha_pareto)
        heterog_prec.append(weight)
    # Normalize to total_precision
    scale = total_precision / sum(heterog_prec)
    heterog_prec = [p * scale for p in heterog_prec]

    # Moderate heterogeneity: intermediate
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
            # Aggregate distortion (Definition 2): Delta = 1 - rho
            trial_deltas.append(1.0 - final_state.rho)
            trial_rhos.append(final_state.rho)

        mean_acc = float(np.mean(trial_accs))
        std_acc = float(np.std(trial_accs))
        mean_delta = float(np.mean(trial_deltas))
        mean_rho = float(np.mean(trial_rhos))

        results[name] = {
            "accuracy": mean_acc,
            "accuracy_std": std_acc,
            "distortion_delta": mean_delta,
            "rho": mean_rho,
            "gini": compute_gini(precisions),
        }

        print(
            f"  {name:25s}: Accuracy={mean_acc:.4f} (±{std_acc:.4f}), "
            f"Delta={mean_delta:.4f}, Rho={mean_rho:.4f}, Gini={results[name]['gini']:.3f}"
        )

    # Prediction check: higher heterogeneity -> lower delta, higher accuracy
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
    print(f"\n  Gini vs Delta correlation = {corr_delta:+.3f} (Expected: negative)")
    print(f"  Gini vs Rho   correlation = {corr_rho:+.3f} (Expected: positive)")
    confirmed = (corr_delta < 0.0) and (corr_rho > 0.0)
    print(f"  Prediction confirmed (higher heterogeneity -> lower distortion): {confirmed}")

    return results


def compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient of a distribution."""
    arr = np.array(sorted(values))
    n = len(arr)
    cumsum = np.cumsum(arr)
    return (2.0 * np.sum((np.arange(1, n + 1) * arr)) / (n * cumsum[-1])) - (n + 1.0) / n


# ============================================================================
# Experiment 6: Cost Irrelevance for Depth (Prediction 7.4)
# ============================================================================

def run_experiment_6_cost_irrelevance(
    budget_tokens: int = 80_000,
    kappa_values: List[float] = None,
    max_depth: int = 10,
) -> Dict:
    """
    Experiment 6: Reducing API cost does not increase optimal depth.

    Prediction: L* converges to a finite limit as κ → 0 (Prediction 7.4).
    """
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
        # Compute net value for each depth
        net_values = []
        for L in range(1, max_depth + 1):
            budgets = [effective_budget] * L
            K_list = [DEFAULT_K] * L
            states = chain.precision_path(budgets, K_list)
            final_tau = states[-1].tau_post if states else 0.0
            # Value = -psi / tau (quadratic loss)
            value = -chain.psi / max(final_tau, 1e-6)
            # Cost = sum_l [kappa * A_l + phi/2 * A_l^2]
            cost = sum(kappa * A + 0.5 * PHI * (A ** 2) for A in budgets)
            net_value = value - cost
            net_values.append(net_value)

        # Find optimal depth (max net value)
        L_star = int(np.argmax(net_values) + 1)
        results[f"kappa_{kappa}"] = {
            "kappa": kappa,
            "optimal_depth": L_star,
            "net_values": [float(v) for v in net_values],
        }
        print(f"  kappa = {kappa:>10.2e}: L* = {L_star}, NetValue[L*] = {net_values[L_star-1]:.4f}")

    # Check convergence
    depths = [results[k]["optimal_depth"] for k in results]
    converged = len(set(depths[-3:])) == 1
    print(f"\n  Optimal depth converges as kappa -> 0: {converged}")
    print(f"  Final L* = {depths[-1]}")

    return results


# ============================================================================
# Experiment 7: Budget Expansion Increases Depth (Prediction 7.5)
# ============================================================================

def run_experiment_7_budget_depth(
    budget_list_tokens: List[int] = None,
    max_depth: int = 10,
) -> Dict:
    """
    Experiment 7: Larger context window increases optimal depth.

    Prediction: L* is increasing in A (Prediction 7.5).
    """
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

    return results


# ============================================================================
# LaTeX Export
# ============================================================================

def export_advanced_to_latex(
    exp4: Dict,
    exp5: Dict,
    exp6: Dict,
    exp7: Dict,
) -> str:
    """Export advanced experiment results to LaTeX tables."""
    output = []
    output.append("% LaTeX tables generated by exp_advanced.py")
    output.append("% Do not edit manually")
    output.append("")

    # Table 4: Signal Overload
    output.append("% Table: Signal Overload")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 4: Signal Overload (Fixed Budget, Varying $K$)}")
    output.append("\\label{tab:exp4_signal_overload}")
    output.append("\\begin{tabular}{@{}lccccc@{}}")
    output.append("\\toprule")
    output.append("Depth $L$ & " + " & ".join([f"$K={k}$" for k in exp4[list(exp4.keys())[0]]["n_signals"]]) + " \\\\")
    output.append("\\midrule")
    for key, data in exp4.items():
        rhos_str = " & ".join([f"{r:.3f}" for r in data["rhos"]])
        output.append(f"{key.replace('depth_', 'L=')} & {rhos_str} \\\\")
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    # Table 5: Heterogeneity
    output.append("")
    output.append("% Table: Heterogeneity Reduces Distortion")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 5: Heterogeneity and Aggregate Distortion}")
    output.append("\\label{tab:exp5_heterogeneity}")
    output.append("\\begin{tabular}{@{}lcccc@{}}")
    output.append("\\toprule")
    output.append("Distribution & Accuracy & $\\Delta$ & $\\rho$ & Gini \\\\")
    output.append("\\midrule")
    for name, data in exp5.items():
        label = name.replace("_", " ")
        output.append(
            f"{label} & {data['accuracy']:.4f} & {data['distortion_delta']:.4f} & "
            f"{data['rho']:.4f} & {data['gini']:.3f} \\\\"
        )
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    # Table 6: Cost Irrelevance
    output.append("")
    output.append("% Table: Cost Irrelevance")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 6: Cost Irrelevance for Optimal Depth}")
    output.append("\\label{tab:exp6_cost}")
    output.append("\\begin{tabular}{@{}cc@{}}")
    output.append("\\toprule")
    output.append("$\\kappa$ & Optimal Depth $L^*$ \\\\")
    output.append("\\midrule")
    for key, data in exp6.items():
        output.append(f"{data['kappa']:.4f} & {data['optimal_depth']} \\\\")
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    # Table 7: Budget-Depth
    output.append("")
    output.append("% Table: Budget Expansion")
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Experiment 7: Budget Expansion and Optimal Depth}")
    output.append("\\label{tab:exp7_budget}")
    output.append("\\begin{tabular}{@{}cc@{}}")
    output.append("\\toprule")
    output.append("Budget $A$ (tokens) & Optimal Depth $L^*$ \\\\")
    output.append("\\midrule")
    for key, data in exp7.items():
        output.append(f"{data['budget_tokens']:,} & {data['optimal_depth']} \\\\")
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
    print("Advanced Experiments for Information Depreciation Model")
    print("=" * 70)

    # Run experiments
    exp4 = run_experiment_4_signal_overload()
    exp5 = run_experiment_5_heterogeneity()
    exp6 = run_experiment_6_cost_irrelevance()
    exp7 = run_experiment_7_budget_depth()

    # Export LaTeX
    latex_output = export_advanced_to_latex(exp4, exp5, exp6, exp7)

    # Save JSON
    results = {
        "experiment_4_signal_overload": exp4,
        "experiment_5_heterogeneity": exp5,
        "experiment_6_cost_irrelevance": exp6,
        "experiment_7_budget_depth": exp7,
    }
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "advanced_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "advanced_tables.tex", "w") as f:
        f.write(latex_output)

    print(f"\n\nResults saved to {output_dir / 'advanced_results.json'}")
    print("All advanced experiments completed successfully.")
