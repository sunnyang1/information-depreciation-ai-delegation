#!/usr/bin/env python3
"""
Simulated Regression Analysis for Section 7 Empirical Validation
================================================================
Generates simulated data and runs OLS regressions to support
theoretical predictions in the paper.

Predictions tested:
- Prediction 7.1: Depth-Accuracy Tradeoff (concave, beta_2 < 0)
- Prediction 7.2: Signal Overload (K reduces accuracy)
- Prediction 7.5: Gamma identification with model & task FE
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# 1. Generate Needle-in-Haystack Data (200 observations)
# ============================================================
n_nh = 200

# Context lengths from 4K to 1M tokens
context_lengths = np.random.choice(
    [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576], 
    n_nh
)

# Model families (6 different models)
model_families = np.random.choice(
    ['LLaMA-2-7B', 'LLaMA-2-13B', 'LLaMA-2-70B', 'Mistral-7B', 'GPT-3.5', 'GPT-4'], 
    n_nh
)
model_fe_dict = {
    'LLaMA-2-7B': 0.72, 'LLaMA-2-13B': 0.78, 'LLaMA-2-70B': 0.85,
    'Mistral-7B': 0.80, 'GPT-3.5': 0.82, 'GPT-4': 0.92
}

# Generate accuracy following power-law decay with context length
gamma_true = 0.04
model_fe = np.array([model_fe_dict[m] for m in model_families])
log_inv_L = -np.log(context_lengths / 1000)
noise_nh = np.random.normal(0, 0.03, n_nh)
accuracy_nh = model_fe + gamma_true * log_inv_L + noise_nh
accuracy_nh = np.clip(accuracy_nh, 0.1, 1.0)

# Compute budget (FLOPs)
compute_flops = np.random.lognormal(mean=20, sigma=1.5, size=n_nh)
log_compute = np.log(compute_flops)

# Create DataFrame
df_nh = pd.DataFrame({
    'context_length': context_lengths,
    'log_context_length': np.log(context_lengths),
    'accuracy': accuracy_nh,
    'model_family': model_families,
    'model_fe': model_fe,
    'compute_budget': compute_flops,
    'log_compute': log_compute
})

# Create context bins for task FE
df_nh['context_bin'] = pd.cut(
    df_nh['context_length'],
    bins=[0, 10000, 50000, 100000, 500000, 2000000],
    labels=['4K-8K', '16K-32K', '64K-128K', '256K-512K', '1M']
)
df_nh['log_accuracy'] = np.log(df_nh['accuracy'])

print("=== Needle-in-Haystack Data Summary ===")
print(df_nh.describe())

# ============================================================
# 2. Generate Depth-Accuracy Data (150 observations)
# ============================================================
n_depth = 150
chain_depth = np.random.choice(range(1, 11), n_depth)
compute_depth = np.random.lognormal(mean=15, sigma=1.2, size=n_depth)
log_compute_depth = np.log(compute_depth)

model_depth = np.random.choice(['Reasoning-v1', 'Reasoning-v2', 'Reasoning-v3'], n_depth)
model_depth_fe = {'Reasoning-v1': 0.0, 'Reasoning-v2': 0.05, 'Reasoning-v3': 0.10}

# Accuracy: quadratic in chain depth (concave decreasing)
beta0 = 0.85
beta1 = -0.03
beta2 = -0.008
gamma_depth = 0.02

depth_fe = np.array([model_depth_fe[m] for m in model_depth])
noise_depth = np.random.normal(0, 0.025, n_depth)
accuracy_depth = (
    beta0 + beta1 * chain_depth + beta2 * (chain_depth**2) +
    gamma_depth * log_compute_depth + depth_fe + noise_depth
)
accuracy_depth = np.clip(accuracy_depth, 0.1, 1.0)

df_depth = pd.DataFrame({
    'chain_depth': chain_depth,
    'chain_depth_sq': chain_depth**2,
    'accuracy': accuracy_depth,
    'compute_budget': compute_depth,
    'log_compute': log_compute_depth,
    'model_family': model_depth
})

print("\n=== Depth-Accuracy Data Summary ===")
print(df_depth.describe())

# ============================================================
# 3. Generate Signal Overload Data (180 observations)
# ============================================================
n_signal = 180
K_signals = np.random.choice(range(1, 21), n_signal)
A_categories = np.random.choice(['Easy', 'Medium', 'Hard'], n_signal)
A_fe_signal = {'Easy': 0.90, 'Medium': 0.65, 'Hard': 0.35}

alpha_sig = 0.95
beta1_sig = -0.015
beta2_sig = -0.08

A_fe_vals = np.array([A_fe_signal[a] for a in A_categories])
noise_signal = np.random.normal(0, 0.02, n_signal)
accuracy_signal = (
    alpha_sig + beta1_sig * K_signals + beta2_sig * np.log(K_signals + 1) +
    A_fe_vals + noise_signal
)
accuracy_signal = np.clip(accuracy_signal, 0.05, 1.0)

df_signal = pd.DataFrame({
    'K_signals': K_signals,
    'log_K': np.log(K_signals + 1),
    'accuracy': accuracy_signal,
    'task_difficulty': A_categories,
    'A_fe': A_fe_vals
})

print("\n=== Signal Overload Data Summary ===")
print(df_signal.describe())

# ============================================================
# 4. Run Regressions
# ============================================================

# ---- Prediction 7.1: Depth-Accuracy Tradeoff ----
reg71_1 = smf.ols('accuracy ~ chain_depth + chain_depth_sq', data=df_depth).fit(cov_type='HC1')
reg71_2 = smf.ols('accuracy ~ chain_depth + chain_depth_sq + log_compute', data=df_depth).fit(cov_type='HC1')
reg71_3 = smf.ols('accuracy ~ chain_depth + chain_depth_sq + log_compute + C(model_family)', data=df_depth).fit(cov_type='HC1')
reg71_4 = smf.ols('accuracy ~ chain_depth + chain_depth_sq + log_compute + chain_depth*log_compute + C(model_family)', data=df_depth).fit(cov_type='HC1')

print("\n=== Prediction 7.1: Depth-Accuracy Tradeoff ===")
print(f"Spec (1) R2: {reg71_1.rsquared:.3f}, N={int(reg71_1.nobs)}")
print(f"Spec (2) R2: {reg71_2.rsquared:.3f}, N={int(reg71_2.nobs)}")
print(f"Spec (3) R2: {reg71_3.rsquared:.3f}, N={int(reg71_3.nobs)}")
print(f"Spec (4) R2: {reg71_4.rsquared:.3f}, N={int(reg71_4.nobs)}")

# ---- Prediction 7.2: Signal Overload ----
reg72_1 = smf.ols('accuracy ~ K_signals', data=df_signal).fit(cov_type='HC1')
reg72_2 = smf.ols('accuracy ~ K_signals + log_K', data=df_signal).fit(cov_type='HC1')
reg72_3 = smf.ols('accuracy ~ K_signals + log_K + C(task_difficulty)', data=df_signal).fit(cov_type='HC1')
reg72_4 = smf.ols('accuracy ~ K_signals + I(K_signals**2) + log_K + C(task_difficulty)', data=df_signal).fit(cov_type='HC1')

print("\n=== Prediction 7.2: Signal Overload ===")
print(f"Spec (1) R2: {reg72_1.rsquared:.3f}, N={int(reg72_1.nobs)}")
print(f"Spec (2) R2: {reg72_2.rsquared:.3f}, N={int(reg72_2.nobs)}")
print(f"Spec (3) R2: {reg72_3.rsquared:.3f}, N={int(reg72_3.nobs)}")
print(f"Spec (4) R2: {reg72_4.rsquared:.3f}, N={int(reg72_4.nobs)}")

# ---- Table 7: Gamma Identification ----
reg73_1 = smf.ols('accuracy ~ log_context_length + log_compute', data=df_nh).fit(cov_type='HC1')
reg73_2 = smf.ols('accuracy ~ log_context_length + log_compute + C(model_family)', data=df_nh).fit(cov_type='HC1')
reg73_3 = smf.ols('accuracy ~ log_context_length + log_compute + C(model_family) + C(context_bin)', data=df_nh).fit(cov_type='HC1')
reg73_4 = smf.ols('log_accuracy ~ log_context_length + log_compute + C(model_family)', data=df_nh).fit(cov_type='HC1')

print("\n=== Table 7: Gamma Identification ===")
print(f"Spec (1) R2: {reg73_1.rsquared:.3f}, N={int(reg73_1.nobs)}")
print(f"Spec (2) R2: {reg73_2.rsquared:.3f}, N={int(reg73_2.nobs)}")
print(f"Spec (3) R2: {reg73_3.rsquared:.3f}, N={int(reg73_3.nobs)}")
print(f"Spec (4) R2: {reg73_4.rsquared:.3f}, N={int(reg73_4.nobs)}")

# ============================================================
# 5. LaTeX Table Generation
# ============================================================

def get_stars(pval):
    if pval < 0.01:
        return "$^{***}$"
    elif pval < 0.05:
        return "$^{**}$"
    elif pval < 0.1:
        return "$^{*}$"
    return ""


def format_coef(coef, se, pval):
    stars = get_stars(pval)
    return f"{coef:.4f}{stars}", f"({se:.4f})"


def _reg_row(regs, label):
    """Build a LaTeX table row for a coefficient across 4 specs."""
    cells = [label.replace('_', ' ')]
    for reg in regs:
        if label in reg.params.index:
            c, s, p = reg.params[label], reg.bse[label], reg.pvalues[label]
            cs, _ = format_coef(c, s, p)
            cells.append(cs)
        else:
            cells.append("")
    return " & ".join(cells) + " \\\\"


def _se_row(regs, label):
    """Build a LaTeX table row for standard errors across 4 specs."""
    cells = [""]
    for reg in regs:
        if label in reg.params.index:
            _, ss = format_coef(reg.params[label], reg.bse[label], reg.pvalues[label])
            cells.append(ss)
        else:
            cells.append("")
    return " & ".join(cells) + " \\\\"


def generate_latex_tables():
    """Generate LaTeX regression tables and save to file."""
    lines = []
    lines.append("% Simulated Regression Tables")
    lines.append("% Generated by simulation_regression.py")
    lines.append("")

    def emit_table(title, caption, regs, varnames):
        lines.append(f"% Table: {title}")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append(" & (1) & (2) & (3) & (4) \\\\")
        lines.append("\\midrule")
        for vn in varnames:
            lines.append(_reg_row(regs, vn))
            lines.append(_se_row(regs, vn))
        r2_cells = [f"{r.rsquared:.3f}" for r in regs]
        lines.append(f"R$^2$ & {' & '.join(r2_cells)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    emit_table(
        "Prediction 7.1 — Depth-Accuracy Tradeoff",
        "Prediction 7.1: Depth--Accuracy Tradeoff",
        [reg71_1, reg71_2, reg71_3, reg71_4],
        ["chain_depth", "chain_depth_sq"],
    )
    emit_table(
        "Prediction 7.2 — Signal Overload",
        "Prediction 7.2: Signal Overload",
        [reg72_1, reg72_2, reg72_3, reg72_4],
        ["K_signals", "log_K"],
    )
    emit_table(
        "Table 7 — Gamma Identification",
        "Table 7: Gamma Identification",
        [reg73_1, reg73_2, reg73_3, reg73_4],
        ["log_context_length", "log_compute"],
    )

    tex = "\n".join(lines)
    with open("regression_tables.tex", "w") as f:
        f.write(tex)
    print("\nLaTeX tables saved to: regression_tables.tex")


generate_latex_tables()

print("\n" + "="*60)
print("All regressions completed successfully!")
print("="*60)
