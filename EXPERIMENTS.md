# Experimental Design and Implementation

This document describes the complete experimental protocol for validating the seven theoretical predictions in the paper.

## Philosophy

All simulations use a **theory-faithful engine** (`TheoreticalLLMChain` in `exp_framework.py`) that derives every quantity from the paper's structural equations:

- **Assumption 2** (Eq. 4): Endogenous preservation rate `η(A)`
- **Assumption 3** (Eq. 8): Attention technology `g(α) = min(α^γ, 1)`
- **Proposition 1** (Eq. 13): Optimal attention allocation
- **Definition 3** (Eq. 16): Endogenous transmission factor `ρ`
- **Proposition 5** (Eq. 22): Recursive precision path `τ*_ℓ`

No ad-hoc accuracy formulas are used. Observable accuracy is mapped from precision via a calibrated sigmoid.

---

## Experiment 1: Depth–Accuracy Tradeoff

**Prediction**: Output accuracy is decreasing and concave in chain depth `L` (Prediction 7.1).

**Design**:
- Fix total budget `B = 200,000` tokens, uniform allocation across `L` layers.
- Vary `L ∈ {1, 2, 3, 4, 5}`.
- Task: simulated reading comprehension with task difficulty = 0.3.
- 100 trials per depth.

**Metrics**:
- Mean accuracy and standard deviation
- Final posterior precision `τ*_L`
- Quadratic fit: `Accuracy = β₀ + β₁L + β₂L²`; test `β₂ < 0`

**Expected Result**: Hump-shaped or monotonically decreasing accuracy. In our parameterization, accuracy peaks at `L = 2–3` and then declines.

---

## Experiment 2: Front-Loading Advantage

**Prediction**: Front-loaded budget allocation outperforms uniform and back-loaded (Proposition 4).

**Design**:
- Fix `B = 200,000` tokens and `L = 3`.
- Compare three strategies:
  - *Uniform*: `[67, 67, 66]` (effective attention units)
  - *Front-loaded*: `[115, 57, 28]`
  - *Back-loaded*: `[28, 57, 115]`
- 100 trials per strategy.

**Metrics**:
- Mean accuracy by strategy
- Pairwise approximate t-tests
- Ranking confirmation

**Expected Result**: `Front-loaded > Uniform > Back-loaded`.

**Why it works**: Early layers process raw signals with highest intrinsic precision. A unit of precision lost at layer 0 is compounded by all subsequent layers. Front-loading preserves more information early, creating better downstream inputs.

---

## Experiment 3: Exponential Decay of Information Retention

**Prediction**: Per-layer retention follows exponential decay `∏ρ_j ≈ η̄^ℓ` (Proposition 5).

**Design**:
- Insert `n = 100` unit-precision signals at layer 0.
- Build chains of length `ℓ ∈ {1, 2, 3, 4, 5}` with fixed budget per layer.
- Measure cumulative transmission factor `∏_{j=0}^{ℓ-1} ρ_j`.

**Metrics**:
- Retention rate with 95% binomial CI
- Log-linear fit: `ln(retention) = ln(η̂) · ℓ`
- `R²` of exponential fit

**Expected Result**: Excellent fit (`R² > 0.95`) with estimated `η̂` close to the calibrated average `η`.

---

## Experiment 4: Signal Overload

**Prediction**: For fixed context window, increasing the number of input signals `K` decreases per-signal output precision (Prediction 7.2).

**Design**:
- Fix effective budget `A = 30` attention units (`30,000` tokens).
- Vary `K ∈ {20, 50, 100, 200, 500}`.
- Test at depths `L ∈ {1, 3, 5}`.

**Metrics**:
- Per-signal transmission factor `ρ` (not total precision)
- Regress `ρ` on `K`; test negative slope

**Expected Result**: `ρ` decreases with `K`. When `K ≤ A`, all signals fully processed (`ρ = 1`). When `K > A`, attention must be selective and `ρ < 1`.

---

## Experiment 5: Heterogeneity Reduces Distortion

**Prediction**: Holding total precision constant, a more heterogeneous signal distribution yields lower aggregate distortion `Δ` (Prediction 7.3).

**Design**:
- Fix `L = 3`, `K = 100`, total precision `= 100`.
- Effective budget `= 20` (budget < K, so selective attention matters).
- Compare three distributions:
  - *Homogeneous*: all signals precision = 1.0
  - *Moderate heterogeneity*: Pareto-like with `α = 0.8`
  - *High heterogeneity*: Pareto-like with `α = 1.5`

**Metrics**:
- Aggregate distortion `Δ = 1 − ρ`
- Transmission factor `ρ`
- Gini coefficient of precision distribution
- Correlation: `Gini vs Δ` (expected positive), `Gini vs ρ` (expected negative)

**Expected Result**: Higher Gini → lower Δ → higher ρ. Selective attention concentrates on high-precision signals, preserving more total information.

---

## Experiment 6: Cost Irrelevance for Depth

**Prediction**: Reducing API cost `κ` does not increase optimal depth `L*` beyond a finite limit (Prediction 7.4).

**Design**:
- Fix effective budget `A = 80` (`80,000` tokens).
- Vary `κ ∈ {10⁻⁴, 10⁻⁵, 10⁻⁶, 10⁻⁷, 10⁻⁸, 10⁻⁹, 0}`.
- Compute net value `V(τ*_L) − Σ c(A_ℓ)` for `L = 1..10`.
- Optimal depth `L*` = argmax net value.

**Metrics**:
- `L*` as function of `κ`
- Convergence check: `L*` stabilizes as `κ → 0`

**Expected Result**: `L*` increases with lower `κ` but converges to a finite limit (e.g., 6–10). Even with `κ = 0`, finite depth is optimal because `τ*_L` eventually declines due to information depreciation (`η̄ < 1`).

---

## Experiment 7: Budget Expansion Increases Depth

**Prediction**: Larger context window `A` increases optimal depth `L*` (Prediction 7.5).

**Design**:
- Vary `A ∈ {4K, 16K, 32K, 64K, 128K, 256K, 512K}` tokens.
- Compute `L*` = argmax net value for each `A`.

**Metrics**:
- `L*` as function of `A`
- Monotonicity test

**Expected Result**: `L*` is increasing in `A` (with possible local non-monotonicity due to integer depth and parameter thresholds). Larger budgets reduce per-layer information loss, making deeper chains profitable at the margin.

---

## Relationship to Real LLM Experiments

The simulation engine is designed to be **structurally isomorphic** to real LLM chains:

| Simulation Concept | Real LLM Mapping |
|--------------------|-----------------|
| Attention budget `A_ℓ` | Context-window length (tokens) |
| Number of signals `K_ℓ` | Number of retrieved documents / code snippets |
| Attention weight `α_{ℓ,k}` | Fraction of context devoted to signal `k` |
| Preservation rate `η_ℓ` | Signal fidelity after tokenization + attention |
| Transmission factor `ρ_ℓ` | Effective precision ratio (measured by F1) |

To run with real models, replace `TheoreticalLLMChain` with `LLMChain` from `run_real_experiments.py`. The budget allocation and experimental design remain identical.

---

## Reproducibility

All experiments use fixed random seeds (`SEED = 42`). To reproduce:

```bash
cd experiments
python exp_framework.py   # Baseline experiments 1–3
python exp_advanced.py    # Advanced experiments 4–7
python visualize_results.py  # Generate figures
```

Results are saved to:
- `experiments/results/results.json` — baseline results
- `experiments/results/advanced_results.json` — advanced results
- `experiments/results/*.tex` — LaTeX tables
- `experiments/figures/*.png` — publication figures
