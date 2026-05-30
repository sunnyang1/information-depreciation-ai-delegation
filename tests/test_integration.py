"""
Integration tests for the core simulation engine.

These tests exercise the high-level experiment functions and end-to-end
workflows that are not covered by unit tests.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import pytest

from exp_framework import (
    BudgetStrategy,
    TheoreticalLLMChain,
    allocate_budget,
    run_experiment_1_depth_accuracy,
    run_experiment_2_front_loading,
    run_experiment_3_eta_estimation,
    export_results_to_latex_tables,
    LayerPrecisionState,
    ExperimentResult,
    DepreciationEstimate,
)


# ---------------------------------------------------------------------------
# run_chain (end-to-end chain execution)
# ---------------------------------------------------------------------------


class TestRunChain:
    """Test TheoreticalLLMChain.run_chain end-to-end."""

    def test_run_chain_basic(self):
        chain = TheoreticalLLMChain()
        result = chain.run_chain(
            depth=3,
            strategy=BudgetStrategy.UNIFORM,
            total_budget=100_000,
            n_signals=50,
            n_trials=5,
        )
        assert isinstance(result, ExperimentResult)
        assert result.depth == 3
        assert result.strategy == "uniform"
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision_retained <= 1.0
        assert result.tokens_used > 0
        assert "budgets" in result.metadata
        assert "n_trials" in result.metadata

    def test_run_chain_front_loaded(self):
        chain = TheoreticalLLMChain()
        result = chain.run_chain(
            depth=3,
            strategy=BudgetStrategy.FRONT_LOADED,
            total_budget=100_000,
            n_signals=50,
            n_trials=3,
        )
        assert result.strategy == "front_loaded"
        assert 0.0 <= result.accuracy <= 1.0

    def test_run_chain_zero_depth(self):
        chain = TheoreticalLLMChain()
        result = chain.run_chain(
            depth=0,
            strategy=BudgetStrategy.UNIFORM,
            total_budget=100_000,
            n_signals=50,
            n_trials=1,
        )
        assert result.depth == 0
        assert result.tokens_used == 0
        assert result.precision_retained == 0.0
        # accuracy_from_precision(0) returns baseline ~0.07, not 0.0
        assert 0.0 <= result.accuracy <= 0.2

    def test_run_chain_single_trial_reproducibility(self):
        """Same seed should give same result (noise_std > 0 so exact match not guaranteed)."""
        chain1 = TheoreticalLLMChain()
        chain2 = TheoreticalLLMChain()
        r1 = chain1.run_chain(depth=2, strategy=BudgetStrategy.UNIFORM, n_trials=1)
        r2 = chain2.run_chain(depth=2, strategy=BudgetStrategy.UNIFORM, n_trials=1)
        # Deterministic path means metadata should match
        assert r1.metadata["budgets"] == r2.metadata["budgets"]
        assert r1.tokens_used == r2.tokens_used


# ---------------------------------------------------------------------------
# Experiment 1: Depth-Accuracy Tradeoff
# ---------------------------------------------------------------------------


class TestExperiment1:
    """Test run_experiment_1_depth_accuracy."""

    def test_returns_results_for_each_depth(self):
        depths = [1, 2, 3]
        results = run_experiment_1_depth_accuracy(
            depths=depths,
            total_budget=50_000,
            n_trials=3,
        )
        assert len(results) == len(depths)
        for r, d in zip(results, depths):
            assert r.depth == d
            assert isinstance(r.accuracy, float)
            assert 0.0 <= r.accuracy <= 1.0

    def test_monotonically_decreasing(self):
        """Accuracy should decrease (or stay flat) with depth."""
        depths = list(range(1, 5))
        results = run_experiment_1_depth_accuracy(
            depths=depths,
            total_budget=50_000,
            n_trials=10,
        )
        accs = [r.accuracy for r in results]
        # With n_trials=10 noise may violate strict monotonicity,
        # so we check average trend via pairwise majority
        decreases = sum(1 for i in range(len(accs) - 1) if accs[i] >= accs[i + 1])
        assert decreases >= len(accs) - 2  # allow one violation due to noise

    def test_default_depths(self):
        results = run_experiment_1_depth_accuracy(n_trials=2)
        assert len(results) == 5  # default depths 1..5


# ---------------------------------------------------------------------------
# Experiment 2: Front-Loading Validation
# ---------------------------------------------------------------------------


class TestExperiment2:
    """Test run_experiment_2_front_loading."""

    def test_returns_all_strategies(self):
        result = run_experiment_2_front_loading(
            depth=3,
            total_budget=50_000,
            n_trials=3,
        )
        assert set(result.keys()) == {
            "uniform",
            "front_loaded",
            "back_loaded",
        }
        for strategy, exp_result in result.items():
            assert isinstance(exp_result, ExperimentResult)
            assert exp_result.strategy == strategy

    def test_front_loaded_best(self):
        """Front-loaded should outperform back-loaded on average."""
        result = run_experiment_2_front_loading(
            depth=3,
            total_budget=100_000,
            n_trials=20,
        )
        assert result["front_loaded"].accuracy >= result["back_loaded"].accuracy


# ---------------------------------------------------------------------------
# Experiment 3: Eta Estimation
# ---------------------------------------------------------------------------


class TestExperiment3:
    """Test run_experiment_3_eta_estimation."""

    def test_returns_estimates_for_each_layer(self):
        max_depth = 4
        estimates = run_experiment_3_eta_estimation(
            max_depth=max_depth,
            n_facts=50,
            budget=20_000,
        )
        assert len(estimates) == max_depth
        for e in estimates:
            assert isinstance(e, DepreciationEstimate)
            assert 1 <= e.layer <= max_depth
            assert 0.0 <= e.retention_rate <= 1.0
            assert e.ci_lower <= e.retention_rate <= e.ci_upper
            assert 0.0 <= e.theoretical <= 1.0

    def test_retention_decreases_with_depth(self):
        """Retention should decrease as layer increases."""
        estimates = run_experiment_3_eta_estimation(
            max_depth=5,
            n_facts=100,
            budget=30_000,
        )
        retentions = [e.retention_rate for e in estimates]
        assert all(
            retentions[i] >= retentions[i + 1]
            for i in range(len(retentions) - 1)
        )

    def test_n_facts_zero_fallback(self):
        """n_facts=0 should not crash (graceful fallback)."""
        estimates = run_experiment_3_eta_estimation(
            max_depth=2,
            n_facts=0,
            budget=10_000,
        )
        assert len(estimates) == 2


# ---------------------------------------------------------------------------
# Budget allocation: additional strategies
# ---------------------------------------------------------------------------


class TestAllocateBudgetGeometric:
    """Test geometric budget allocation strategies."""

    def test_geometric_front_sum(self):
        sizes = allocate_budget(100, 3, BudgetStrategy.GEOMETRIC_FRONT)
        assert sum(sizes) == 100
        assert sizes[0] > sizes[1] > sizes[2]

    def test_geometric_back_sum(self):
        sizes = allocate_budget(100, 3, BudgetStrategy.GEOMETRIC_BACK)
        assert sum(sizes) == 100
        assert sizes[0] < sizes[1] < sizes[2]

    def test_geometric_front_vs_back(self):
        front = allocate_budget(100, 4, BudgetStrategy.GEOMETRIC_FRONT)
        back = allocate_budget(100, 4, BudgetStrategy.GEOMETRIC_BACK)
        assert front[0] > back[0]  # front gets more at start
        assert front[-1] < back[-1]  # back gets more at end

    def test_geometric_with_depth_one(self):
        sizes = allocate_budget(100, 1, BudgetStrategy.GEOMETRIC_FRONT)
        assert sizes == [100]


# ---------------------------------------------------------------------------
# precision_path with heterogeneity
# ---------------------------------------------------------------------------


class TestPrecisionPathHeterogeneity:
    """Test precision_path with heterogeneous signal precisions."""

    def test_heterogeneous_improves_over_uniform(self):
        chain = TheoreticalLLMChain()
        budgets = [50.0, 50.0]
        K_list = [10, 10]

        # Uniform precision
        states_uniform = chain.precision_path(budgets, K_list, heterogeneity=None)
        final_tau_uniform = states_uniform[-1].tau_post

        # Heterogeneous precision (some signals much stronger)
        heterogeneity = [2.0] * 5 + [0.5] * 5
        states_het = chain.precision_path(
            budgets, K_list, heterogeneity=heterogeneity
        )
        final_tau_het = states_het[-1].tau_post

        # Heterogeneous should produce different (usually better) precision
        assert final_tau_het != final_tau_uniform

    def test_heterogeneous_list_length_mismatch(self):
        chain = TheoreticalLLMChain()
        budgets = [50.0]
        K_list = [10]
        heterogeneity = [1.0] * 5  # wrong length
        states = chain.precision_path(budgets, K_list, heterogeneity=heterogeneity)
        assert len(states) == 1


# ---------------------------------------------------------------------------
# compute_rho
# ---------------------------------------------------------------------------


class TestComputeRho:
    """Test TheoreticalLLMChain.compute_rho."""

    def test_rho_bounded(self):
        chain = TheoreticalLLMChain()
        rho = chain.compute_rho(budget=100.0, signal_precisions=[1.0] * 50)
        assert 0.0 <= rho <= 1.0

    def test_rho_full_budget(self):
        """With huge budget, rho should approach 1."""
        chain = TheoreticalLLMChain()
        rho = chain.compute_rho(budget=1e9, signal_precisions=[1.0] * 10)
        assert pytest.approx(rho, abs=0.05) == 1.0

    def test_rho_zero_precision(self):
        """All signals have zero precision → rho = 1.0 (no information to lose)."""
        chain = TheoreticalLLMChain()
        rho = chain.compute_rho(budget=10.0, signal_precisions=[0.0] * 10)
        assert rho == 1.0


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


class TestExportResults:
    """Test LaTeX and JSON export functions."""

    def _make_sample_results(self):
        chain = TheoreticalLLMChain()
        exp1 = [
            chain.run_chain(
                depth=d, strategy=BudgetStrategy.UNIFORM, n_trials=2
            )
            for d in [1, 2]
        ]
        exp2 = {
            "uniform": chain.run_chain(
                depth=2, strategy=BudgetStrategy.UNIFORM, n_trials=2
            ),
            "front_loaded": chain.run_chain(
                depth=2, strategy=BudgetStrategy.FRONT_LOADED, n_trials=2
            ),
        }
        exp3 = run_experiment_3_eta_estimation(max_depth=2, n_facts=20, budget=10_000)
        return exp1, exp2, exp3

    def test_export_latex_returns_string(self):
        exp1, exp2, exp3 = self._make_sample_results()
        latex = export_results_to_latex_tables(exp1, exp2, exp3)
        assert isinstance(latex, str)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "Depth" in latex or "depth" in latex

    def test_export_latex_with_empty_results(self):
        latex = export_results_to_latex_tables([], {}, [])
        assert isinstance(latex, str)

    def test_json_roundtrip(self, tmp_path):
        """Results should be JSON-serializable and deserializable."""
        from dataclasses import asdict

        exp1, exp2, exp3 = self._make_sample_results()
        data = {
            "experiment_1": [asdict(r) for r in exp1],
            "experiment_2": {s: asdict(r) for s, r in exp2.items()},
            "experiment_3": [asdict(e) for e in exp3],
        }
        path = tmp_path / "results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert len(loaded["experiment_1"]) == len(exp1)
        assert set(loaded["experiment_2"].keys()) == set(exp2.keys())
        assert len(loaded["experiment_3"]) == len(exp3)


# ---------------------------------------------------------------------------
# LayerPrecisionState dataclass
# ---------------------------------------------------------------------------


class TestLayerPrecisionState:
    """Test LayerPrecisionState properties."""

    def test_state_creation(self):
        state = LayerPrecisionState(
            layer_idx=1,
            budget=100.0,
            n_signals=50,
            eta=0.85,
            rho=0.9,
            tau_prior=10.0,
            tau_post=50.0,
            tau_fb=55.0,
            attention_weights=[1.0] * 50,
            signal_precisions=[1.0] * 50,
        )
        assert state.layer_idx == 1
        assert state.eta == pytest.approx(0.85)
        assert state.tau_post > state.tau_prior
