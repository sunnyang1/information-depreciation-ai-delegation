"""
Unit tests for the core simulation engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import pytest

from exp_framework import (
    ETA_BAR,
    ETA_UNDERLINE,
    BudgetStrategy,
    TheoreticalLLMChain,
    allocate_budget,
)


class TestComputeEta:
    """Test endogenous preservation rate."""

    def test_eta_bounds(self):
        chain = TheoreticalLLMChain()
        eta = chain.compute_eta(budget=10.0, n_signals=100)
        assert ETA_UNDERLINE <= eta <= ETA_BAR

    def test_eta_increases_with_budget(self):
        chain = TheoreticalLLMChain()
        eta_small = chain.compute_eta(budget=1.0, n_signals=100)
        eta_large = chain.compute_eta(budget=1000.0, n_signals=100)
        assert eta_small < eta_large

    def test_eta_saturates(self):
        chain = TheoreticalLLMChain()
        eta = chain.compute_eta(budget=1e9, n_signals=1)
        assert pytest.approx(eta, abs=0.01) == ETA_BAR

    def test_zero_signals_fallback(self):
        chain = TheoreticalLLMChain()
        eta = chain.compute_eta(budget=100.0, n_signals=0)
        assert ETA_UNDERLINE <= eta <= ETA_BAR


class TestSolveAttentionAllocation:
    """Test optimal attention allocation."""

    def test_fully_funded(self):
        chain = TheoreticalLLMChain()
        weights, _, tau = chain.solve_attention_allocation(
            budget=100.0, signal_precisions=[1.0] * 50
        )
        assert all(w == 1.0 for w in weights)
        assert tau == pytest.approx(50.0)

    def test_under_funded(self):
        chain = TheoreticalLLMChain()
        weights, _, tau = chain.solve_attention_allocation(
            budget=10.0, signal_precisions=[1.0] * 50
        )
        assert sum(weights) == pytest.approx(10.0, abs=0.01)
        assert 0 < tau < 50.0

    def test_empty_signals(self):
        chain = TheoreticalLLMChain()
        weights, lam, tau = chain.solve_attention_allocation(budget=10.0, signal_precisions=[])
        assert weights == []
        assert tau == 0.0


class TestPrecisionPath:
    """Test recursive precision dynamics."""

    def test_monotonic_depth(self):
        chain = TheoreticalLLMChain()
        states = chain.precision_path(
            budgets=[100.0, 100.0, 100.0],
            n_signals_per_layer=[50, 50, 50],
        )
        assert len(states) == 3
        # rho should be in [0, 1]
        for s in states:
            assert 0.0 <= s.rho <= 1.0

    def test_deeper_chain_lower_retention(self):
        chain = TheoreticalLLMChain()
        states_1 = chain.precision_path(budgets=[100.0], n_signals_per_layer=[50])
        states_3 = chain.precision_path(
            budgets=[100.0] * 3, n_signals_per_layer=[50] * 3
        )
        assert states_3[-1].rho <= states_1[-1].rho


class TestAllocateBudget:
    """Test budget allocation strategies."""

    def test_uniform_sum(self):
        sizes = allocate_budget(100, 3, BudgetStrategy.UNIFORM)
        assert sum(sizes) == 100

    def test_front_loaded_decreasing(self):
        sizes = allocate_budget(100, 3, BudgetStrategy.FRONT_LOADED)
        assert sizes[0] > sizes[1] > sizes[2]

    def test_back_loaded_increasing(self):
        sizes = allocate_budget(100, 3, BudgetStrategy.BACK_LOADED)
        assert sizes[0] < sizes[1] < sizes[2]

    def test_zero_depth(self):
        sizes = allocate_budget(100, 0, BudgetStrategy.UNIFORM)
        assert sizes == []


class TestAccuracyFromPrecision:
    """Test accuracy mapping."""

    def test_bounded(self):
        chain = TheoreticalLLMChain()
        acc = chain.accuracy_from_precision(tau=150.0, task_difficulty=0.0)
        assert 0.0 <= acc <= 1.0

    def test_increasing_in_tau(self):
        chain = TheoreticalLLMChain()
        acc_low = chain.accuracy_from_precision(tau=50.0, task_difficulty=0.0)
        acc_high = chain.accuracy_from_precision(tau=300.0, task_difficulty=0.0)
        assert acc_low < acc_high


class TestGammaValidation:
    """Test gamma parameter validation."""

    def test_gamma_out_of_range_raises(self):
        with pytest.raises(ValueError, match="gamma must be in"):
            TheoreticalLLMChain(gamma=1.5)
        with pytest.raises(ValueError, match="gamma must be in"):
            TheoreticalLLMChain(gamma=-0.1)
        with pytest.raises(ValueError, match="gamma must be in"):
            TheoreticalLLMChain(gamma=1.0)

    def test_gamma_near_one_warns_on_overflow(self):
        """gamma=0.99 should not crash (falls back to proportional in allocation)."""
        chain = TheoreticalLLMChain(gamma=0.99)
        weights, lam, tau = chain.solve_attention_allocation(10.0, [1.0] * 50)
        assert sum(weights) == pytest.approx(10.0, abs=0.01)
        assert tau >= 0.0


class TestGAttention:
    """Test g_attention edge cases."""

    def test_negative_alpha_returns_zero(self):
        chain = TheoreticalLLMChain()
        assert chain.g_attention(-1.0) == 0.0
        assert chain.g_attention(-0.5) == 0.0

    def test_zero_alpha(self):
        chain = TheoreticalLLMChain()
        assert chain.g_attention(0.0) == 0.0
