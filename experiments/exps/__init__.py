# Auto-import all experiment modules to trigger @register decorators.
# pylint: disable=unused-import

import importlib
import sys
import warnings

_MODULE_NAMES = [
    "exp01_depth_accuracy",
    "exp02_front_loading",
    "exp03_exponential_decay",
    "exp04_signal_overload",
    "exp05_heterogeneity",
    "exp06_cost_irrelevance",
    "exp07_budget_depth",
    "exp08_task_complexity",
    "exp09_heterogeneous_agents",
    "exp10_human_ai_hybrid",
    "exp11_iv_simulation",
    "exp12_lab_protocol",
    "exp13_rate_distortion",
    "exp14_memory_capacity",
    "exp15_skip_connections",
    "exp16_parallel_branches",
    "exp17_memory_augmented",
    "exp18_backloading_boundary",
    "exp19_sufficient_conditions",
    "sup01_precision_decay",
    "sup02_rho_vs_budget",
    "sup03_profit_function",
    "sup04_optimal_depth_vs_budget",
    "sup05_retention_marginal_loss",
    "sup06_ghm_benchmark",
    "sup07_sensitivity_analysis",
]

for _mod_name in _MODULE_NAMES:
    try:
        importlib.import_module(f".{_mod_name}", package=__name__)
    except ImportError as _e:
        warnings.warn(
            f"Failed to import {_mod_name}: {_e}. "
            "Some experiments may be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
