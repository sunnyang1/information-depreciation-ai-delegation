"""
Unified Experiment Registry

All experiments register themselves via the @register decorator.
The runner script discovers and executes registered experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional


@dataclass
class Experiment:
    id: str
    name: str
    description: str
    category: str
    run: Callable[[], Dict[str, Any]]
    depends_on: List[str] = field(default_factory=list)


_REGISTRY: Dict[str, Experiment] = {}


def register(
    id: str,  # pylint: disable=redefined-builtin
    name: str,
    description: str = "",
    category: str = "",
    depends_on: Optional[List[str]] = None,
):
    """Decorator to register an experiment function."""

    def decorator(func: Callable) -> Callable:
        _REGISTRY[id] = Experiment(
            id=id,
            name=name,
            description=description,
            category=category,
            run=func,
            depends_on=depends_on or [],
        )
        return func

    return decorator


def get_experiment(exp_id: str) -> Experiment:
    if exp_id not in _REGISTRY:
        raise KeyError(
            f"Experiment '{exp_id}' not found. Use list_experiments() to see available IDs."
        )
    return _REGISTRY[exp_id]


def list_experiments(category: Optional[str] = None) -> Dict[str, Experiment]:
    if category is None:
        return dict(_REGISTRY)
    return {k: v for k, v in _REGISTRY.items() if v.category == category}


def get_all_ids() -> List[str]:
    return list(_REGISTRY.keys())


def get_categories() -> List[str]:
    return sorted(set(v.category for v in _REGISTRY.values() if v.category))


def print_registry():
    print("=" * 70)
    print("Registered Experiments")
    print("=" * 70)
    for cat in get_categories():
        print(f"\n[{cat}]")
        for exp in _REGISTRY.values():
            if exp.category == cat:
                print(f"  {exp.id:20s}  {exp.name}")
                if exp.description:
                    print(f"                      {exp.description}")
    print("\n" + "=" * 70)


# test comment
# test comment
