"""Task and evaluator registries for deployment entrypoints."""

from __future__ import annotations

from typing import Dict
from ..controllers.controller_cfg import ControllerCfg, EvaluatorCfg

# Registry mapping task name -> dict with key: cfg
_TASK_REGISTRY: Dict[str, ControllerCfg] = {}
_EVALUATOR_REGISTRY: Dict[str, EvaluatorCfg] = {}


def register_task(name: str, cfg: ControllerCfg) -> None:
    """Register a task by name with its configuration object.

    The registry stores only the cfg; deploy will decide how to run it.
    """
    if name in _TASK_REGISTRY:
        raise KeyError(f"Task '{name}' already registered")
    _TASK_REGISTRY[name] = cfg


def get_task(name: str) -> ControllerCfg:
    """Return task configuration by registered name.

    Args:
        name: Task key used during registration.

    Returns:
        Registered controller configuration.

    """
    return _TASK_REGISTRY[name]


def list_tasks() -> Dict[str, ControllerCfg]:
    """Return a copy of all registered tasks.

    Returns:
        Mapping from task name to task configuration.

    """
    return dict(_TASK_REGISTRY)


def register_evaluator(name: str, cfg: EvaluatorCfg) -> None:
    """Register an evaluator by name with its configuration object.

    The registry stores only the cfg; deploy will decide how to run it.
    """
    if name in _EVALUATOR_REGISTRY:
        raise KeyError(f"Evaluator '{name}' already registered")
    _EVALUATOR_REGISTRY[name] = cfg


def get_evaluator(name: str) -> EvaluatorCfg:
    """Return evaluator configuration by registered name.

    Args:
        name: Evaluator key used during registration.

    Returns:
        Registered evaluator configuration.

    """
    return _EVALUATOR_REGISTRY[name]


def list_evaluators() -> Dict[str, EvaluatorCfg]:
    """Return a copy of all registered evaluators.

    Returns:
        Mapping from evaluator name to evaluator configuration.

    """
    return dict(_EVALUATOR_REGISTRY)


__all__ = [
    "register_task", "get_task", "list_tasks",
    "register_evaluator", "get_evaluator", "list_evaluators"
]
