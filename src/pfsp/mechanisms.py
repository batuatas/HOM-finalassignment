"""Utilities for configuring operator scheduling mechanisms.

The final assignment expects two baseline mechanisms:

* ``fixed`` – a deterministic Variable Neighbourhood Descent sweep where the
  neighbourhoods are explored in a fixed order.  This corresponds to
  Mechanism 1A in the rubric.
* ``adaptive`` – an adaptive probability matching mechanism that assigns
  credits to operators based on recent performance and samples operators
  proportionally to the credit.  This matches Mechanism 2A.

This module centralises the logic for building schedulers so that the rest of
the codebase can treat mechanisms as simple configuration values.  New
mechanisms can be registered by extending :data:`MECHANISMS` with a new
``MechanismConfig`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

from .scheduler import AdaptiveScheduler, FixedScheduler


class SchedulerProtocol:
    """Protocol implemented by all schedulers used by the metaheuristic."""

    def start_iter(self) -> None:  # pragma: no cover - behavioural interface
        ...

    def next_operator(self) -> str | None:  # pragma: no cover - behavioural interface
        ...

    def update(self, op: str, reward: float) -> None:  # pragma: no cover - behavioural interface
        ...


SchedulerFactory = Callable[[Sequence[str], Mapping[str, object]], SchedulerProtocol]


@dataclass(frozen=True)
class MechanismConfig:
    """Container describing a scheduling mechanism.

    Attributes
    ----------
    name:
        Short identifier used from the command line (e.g. ``"fixed"``).
    description:
        Human readable summary displayed in help messages and reports.
    factory:
        Callable that builds a scheduler for a given list of operators.  The
        callable receives the operator names and a mapping of keyword options
        extracted from user input.
    """

    name: str
    description: str
    factory: SchedulerFactory


def _build_fixed(operators: Sequence[str], _: Mapping[str, object]) -> SchedulerProtocol:
    return FixedScheduler(operators)


def _build_adaptive(operators: Sequence[str], options: Mapping[str, object]) -> SchedulerProtocol:
    window_size = int(options.get("window_size", 50))
    p_min = float(options.get("p_min", 0.1))
    return AdaptiveScheduler(operators, window_size=window_size, p_min=p_min)


MECHANISMS: Dict[str, MechanismConfig] = {
    "fixed": MechanismConfig(
        name="fixed",
        description="Deterministic VND sweep (Mechanism 1A)",
        factory=_build_fixed,
    ),
    "adaptive": MechanismConfig(
        name="adaptive",
        description="Adaptive probability matching (Mechanism 2A)",
        factory=_build_adaptive,
    ),
}


def available_mechanisms() -> Dict[str, str]:
    """Return mapping of mechanism name to human readable description."""

    return {name: config.description for name, config in MECHANISMS.items()}


def build_scheduler(
    mechanism: str,
    operators: Sequence[str],
    options: Mapping[str, object] | None = None,
) -> SchedulerProtocol:
    """Create a scheduler instance for the requested mechanism.

    Parameters
    ----------
    mechanism:
        Key identifying the mechanism to instantiate.
    operators:
        Sequence of operator names understood by the scheduler.
    options:
        Optional mapping of additional keyword arguments.  Only options
        relevant to the chosen mechanism are used; unknown keys are ignored.
    """

    if mechanism not in MECHANISMS:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. Available mechanisms: {', '.join(MECHANISMS)}"
        )
    config = MECHANISMS[mechanism]
    opts: Mapping[str, object] = options or {}
    return config.factory(operators, opts)


def normalise_mechanism_options(
    mechanism: str,
    options: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Filter option mapping so that irrelevant keys are discarded.

    This is mainly used by the CLI scripts to ensure that options intended for
    the adaptive mechanism (such as ``window_size`` and ``p_min``) are not
    mistakenly propagated to other mechanisms.
    """

    if mechanism == "adaptive":
        return options
    # Remove adaptive specific options for deterministic mechanisms
    options.pop("window_size", None)
    options.pop("p_min", None)
    return options

