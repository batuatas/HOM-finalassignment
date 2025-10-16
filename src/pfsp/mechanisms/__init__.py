"""Operator scheduling mechanisms and registry for the PFSP metaheuristic.

This package exposes concrete mechanism implementations (Adaptive/Fixed)
and a small registry mapping mechanism keys to scheduler factories so the
rest of the codebase can build schedulers by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

from .adaptive import AdaptiveMechanism
from .base import Mechanism
from .factory import build_mechanism
from .fixed import FixedMechanism

from .design import MechanismDesign, get_design
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
class MechanismSpec:
    """Pair a high-level design description with a scheduler factory."""

    design: MechanismDesign
    factory: SchedulerFactory


def _build_fixed(operators: Sequence[str], _: Mapping[str, object]) -> SchedulerProtocol:
    return FixedScheduler(operators)


def _build_adaptive(operators: Sequence[str], options: Mapping[str, object]) -> SchedulerProtocol:
    window_size = int(options.get("window_size", 50))
    p_min = float(options.get("p_min", 0.1))
    learning_rate = float(options.get("learning_rate", 0.2))
    return AdaptiveScheduler(
        operators,
        window_size=window_size,
        p_min=p_min,
        learning_rate=learning_rate,
    )


MECHANISMS: Dict[str, MechanismSpec] = {
    key: MechanismSpec(design=get_design(key), factory=factory)
    for key, factory in {
        "fixed": _build_fixed,
        "adaptive": _build_adaptive,
    }.items()
}


def available_mechanisms() -> Dict[str, str]:
    """Return mapping of mechanism key to assignment identifier."""

    return {name: spec.design.identifier for name, spec in MECHANISMS.items()}


def get_mechanism(key: str) -> MechanismSpec:
    """Return the mechanism specification for ``key``."""

    if key not in MECHANISMS:
        raise ValueError(
            f"Unknown mechanism '{key}'. Available mechanisms: {', '.join(sorted(MECHANISMS))}"
        )
    return MECHANISMS[key]


def build_scheduler(
    mechanism: str,
    operators: Sequence[str],
    options: Mapping[str, object] | None = None,
) -> SchedulerProtocol:
    """Create a scheduler instance for the requested mechanism."""

    spec = get_mechanism(mechanism)
    opts: Mapping[str, object] = options or {}
    return spec.factory(operators, opts)


def normalise_mechanism_options(
    mechanism: str,
    options: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Filter option mapping so that irrelevant keys are discarded."""

    if mechanism == "adaptive":
        return options
    options.pop("window_size", None)
    options.pop("p_min", None)
    options.pop("learning_rate", None)
    return options


__all__ = [
    "AdaptiveMechanism",
    "FixedMechanism",
    "Mechanism",
    "build_mechanism",
    "MECHANISMS",
    "available_mechanisms",
    "get_mechanism",
    "build_scheduler",
    "normalise_mechanism_options",
]
