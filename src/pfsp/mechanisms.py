"""Glue: high-level mechanism design <-> concrete scheduler factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

from .design import MechanismDesign, get_design
from .scheduler import FixedScheduler, QLearningScheduler

class SchedulerProtocol:
    def start_iter(self) -> None: ...
    def next_operator(self) -> str | None: ...
    def update(self, op: str, reward: float) -> None: ...

SchedulerFactory = Callable[[Sequence[str], Mapping[str, object]], SchedulerProtocol]

@dataclass(frozen=True)
class MechanismSpec:
    design: MechanismDesign
    factory: SchedulerFactory

def _build_fixed(operators: Sequence[str], _: Mapping[str, object]) -> SchedulerProtocol:
    return FixedScheduler(operators)

def _build_qlearn(operators: Sequence[str], options: Mapping[str, object]) -> SchedulerProtocol:
    # CLI compat: map p_min->epsilon, learning_rate->alpha
    window_size = int(options.get("window_size", 50))
    epsilon = float(options.get("p_min", 0.10))
    alpha = float(options.get("learning_rate", 0.30))
    gamma = float(options.get("gamma", 0.60))
    episode_len = int(options.get("episode_len", 10))
    return QLearningScheduler(
        operators,
        window_size=window_size,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        episode_len=episode_len,
    )

MECHANISMS: Dict[str, MechanismSpec] = {
    "fixed":   MechanismSpec(design=get_design("fixed"),   factory=_build_fixed),
    "adaptive": MechanismSpec(design=get_design("adaptive"), factory=_build_qlearn),  # key kept for CLI
}

def available_mechanisms() -> Dict[str, str]:
    return {k: spec.design.identifier for k, spec in MECHANISMS.items()}

def get_mechanism(key: str) -> MechanismSpec:
    if key not in MECHANISMS:
        raise ValueError(f"Unknown mechanism '{key}'. Available: {', '.join(sorted(MECHANISMS))}")
    return MECHANISMS[key]

def build_scheduler(mechanism: str, operators: Sequence[str], options: Mapping[str, object] | None = None) -> SchedulerProtocol:
    spec = get_mechanism(mechanism)
    return spec.factory(operators, options or {})

def normalise_mechanism_options(mechanism: str, options: MutableMapping[str, object]) -> MutableMapping[str, object]:
    # all options accepted by qlearn; nothing to strip for fixed
    if mechanism == "fixed":
        options.pop("window_size", None)
        options.pop("p_min", None)
        options.pop("learning_rate", None)
        options.pop("gamma", None)
        options.pop("episode_len", None)
    return options
