# src/pfsp/design.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

@dataclass(frozen=True)
class MechanismDesign:
    key: str
    identifier: str
    objective: str
    operators: Sequence[str]
    scheduler: str
    parameters: Mapping[str, str] = field(default_factory=dict)
    notes: Sequence[str] = field(default_factory=tuple)

DESIGNS: Mapping[str, MechanismDesign] = {
    "fixed": MechanismDesign(
        key="fixed",
        identifier="Mechanism 1A",
        objective="Deterministic VND with relocate → swap → block; granular neighborhoods (critical-path focus, bounded distance).",
        operators=("relocate", "swap", "block"),
        scheduler=("Fixed order; restart sweep whenever an improving move is accepted. Deterministic."),
        parameters={
            "granular_window": "Max relocate/block distance from source index.",
            "critical_only":   "If true, only jobs in critical set are moved.",
            "critical_take_frac": "Fraction of jobs considered 'critical'.",
            "block_lengths":   "Block sizes for block reinsertion.",
        },
        notes=("NEH initialization; IG destroy–repair; occasional ruin–recreate; deadline-aware.",),
    ),
    "adaptive": MechanismDesign(
        key="adaptive",
        identifier="Mechanism 2B (Q-learning)",
        objective="Episode-based ε-greedy Q-learning to select operator online with richer state and softmax tie-break.",
        operators=("relocate", "swap", "block"),
        scheduler=(
            "State = (improvement bin, stagnation bin, search-phase bin). "
            "ε-greedy with mild decay per episode; optimistic initialization; softmax tie-break."
        ),
        parameters={
            "window_size":   "Size of sliding window for improvement signal (if used).",
            "p_min (ε)":     "Exploration probability.",
            "learning_rate": "Q-learning step size (α).",
            "gamma (γ)":     "Discount factor.",
            "episode_len":   "Steps per episode.",
            "tau":           "Softmax temperature (tie-break).",
            "optimistic_init":"Initial Q-values for optimism in face of uncertainty.",
        },
        notes=(
            "Reward = normalized makespan improvement per step; small penalty on non-improving steps.",
            "Shares neighborhoods and IG diversification with 1A; deadline-aware.",
        ),
    ),
}

def get_design(key: str) -> MechanismDesign:
    if key not in DESIGNS:
        raise KeyError(f"Unknown mechanism '{key}'. Available: {', '.join(sorted(DESIGNS))}")
    return DESIGNS[key]

def describe_design(key: str) -> str:
    d = get_design(key)
    lines = [f"{d.identifier} ({d.key})", d.objective, "Operators:"]
    lines += [f"  - {op}" for op in d.operators]
    lines.append(f"Scheduler: {d.scheduler}")
    if d.parameters:
        lines.append("Parameters:")
        for k, v in d.parameters.items():
            lines.append(f"  - {k}: {v}")
    if d.notes:
        lines.append("Notes:")
        for t in d.notes:
            lines.append(f"  - {t}")
    return "\n".join(lines)
