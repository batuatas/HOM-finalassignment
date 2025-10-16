"""Human-readable mechanism descriptions used by --describe flags."""

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
        objective="Deterministic VND sweep with relocate → swap → block.",
        operators=("relocate", "swap", "block"),
        scheduler=("Apply operators in fixed order; restart sweep whenever an improving move is accepted."),
        parameters={},
        notes=("Baseline per brief.",),
    ),
    "adaptive": MechanismDesign(
        key="adaptive",
        identifier="Mechanism 2B (Q-learning)",
        objective="Episode-based ε-greedy Q-learning to select the most promising operator online.",
        operators=("relocate", "swap", "block"),
        scheduler=(
            "State = (recent improvement bin, stagnation bin). "
            "At each step pick operator with ε-greedy policy; "
            "update tabular Q with one-step TD target. Episodes have fixed length."
        ),
        parameters={
            "window_size": "Size of sliding window to compute recent-improvement signal.",
            "p_min (ε)": "Exploration probability for ε-greedy (mapped from --p-min).",
            "learning_rate (α)": "Q-learning step size (mapped from --learning-rate).",
            "gamma (γ)": "Discount factor (default 0.60).",
            "episode_len": "Steps per episode before mild ε decay.",
        },
        notes=(
            "Reward = normalised makespan improvement (Δ / current makespan).",
            "Meets the assignment’s Mechanism 2B requirement (Q-learning episodes).",
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
