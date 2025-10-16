"""Design summaries for the final assignment mechanisms.

The final assignment asks for two mechanisms:

* Mechanism 1A – a deterministic Variable Neighbourhood Descent (VND)
  schedule cycling through the relocate, swap and block operators.
* Mechanism 2A – an adaptive operator selection scheme using probability
  matching with sliding–window credit assignment.

This module captures the high level design decisions for each mechanism so
that the rest of the codebase can reference a single source of truth when
producing reports or documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MechanismDesign:
    """High level description of an assignment mechanism.

    Attributes
    ----------
    key:
        Registry key used throughout the codebase (``"fixed"`` or
        ``"adaptive"``).
    identifier:
        Human readable identifier from the assignment brief (e.g.
        ``"Mechanism 1A"``).
    objective:
        Short sentence describing the design rationale.
    operators:
        Ordered sequence of operators used inside the VND loop.
    scheduler:
        Summary of how the operators are scheduled.
    parameters:
        Mapping of configurable parameters to a short explanation.  This is
        primarily used when rendering documentation.
    notes:
        Additional implementation notes relevant to the rubric.
    """

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
        objective="Deterministic VND sweep with relocate, swap and block operators",
        operators=("relocate", "swap", "block"),
        scheduler=(
            "Operators are applied in a fixed order with the sweep restarting "
            "whenever an improving move is accepted."
        ),
        parameters={},
        notes=(
            "Implements the deterministic baseline requested in the brief.",
            "Relies solely on move acceptance to drive diversification.",
        ),
    ),
    "adaptive": MechanismDesign(
        key="adaptive",
        identifier="Mechanism 2A",
        objective="Adaptive probability matching with sliding-window credits",
        operators=("relocate", "swap", "block"),
        scheduler=(
            "Credits are updated after each move using the recent reward window; "
            "selection probabilities are recomputed with a minimum exploration rate."
        ),
        parameters={
            "window_size": "Number of recent rewards tracked when updating credits.",
            "p_min": "Lower bound on operator selection probabilities to retain exploration.",
        },
        notes=(
            "Matches the adaptive operator selection mechanism described in the rubric.",
            "Uses reward normalisation by current makespan improvement to stabilise updates.",
        ),
    ),
}


def get_design(key: str) -> MechanismDesign:
    """Return the design entry for ``key`` raising ``KeyError`` if unknown."""

    if key not in DESIGNS:
        raise KeyError(
            f"Unknown mechanism '{key}'. Available designs: {', '.join(sorted(DESIGNS))}"
        )
    return DESIGNS[key]


def describe_design(key: str) -> str:
    """Return a formatted multi-line description for ``key``."""

    design = get_design(key)
    lines = [f"{design.identifier} ({design.key})", design.objective, "Operators:"]
    for op in design.operators:
        lines.append(f"  - {op}")
    lines.append(f"Scheduler: {design.scheduler}")
    if design.parameters:
        lines.append("Parameters:")
        for name, desc in design.parameters.items():
            lines.append(f"  - {name}: {desc}")
    if design.notes:
        lines.append("Notes:")
        for note in design.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)
