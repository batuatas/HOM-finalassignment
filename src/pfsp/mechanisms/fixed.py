"""Deterministic Variable Neighbourhood Descent scheduler."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .base import Mechanism


class FixedMechanism(Mechanism):
    """Implement Mechanism 1A from the assignment brief.

    The deterministic schedule cycles through the operators in the order in
    which they were provided.  Whenever the local search finds an improving
    move, the sweep is restarted from the first operator.  This behaviour is
    achieved by letting :meth:`select_operator` return ``None`` once the end of
    the list is reached; the metaheuristic resets the mechanism when it wants
    to restart the sweep.
    """

    def __init__(self, operators: Iterable[str]):
        super().__init__(operators)
        self._index = 0

    def start_iteration(self) -> None:
        self._index = 0

    def select_operator(self) -> Optional[str]:
        if self._index >= len(self.operators):
            return None
        op = self.operators[self._index]
        self._index += 1
        return op

    def update(self, operator: str, reward: float) -> None:  # pragma: no cover - nothing to do
        # The deterministic mechanism is oblivious to rewards.
        return


__all__ = ["FixedMechanism"]
