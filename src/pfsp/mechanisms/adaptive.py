"""Adaptive probability matching mechanism (Mechanism 2A)."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Iterable, List

from .base import Mechanism


class AdaptiveMechanism(Mechanism):
    """Sliding-window credit assignment with probability matching.

    Parameters
    ----------
    operators:
        Sequence of operator identifiers.
    window_size:
        Number of recent rewards retained for each operator.  The average of
        this buffer constitutes the operator credit.
    p_min:
        Minimum selection probability to keep exploring low-credit operators.
    rng:
        Optional :class:`random.Random` instance.  Defaults to the module-level
        RNG which allows the experiment scripts to seed :mod:`random` globally.
    """

    def __init__(
        self,
        operators: Iterable[str],
        *,
        window_size: int = 50,
        p_min: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(operators)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= p_min <= 1.0:
            raise ValueError("p_min must be within [0, 1]")
        if p_min > 1.0 / len(self.operators):
            raise ValueError("p_min cannot exceed 1/len(operators)")
        self.window_size = window_size
        self.p_min = p_min
        self._rng = rng or random
        self._history: Dict[str, Deque[float]] = {
            op: deque(maxlen=window_size) for op in self.operators
        }
        self._credits: Dict[str, float] = {op: 0.0 for op in self.operators}
        self._probabilities: List[float] = [1.0 / len(self.operators)] * len(self.operators)

    def start_iteration(self) -> None:  # pragma: no cover - nothing to reset
        return

    def select_operator(self) -> str:
        return self._rng.choices(self.operators, weights=self._probabilities, k=1)[0]

    def update(self, operator: str, reward: float) -> None:
        if operator not in self._history:
            raise KeyError(f"Unknown operator '{operator}'")
        self._history[operator].append(reward)
        for op in self.operators:
            history = self._history[op]
            self._credits[op] = sum(history) / len(history) if history else 0.0
        self._recompute_probabilities()

    def _recompute_probabilities(self) -> None:
        total_credit = sum(self._credits.values())
        k = len(self.operators)
        if total_credit > 0.0:
            scale = 1.0 - self.p_min * k
            probs = [self.p_min + scale * (self._credits[op] / total_credit) for op in self.operators]
        else:
            probs = [1.0 / k] * k
        total = sum(probs)
        if total <= 0:
            self._probabilities = [1.0 / k] * k
        else:
            self._probabilities = [p / total for p in probs]


__all__ = ["AdaptiveMechanism"]
