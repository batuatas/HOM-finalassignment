"""Common abstractions for PFSP operator scheduling mechanisms.

The final assignment rubric emphasises clean separation between the
metaheuristic logic and the adaptive scheduling mechanisms.  To satisfy
that requirement we expose a small interface that every mechanism must
implement.  The :class:`Mechanism` base class formalises that contract and
provides a couple of convenience features shared by all concrete
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional


class Mechanism(ABC):
    """Base class for operator scheduling strategies.

    Parameters
    ----------
    operators:
        Iterable of operator identifiers understood by the
        :class:`~pfsp.operators.Operators` helper.  Concrete mechanisms keep
        this ordering internally and may rely on it when exposing metrics to
        the experiment scripts.
    """

    def __init__(self, operators: Iterable[str]):
        self._operators: List[str] = list(operators)
        if not self._operators:
            raise ValueError("At least one operator must be provided")

    @property
    def operators(self) -> List[str]:
        """Return the list of operators managed by this mechanism."""

        return self._operators

    @abstractmethod
    def start_iteration(self) -> None:
        """Prepare the mechanism for a new local-search sweep."""

    @abstractmethod
    def select_operator(self) -> Optional[str]:
        """Return the next operator to apply.

        For deterministic schedules ``None`` signals that the sweep has
        finished.  Adaptive mechanisms always return a valid operator.
        """

    @abstractmethod
    def update(self, operator: str, reward: float) -> None:
        """Notify the mechanism about the reward of ``operator``."""


__all__ = ["Mechanism"]
