"""Operator scheduling mechanisms for the PFSP metaheuristic.

Two scheduler implementations are provided:

* ``FixedScheduler`` follows a predetermined sequence of operators.  It
  implements the Variable Neighbourhood Descent logic: iterate through
  operators in order; whenever a move yields an improvement, the sequence
  restarts from the first operator.
* ``AdaptiveScheduler`` implements a simple adaptive operator selection
  mechanism based on credit assignment and probability matching.  Each
  operator accumulates a credit equal to the average reward obtained in
  the last ``window_size`` calls.  The probability of selecting an
  operator is proportional to its credit (with a minimum probability
  ``p_min`` to ensure exploration).  Credits and probabilities are
  updated immediately after each operator call.

Both schedulers expose a common interface comprising three methods:

``start_iter()``
    Reset internal state at the beginning of a local search iteration.

``next_operator() -> str | None``
    Return the name of the next operator to apply.  Returns ``None`` when
    all operators have been exhausted (for the fixed scheduler).

``update(op: str, reward: float)``
    Notify the scheduler of the reward obtained from applying operator
    ``op``.  This triggers credit updates and probability recomputation
    in the adaptive case.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple


class FixedScheduler:
    """A deterministic operator scheduler implementing Mechanism 1A.

    Parameters
    ----------
    operators : Sequence[str]
        A list of operator names (e.g. ``['relocate', 'swap', 'block']``) that
        defines the order in which operators are applied.
    """

    def __init__(self, operators: Sequence[str]):
        self.operators: List[str] = list(operators)
        self._index = 0

    def start_iter(self) -> None:
        """Reset the scheduler at the start of a local search sweep."""
        self._index = 0

    def next_operator(self) -> Optional[str]:
        """Return the next operator in the fixed sequence or ``None`` when done."""
        if self._index >= len(self.operators):
            return None
        op = self.operators[self._index]
        self._index += 1
        return op

    def update(self, op: str, reward: float) -> None:
        """Fixed scheduler does not adapt; this method is a no‐op."""
        # No adaptive behaviour
        pass


class AdaptiveScheduler:
    """An adaptive operator scheduler implementing Mechanism 2A.

    This scheduler assigns a numerical credit to each operator based on the
    average reward obtained over a sliding window.  At each call to
    ``next_operator``, it samples an operator according to the probability
    matching rule:

    ``p_j = p_min + (1 - p_min * K) * (credit_j / sum credits)``

    where ``K`` is the number of operators.  When no credits are available
    (e.g. at the beginning), all operators are selected with equal
    probability.  After each operator application, the scheduler must be
    notified via ``update(op, reward)`` so that credits and probabilities
    are updated.

    Parameters
    ----------
    operators : Sequence[str]
        A list of operator names.
    window_size : int, optional
        The length of the sliding window for credit computation.  Defaults
        to 50.  Larger values smooth out credit fluctuations but slow down
        adaptation.
    p_min : float, optional
        The minimum selection probability for any operator.  Must satisfy
        0 <= p_min <= 1/len(operators).  Defaults to 0.1.
    """

    def __init__(self, operators: Sequence[str], window_size: int = 50, p_min: float = 0.1):
        self.operators: List[str] = list(operators)
        self.window_size = window_size
        self.p_min = p_min
        self.history: Dict[str, deque] = {op: deque(maxlen=window_size) for op in self.operators}
        self.credits: Dict[str, float] = {op: 0.0 for op in self.operators}
        self.probabilities: List[float] = [1.0 / len(self.operators)] * len(self.operators)

    def start_iter(self) -> None:
        """Nothing to reset at the start of a local search sweep for adaptive scheduler."""
        pass

    def next_operator(self) -> str:
        """Sample an operator according to current probabilities."""
        # Use random.choices to sample according to probabilities
        op = random.choices(self.operators, weights=self.probabilities, k=1)[0]
        return op

    def update(self, op: str, reward: float) -> None:
        """Update the credit and probabilities after applying operator ``op``.

        Parameters
        ----------
        op : str
            Operator name that was applied.
        reward : float
            The reward obtained from applying the operator.  Typical values
            are non‐negative; 0 indicates no improvement.
        """
        # Append reward to history
        self.history[op].append(reward)
        # Recompute credits as average of last window_size rewards
        for o in self.operators:
            if self.history[o]:
                self.credits[o] = sum(self.history[o]) / len(self.history[o])
            else:
                self.credits[o] = 0.0
        # Compute probabilities
        total_credit = sum(self.credits.values())
        k = len(self.operators)
        if total_credit > 0:
            probs = []
            for o in self.operators:
                base = self.p_min
                scaled = (1.0 - self.p_min * k) * (self.credits[o] / total_credit)
                probs.append(base + scaled)
        else:
            # If all credits are zero, assign equal probabilities
            probs = [1.0 / k] * k
        # Normalise to sum to 1 due to possible numerical issues
        s = sum(probs)
        if s > 0:
            self.probabilities = [p / s for p in probs]
        else:
            self.probabilities = [1.0 / k] * k
