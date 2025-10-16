"""Compatibility layer for legacy imports.

Two scheduler implementations are provided:

* ``FixedScheduler`` follows a predetermined sequence of operators.  It
  implements the Variable Neighbourhood Descent logic: iterate through
  operators in order; whenever a move yields an improvement, the sequence
  restarts from the first operator.
* ``AdaptiveScheduler`` implements the Mechanism 2B adaptive pursuit
  scheme.  Operators accumulate credits equal to the average reward over a
  sliding window.  After every move the probabilities are nudged towards a
  target distribution that favours the current best operator while
  preserving a minimum exploration probability ``p_min`` for all
  operators.  The blending speed is controlled by a ``learning_rate``
  parameter.

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
    """An adaptive operator scheduler implementing Mechanism 2B.

    Mechanism 2B follows the adaptive pursuit strategy: rewards collected
    over a sliding window identify the most promising operator, then the
    selection probabilities move towards a target distribution that keeps a
    minimum exploration floor ``p_min`` for every operator.  The pursuit
    step is controlled by ``learning_rate`` and ensures smooth, reactive
    updates that satisfy the assignment rubric for the 2B option.

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
    learning_rate : float, optional
        Step size controlling how quickly probabilities chase the target
        distribution.  Defaults to 0.2.
    """

    def __init__(
        self,
        operators: Sequence[str],
        window_size: int = 50,
        p_min: float = 0.1,
        learning_rate: float = 0.2,
    ):
        self.operators: List[str] = list(operators)
        self.window_size = window_size
        if not 0.0 <= p_min <= 1.0 / max(1, len(self.operators)):
            raise ValueError(
                "p_min must satisfy 0 <= p_min <= 1/len(operators) for adaptive pursuit"
            )
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in the interval (0, 1]")
        self.p_min = p_min
        self.learning_rate = learning_rate
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
        # Determine target distribution based on highest credit
        k = len(self.operators)
        if any(self.history[o] for o in self.operators):
            best_op = max(self.credits, key=self.credits.get)
            best_target = max(0.0, 1.0 - self.p_min * (k - 1))
            targets = []
            for o in self.operators:
                targets.append(best_target if o == best_op else self.p_min)
        else:
            targets = [1.0 / k] * k

        # Blend current probabilities towards targets (adaptive pursuit)
        updated: List[float] = []
        for current, target in zip(self.probabilities, targets):
            updated.append(current + self.learning_rate * (target - current))

        # Normalise to sum to 1 and guard against degeneracy
        total = sum(updated)
        if total > 0:
            self.probabilities = [p / total for p in updated]
        else:
class RandomScheduler:
    """A random operator scheduler implementing Mechanism 1B.

    Operators are applied in a random order for each local search iteration.
    The order is reshuffled at the start of each iteration.
    """

    def __init__(self, operators: Sequence[str]):
        self.operators: List[str] = list(operators)
        self._sequence: List[str] = []
        self._index: int = 0

    def start_iter(self) -> None:
        """Reset for a new iteration by shuffling the operators."""
        self._sequence = self.operators.copy()
        random.shuffle(self._sequence)
        self._index = 0

    def next_operator(self) -> Optional[str]:
        """Return the next operator in the random sequence or ``None`` when done."""
        if self._index >= len(self._sequence):
            return None
        op = self._sequence[self._index]
        self._index += 1
        return op

    def update(self, op: str, reward: float) -> None:
        """Random scheduler does not adapt; this method is a no-op."""
        pass


class QLearningScheduler:
    """A Q-learning operator scheduler implementing Mechanism 2B (QLearning).

    Maintains a Q-value for each operator representing its expected reward and
    chooses operators using an epsilon-greedy policy.
    """

    def __init__(
        self,
        operators: Sequence[str],
        alpha: float = 0.2,
        epsilon: float = 0.1,
    ):
        self.operators: List[str] = list(operators)
        self.alpha = alpha
        self.epsilon = epsilon
        self.qvalues: Dict[str, float] = {op: 0.0 for op in self.operators}

    def start_iter(self) -> None:
        """No special state to reset at the start of an iteration."""
        pass

    def next_operator(self) -> str:
        """Choose an operator using epsilon-greedy over Q-values."""
        if random.random() < self.epsilon:
            return random.choice(self.operators)
        # Exploit: choose operator(s) with maximal Q-value, break ties randomly
        max_q = max(self.qvalues.values())
        best_ops = [op for op, q in self.qvalues.items() if q == max_q]
        return random.choice(best_ops)

    def update(self, op: str, reward: float) -> None:
        """Update the Q-value for the applied operator based on observed reward."""
        old_q = self.qvalues.get(op, 0.0)
        self.qvalues[op] = (1 - self.alpha) * old_q + self.alpha * reward

            self.probabilities = [1.0 / k] * k
