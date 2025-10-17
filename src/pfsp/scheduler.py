# scheduler.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

# Discrete bins for state signals
def _bin_improvement(delta_frac: float) -> int:
    # >2%, >1%, >0.1%, else 0
    if delta_frac > 0.02: return 3
    if delta_frac > 0.01: return 2
    if delta_frac > 0.001: return 1
    return 0

def _bin_stagnation(no_improve_steps: int) -> int:
    if no_improve_steps >= 40: return 3
    if no_improve_steps >= 20: return 2
    if no_improve_steps >= 5: return 1
    return 0


@dataclass
class FixedOrderScheduler:
    op_names: List[str]
    idx: int = 0
    def select(self, state: Tuple[int, int]) -> str:
        op = self.op_names[self.idx]
        self.idx = (self.idx + 1) % len(self.op_names)
        return op
    def update(self, *args, **kwargs) -> None:
        pass


@dataclass
class QLearningScheduler:
    op_names: List[str]
    window_size: int = 50
    eps: float = 0.10
    alpha: float = 0.30
    gamma: float = 0.60
    episode_len: int = 50
    step_in_episode: int = 0
    Q: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    def _ensure_state(self, s: Tuple[int, int]) -> None:
        if s not in self.Q:
            self.Q[s] = np.zeros(len(self.op_names), dtype=np.float64)

    def select(self, state: Tuple[int, int]) -> str:
        self._ensure_state(state)
        self.step_in_episode += 1
        if np.random.rand() < self.eps:
            a = np.random.randint(len(self.op_names))
        else:
            q = self.Q[state]
            a = int(np.argmax(q))
        if self.step_in_episode >= self.episode_len:
            self.step_in_episode = 0
            # mild decay of epsilon to increase exploitation
            self.eps = max(0.02, self.eps * 0.98)
        return self.op_names[a]

    def update(self, s: Tuple[int, int], a_name: str, r: float, s_next: Tuple[int, int]) -> None:
        self._ensure_state(s)
        self._ensure_state(s_next)
        a = self.op_names.index(a_name)
        qsa = self.Q[s][a]
        td_target = r + self.gamma * float(np.max(self.Q[s_next]))
        self.Q[s][a] = qsa + self.alpha * (td_target - qsa)

    @staticmethod
    def state_from_signals(delta_frac: float, no_improve_steps: int) -> Tuple[int, int]:
        return (_bin_improvement(delta_frac), _bin_stagnation(no_improve_steps))

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
>>>>>>> a9b38e8484df61b3a35b9fcb3126ffc4a99f9594
