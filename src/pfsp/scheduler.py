"""Operator schedulers for PFSP mechanisms.

Mechanism 1A -> FixedScheduler (deterministic VND sweep).
Mechanism 2B -> QLearningScheduler (ε-greedy tabular Q-learning with episodes).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple


class FixedScheduler:
    """Deterministic operator order (Mechanism 1A)."""

    def __init__(self, operators: Sequence[str]) -> None:
        self.operators: List[str] = list(operators)
        self._index: int = 0

    def start_iter(self) -> None:
        """Called at the start of each VND sweep."""
        self._index = 0

    def next_operator(self) -> Optional[str]:
        if self._index >= len(self.operators):
            return None
        op = self.operators[self._index]
        self._index += 1
        return op

    def update(self, op: str, reward: float) -> None:
        """Fixed policy ignores rewards."""
        return


class QLearningScheduler:
    """Tabular Q-learning over operator choices (Mechanism 2B).

    State  = (improvement_bin, stagnation_bin)
    Action = one of the operator names
    Reward = normalised makespan improvement (Δ / current makespan, clipped ≥ 0)

    Parameters
    ----------
    operators : list[str]
    window_size : int      sliding window for recent reward average (state signal)
    epsilon : float        ε for ε-greedy action selection
    alpha : float          learning rate
    gamma : float          discount factor
    episode_len : int      steps per episode before mild ε decay
    """

    def __init__(
        self,
        operators: Sequence[str],
        window_size: int = 50,
        epsilon: float = 0.10,
        alpha: float = 0.30,
        gamma: float = 0.60,
        episode_len: int = 10,
    ) -> None:
        self.ops: List[str] = list(operators)
        self.win: int = max(5, int(window_size))

        self.epsilon: float = float(epsilon)
        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)
        self.episode_len: int = max(1, int(episode_len))

        self.hist: Deque[float] = deque(maxlen=self.win)
        self.no_improve: int = 0
        self.step_in_episode: int = 0
        self.episode_idx: int = 0

        # Q-table: key = (state_tuple, op_name)
        self.Q: Dict[Tuple[Tuple[int, int], str], float] = {}

        self._state: Tuple[int, int] = self._compute_state()

    # ------------ public API used by the metaheuristic ------------

    def start_iter(self) -> None:
        """Called at the start of each VND sweep (no reset needed)."""
        return

    def next_operator(self) -> str:
        """Pick next operator by ε-greedy policy from current state."""
        self.step_in_episode += 1
        if random.random() < self.epsilon:
            return random.choice(self.ops)

        s = self._state
        best_op, best_q = None, float("-inf")
        for op in self.ops:
            q = self.Q.get((s, op), 0.0)
            if q > best_q:
                best_q, best_op = q, op
        return best_op if best_op is not None else random.choice(self.ops)

    def update(self, op: str, reward: float) -> None:
        """Observe reward, transition to next state, and perform one-step Q update."""
        r = max(0.0, float(reward))
        prev = self._state

        # track stagnation & improvement history
        if r > 1e-12:
            self.no_improve = 0
        else:
            self.no_improve += 1
        self.hist.append(r)

        next_state = self._compute_state()

        # Q-learning: Q(s,a) ← Q + α [ r + γ max_a' Q(s',a') − Q ]
        key = (prev, op)
        old = self.Q.get(key, 0.0)
        best_next = max(self.Q.get((next_state, a), 0.0) for a in self.ops)
        self.Q[key] = old + self.alpha * (r + self.gamma * best_next - old)

        self._state = next_state

        # simple episode handling: periodic mild ε decay
        if self.step_in_episode >= self.episode_len:
            self.step_in_episode = 0
            self.episode_idx += 1
            self.epsilon = max(0.02, self.epsilon * 0.98)

    # ------------ state representation ------------

    def _compute_state(self) -> Tuple[int, int]:
        """Map recent progress to discrete bins: (improvement, stagnation)."""
        avg = (sum(self.hist) / len(self.hist)) if self.hist else 0.0
        if avg > 0.01:
            impr = 2  # strong
        elif avg > 0.001:
            impr = 1  # weak
        else:
            impr = 0  # none

        if self.no_improve >= 50:
            stag = 3
        elif self.no_improve >= 20:
            stag = 2
        elif self.no_improve >= 5:
            stag = 1
        else:
            stag = 0

        return (impr, stag)


# Backwards-compat alias name sometimes used elsewhere
AdaptiveScheduler = QLearningScheduler
