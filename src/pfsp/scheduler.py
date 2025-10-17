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
