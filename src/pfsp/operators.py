"""Efficient PFSP neighbourhood operators with optional Numba acceleration.

The neighbourhood routines operate on NumPy arrays so they can feed the
JIT-compiled makespan evaluation when Numba is available.  The functions keep
the original first-improvement semantics while drastically reducing the time
spent recomputing makespans inside the VND loops.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional acceleration
    from numba import njit
except ImportError:  # pragma: no cover - numba is optional at runtime
    njit = None  # type: ignore


_HAVE_NUMBA = njit is not None


def _makespan_python(permutation: np.ndarray, p_times: np.ndarray) -> int:
    """Pure NumPy implementation of the makespan calculation."""

    m, n = p_times.shape
    completion_prev = np.zeros(n, dtype=np.int64)
    completion_curr = np.zeros(n, dtype=np.int64)
    for machine in range(m):
        for j in range(n):
            job = permutation[j]
            if j == 0:
                prev_job = 0
            else:
                prev_job = completion_curr[j - 1]
            if machine == 0:
                prev_machine = 0
            else:
                prev_machine = completion_prev[j]
            completion_curr[j] = max(prev_job, prev_machine) + p_times[machine, job]
        completion_prev, completion_curr = completion_curr, completion_prev
    return int(completion_prev[n - 1])


if _HAVE_NUMBA:  # pragma: no cover - covered implicitly when numba is present

    @njit(cache=True)
    def _makespan_numba(permutation: np.ndarray, p_times: np.ndarray) -> int:
        m, n = p_times.shape
        completion_prev = np.zeros(n, dtype=np.int64)
        completion_curr = np.zeros(n, dtype=np.int64)
        for machine in range(m):
            for j in range(n):
                job = permutation[j]
                if j == 0:
                    prev_job = 0
                else:
                    prev_job = completion_curr[j - 1]
                if machine == 0:
                    prev_machine = 0
                else:
                    prev_machine = completion_prev[j]
                completion_curr[j] = max(prev_job, prev_machine) + p_times[machine, job]
            temp = completion_prev
            completion_prev = completion_curr
            completion_curr = temp
        return int(completion_prev[n - 1])


def makespan(permutation: Sequence[int], p_times: np.ndarray) -> int:
    """Compute the makespan of a permutation using the fastest available path."""

    perm_arr = np.ascontiguousarray(permutation, dtype=np.int64)
    p_times_arr = np.ascontiguousarray(p_times, dtype=np.int64)
    if _HAVE_NUMBA:
        return int(_makespan_numba(perm_arr, p_times_arr))
    return _makespan_python(perm_arr, p_times_arr)


def _as_array(perm: Sequence[int]) -> np.ndarray:
    """Convert *perm* to a contiguous ``int64`` NumPy array."""

    if isinstance(perm, np.ndarray):
        if perm.dtype != np.int64:
            return np.ascontiguousarray(perm, dtype=np.int64)
        return np.ascontiguousarray(perm)
    return np.ascontiguousarray(np.array(perm, dtype=np.int64, copy=True))


def _relocate_array(perm: np.ndarray, i: int, k: int) -> np.ndarray:
    job = perm[i]
    n = perm.shape[0]
    new_perm = np.empty_like(perm)
    if k < i:
        if k > 0:
            new_perm[:k] = perm[:k]
        new_perm[k] = job
        if i - k > 0:
            new_perm[k + 1 : i + 1] = perm[k:i]
        if i + 1 < n:
            new_perm[i + 1 :] = perm[i + 1 :]
    else:  # k > i
        if i > 0:
            new_perm[:i] = perm[:i]
        if k - i > 0:
            new_perm[i:k] = perm[i + 1 : k + 1]
        new_perm[k] = job
        if k + 1 < n:
            new_perm[k + 1 :] = perm[k + 1 :]
    return new_perm


def _swap_array(perm: np.ndarray, i: int, j: int) -> np.ndarray:
    new_perm = perm.copy()
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm


def _insert_block(remainder: np.ndarray, block: np.ndarray, insert_pos: int) -> np.ndarray:
    if insert_pos == remainder.shape[0]:
        return np.concatenate((remainder, block))
    return np.concatenate((remainder[:insert_pos], block, remainder[insert_pos:]))


def _block_insert_array(perm: np.ndarray, start: int, length: int, insert_pos: int) -> np.ndarray:
    block = perm[start : start + length]
    remainder = np.concatenate((perm[:start], perm[start + length :]))
    return _insert_block(remainder, block, insert_pos)


def _random_choice(values: Iterable[int]) -> int:
    seq = tuple(values)
    return random.choice(seq)


class Operators:
    """Collection of local search and perturbation operators for PFSP."""

    def __init__(self, p_times: np.ndarray) -> None:
        self.p_times = np.ascontiguousarray(p_times, dtype=np.int64)

    def first_improvement_relocate(self, perm: Sequence[int]) -> Optional[Tuple[np.ndarray, int]]:
        perm_arr = _as_array(perm)
        best_value = makespan(perm_arr, self.p_times)
        n = perm_arr.shape[0]
        for i in range(n):
            for k in range(n):
                if k == i:
                    continue
                neighbour = _relocate_array(perm_arr, i, k)
                val = makespan(neighbour, self.p_times)
                if val < best_value:
                    return neighbour, val
        return None

    def first_improvement_swap(self, perm: Sequence[int]) -> Optional[Tuple[np.ndarray, int]]:
        perm_arr = _as_array(perm)
        best_value = makespan(perm_arr, self.p_times)
        n = perm_arr.shape[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                if perm_arr[i] == perm_arr[j]:
                    continue
                neighbour = _swap_array(perm_arr, i, j)
                val = makespan(neighbour, self.p_times)
                if val < best_value:
                    return neighbour, val
        return None

    def perturb_block_insert(
        self, perm: Sequence[int], block_lengths: Tuple[int, ...] = (2, 3)
    ) -> Tuple[np.ndarray, int]:
        perm_arr = _as_array(perm)
        n = perm_arr.shape[0]
        if n <= 1:
            return perm_arr.copy(), makespan(perm_arr, self.p_times)
        length = _random_choice(block_lengths)
        length = min(length, n)
        start_idx = random.randint(0, n - length)
        block = perm_arr[start_idx : start_idx + length]
        remainder = np.concatenate((perm_arr[:start_idx], perm_arr[start_idx + length :]))
        insert_pos = random.randint(0, remainder.shape[0])
        neighbour = _insert_block(remainder, block, insert_pos)
        return neighbour, makespan(neighbour, self.p_times)

    def first_improvement_block(
        self, perm: Sequence[int], block_lengths: Tuple[int, ...] = (2, 3)
    ) -> Optional[Tuple[np.ndarray, int]]:
        perm_arr = _as_array(perm)
        best_value = makespan(perm_arr, self.p_times)
        n = perm_arr.shape[0]
        for length in block_lengths:
            if length > n:
                continue
            for start in range(n - length + 1):
                block = perm_arr[start : start + length]
                remainder = np.concatenate((perm_arr[:start], perm_arr[start + length :]))
                limit = remainder.shape[0]
                for insert_pos in range(limit + 1):
                    if insert_pos == start:
                        continue
                    neighbour = _insert_block(remainder, block, insert_pos)
                    val = makespan(neighbour, self.p_times)
                    if val < best_value:
                        return neighbour, val
        return None
