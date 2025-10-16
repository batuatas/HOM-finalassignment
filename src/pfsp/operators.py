"""Local search operators and accelerated makespan for PFSP (Numba-backed).

Exposes:
- makespan(perm, p_times) -> int
- class Operators with:
    * first_improvement_relocate
    * first_improvement_swap
    * first_improvement_block
    * perturb_block_insert
The code assumes p_times.shape == (m, n) and perm is a permutation of [0..n-1].
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple
import numpy as np

# Optional Numba import with graceful fallback
try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def dec(f):
            return f
        return dec

# ---------------------------
# Core makespan (Numba JIT)
# ---------------------------

@njit(cache=True, fastmath=True)
def _makespan_jit(perm: np.ndarray, p_times: np.ndarray) -> int:
    # p_times: (m, n); perm: (n,)
    m, n = p_times.shape[0], perm.shape[0]
    C = np.zeros(m, dtype=np.int64)
    for j_idx in range(n):
        job = perm[j_idx]
        # machine 0
        C[0] += p_times[0, job]
        # machines 1..m-1
        for i in range(1, m):
            if C[i] > C[i-1]:
                C[i] = C[i] + p_times[i, job]
            else:
                C[i] = C[i-1] + p_times[i, job]
    return int(C[m-1])


def makespan(perm: Sequence[int] | np.ndarray, p_times: np.ndarray) -> int:
    perm_arr = _as_array(perm)
    return _makespan_jit(perm_arr, np.ascontiguousarray(p_times, dtype=np.int64))


# ---------------------------
# Utilities
# ---------------------------

def _as_array(perm: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(perm, dtype=np.int64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return np.ascontiguousarray(arr, dtype=np.int64)


def _relocate_array(perm: np.ndarray, i: int, k: int) -> np.ndarray:
    # Move element at index i to position k (after removing it).
    n = perm.shape[0]
    if i == k:
        return perm.copy()
    out = np.empty_like(perm)
    if k < i:
        # insert earlier
        out[:k] = perm[:k]
        out[k] = perm[i]
        out[k+1:i+1] = perm[k:i]
        out[i+1:] = perm[i+1:]
    else:
        # insert later
        out[:i] = perm[:i]
        out[i:k] = perm[i+1:k+1]
        out[k] = perm[i]
        out[k+1:] = perm[k+1:]
    return out


def _swap_array(perm: np.ndarray, i: int, j: int) -> np.ndarray:
    out = perm.copy()
    out[i], out[j] = out[j], out[i]
    return out


def _insert_block(remainder: np.ndarray, block: np.ndarray, pos: int) -> np.ndarray:
    n = remainder.shape[0]
    if pos == n:
        return np.concatenate((remainder, block))
    return np.concatenate((remainder[:pos], block, remainder[pos:]))


# ---------------------------
# Operators
# ---------------------------

class Operators:
    """Collection of local search and perturbation operators for PFSP."""

    def __init__(self, p_times: np.ndarray) -> None:
        self.p_times = np.ascontiguousarray(p_times, dtype=np.int64)

    # 1-insert (relocate): first-improvement
    def first_improvement_relocate(self, perm: Sequence[int] | np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
        perm_arr = _as_array(perm)
        n = perm_arr.shape[0]
        best_val = makespan(perm_arr, self.p_times)
        for i in range(n):
            for k in range(n):
                if k == i:
                    continue
                neigh = _relocate_array(perm_arr, i, k)
                val = makespan(neigh, self.p_times)
                if val < best_val:
                    return neigh, val
        return None

    # swap: first-improvement
    def first_improvement_swap(self, perm: Sequence[int] | np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
        perm_arr = _as_array(perm)
        n = perm_arr.shape[0]
        best_val = makespan(perm_arr, self.p_times)
        for i in range(n - 1):
            pi = perm_arr[i]
            for j in range(i + 1, n):
                if pi == perm_arr[j]:
                    continue
                neigh = _swap_array(perm_arr, i, j)
                val = makespan(neigh, self.p_times)
                if val < best_val:
                    return neigh, val
        return None

    # block insert (length in {2,3} default): first-improvement
    def first_improvement_block(
        self,
        perm: Sequence[int] | np.ndarray,
        *,
        block_lengths: Tuple[int, ...] = (2, 3),
    ) -> Optional[Tuple[np.ndarray, int]]:
        perm_arr = _as_array(perm)
        n = perm_arr.shape[0]
        best_val = makespan(perm_arr, self.p_times)
        for L in block_lengths:
            L = min(max(2, L), n)  # clamp
            for start in range(0, n - L + 1):
                block = perm_arr[start : start + L]
                remainder = np.concatenate((perm_arr[:start], perm_arr[start + L :]))
                rN = remainder.shape[0]
                for pos in range(rN + 1):
                    neigh = _insert_block(remainder, block, pos)
                    val = makespan(neigh, self.p_times)
                    if val < best_val:
                        return neigh, val
        return None

    # stochastic block-perturb for IG/ILS
    def perturb_block_insert(
        self,
        perm: Sequence[int] | np.ndarray,
        *,
        block_lengths: Tuple[int, ...] = (2, 3),
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, int]:
        g = rng if rng is not None else np.random.default_rng()
        perm_arr = _as_array(perm)
        n = perm_arr.shape[0]
        if n <= 1:
            return perm_arr.copy(), makespan(perm_arr, self.p_times)
        L = int(g.choice(np.array(block_lengths, dtype=np.int64)))
        L = min(max(2, L), n)
        start = int(g.integers(0, n - L + 1))
        block = perm_arr[start : start + L]
        remainder = np.concatenate((perm_arr[:start], perm_arr[start + L :]))
        pos = int(g.integers(0, remainder.shape[0] + 1))
        neigh = _insert_block(remainder, block, pos)
        return neigh, makespan(neigh, self.p_times)
