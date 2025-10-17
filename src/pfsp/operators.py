# src/pfsp/operators.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import time
import numpy as np

def makespan(order: np.ndarray, p_times: np.ndarray) -> int:
    m, _ = p_times.shape
    n = order.shape[0]
    C = np.zeros((m, n), dtype=np.int64)
    C[0, 0] = p_times[0, order[0]]
    for j in range(1, n):
        C[0, j] = C[0, j-1] + p_times[0, order[j]]
    for i in range(1, m):
        C[i, 0] = C[i-1, 0] + p_times[i, order[0]]
        for j in range(1, n):
            C[i, j] = max(C[i-1, j], C[i, j-1]) + p_times[i, order[j]]
    return int(C[m-1, n-1])

def _best_insert_pos_with_ties(
    p_times: np.ndarray, seq: np.ndarray, job: int, deadline: Optional[float] = None
) -> Tuple[int, int]:
    """Return (best_pos, best_value) to insert 'job' into seq; earliest pos on ties.
    Obeys deadline (may short-circuit)."""
    if seq.size == 0:
        return 0, int(makespan(np.array([job], dtype=np.int64), p_times))
    best_val: Optional[int] = None
    best_pos = 0
    for pos in range(seq.size + 1):
        if deadline and (pos % 16 == 0) and time.time() >= deadline:
            break
        cand = np.insert(seq, pos, int(job))
        val = makespan(cand, p_times)
        if best_val is None or val < best_val:
            best_val = val
            best_pos = pos
    if best_val is None:
        # deadline tripped before any eval; append
        return seq.size, int(makespan(np.append(seq, int(job)), p_times))
    return best_pos, int(best_val)

class Operators:
    def __init__(self, p_times: np.ndarray) -> None:
        self.p_times = np.ascontiguousarray(p_times, dtype=np.int64)

    def best_improvement_relocate(self, perm: np.ndarray, deadline: Optional[float] = None) -> Optional[Tuple[np.ndarray, int]]:
        n = perm.shape[0]
        base = makespan(perm, self.p_times)
        best_val = base
        best_perm: Optional[np.ndarray] = None
        for i in range(n):
            if deadline and (i % 5 == 0) and time.time() >= deadline:
                break
            job = perm[i]
            rest = np.delete(perm, i)
            for pos in range(rest.size + 1):
                if pos == i:
                    continue
                if deadline and (pos % 16 == 0) and time.time() >= deadline:
                    break
                cand = np.insert(rest, pos, job)
                val = makespan(cand, self.p_times)
                if val < best_val:
                    best_val = val
                    best_perm = cand
        if best_perm is None:
            return None
        return best_perm, int(best_val)

    def best_improvement_adjacent_swap(self, perm: np.ndarray, deadline: Optional[float] = None) -> Optional[Tuple[np.ndarray, int]]:
        n = perm.shape[0]
        base = makespan(perm, self.p_times)
        best_val = base
        best_perm: Optional[np.ndarray] = None
        for i in range(n - 1):
            if deadline and (i % 16 == 0) and time.time() >= deadline:
                break
            cand = perm.copy()
            cand[i], cand[i+1] = cand[i+1], cand[i]
            val = makespan(cand, self.p_times)
            if val < best_val:
                best_val = val
                best_perm = cand
        if best_perm is None:
            return None
        return best_perm, int(best_val)

    def best_improvement_block(self, perm: np.ndarray, block_lengths: Iterable[int] = (2, 3), deadline: Optional[float] = None) -> Optional[Tuple[np.ndarray, int]]:
        n = perm.shape[0]
        base = makespan(perm, self.p_times)
        best_val = base
        best_perm: Optional[np.ndarray] = None
        for L in block_lengths:
            if L >= n:
                continue
            for i in range(0, n - L + 1):
                if deadline and (i % 5 == 0) and time.time() >= deadline:
                    break
                block = perm[i:i+L]
                remainder = np.concatenate([perm[:i], perm[i+L:]])
                for pos in range(remainder.size + 1):
                    if pos == i:
                        continue
                    if deadline and (pos % 16 == 0) and time.time() >= deadline:
                        break
                    cand = np.insert(remainder, pos, block)
                    val = makespan(cand, self.p_times)
                    if val < best_val:
                        best_val = val
                        best_perm = cand
        if best_perm is None:
            return None
        return best_perm, int(best_val)

    def destroy_repair(self, perm: np.ndarray, d: int = 2, deadline: Optional[float] = None) -> Tuple[np.ndarray, int]:
        n = perm.shape[0]
        d = max(1, min(d, n-1))
        idx = np.random.choice(n, size=d, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[idx] = False
        remaining = perm[keep_mask]
        removed = perm[~keep_mask]
        seq = remaining.copy()
        for k, job in enumerate(removed):
            if deadline and (k % 8 == 0) and time.time() >= deadline:
                break
            pos, _ = _best_insert_pos_with_ties(self.p_times, seq, int(job), deadline=deadline)
            seq = np.insert(seq, pos, int(job))
        return seq, makespan(seq, self.p_times)

    def ruin_recreate(self, ref_perm: np.ndarray, ratio: float = 0.2, deadline: Optional[float] = None) -> Tuple[np.ndarray, int]:
        n = ref_perm.shape[0]
        d = max(1, min(int(round(ratio * n)), n-1))
        idx = np.random.choice(n, size=d, replace=False)
        keep = ref_perm[np.setdiff1d(np.arange(n), idx)]
        removed = ref_perm[idx]
        np.random.shuffle(removed)
        seq = keep.copy()
        for k, job in enumerate(removed):
            if deadline and (k % 8 == 0) and time.time() >= deadline:
                break
            pos, _ = _best_insert_pos_with_ties(self.p_times, seq, int(job), deadline=deadline)
            seq = np.insert(seq, pos, int(job))
        return seq, makespan(seq, self.p_times)
