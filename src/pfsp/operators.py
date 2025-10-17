# src/pfsp/operators.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, List
import time
import numpy as np

# ---------- Core PFSP makespan (tight, cache-friendly) ----------
def makespan(order: np.ndarray, p_times: np.ndarray) -> int:
    m, _ = p_times.shape
    n = order.shape[0]
    C = np.zeros((m, n), dtype=np.int64)
    # machine 0
    C[0, 0] = p_times[0, order[0]]
    for j in range(1, n):
        C[0, j] = C[0, j-1] + p_times[0, order[j]]
    # machines 1..m-1
    for i in range(1, m):
        C[i, 0] = C[i-1, 0] + p_times[i, order[0]]
        for j in range(1, n):
            a = C[i-1, j]
            b = C[i, j-1]
            if a > b:
                C[i, j] = a + p_times[i, order[j]]
            else:
                C[i, j] = b + p_times[i, order[j]]
    return int(C[m-1, n-1])


# ---------- Helpers for NEH / insert-based eval ----------
def _best_insert_pos_with_ties(
    p_times: np.ndarray,
    seq: np.ndarray,
    job: int,
    deadline: Optional[float] = None,
) -> Tuple[int, int]:
    """Greedy best insertion of `job` into `seq` (ties -> earliest pos). Deadline aware."""
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
        # Deadline tripped before any eval; append safely
        cand = np.append(seq, int(job))
        return seq.size, makespan(cand, p_times)
    return best_pos, int(best_val)


# ---------- Critical path & granular candidate selection ----------
def _critical_jobs(order: np.ndarray, p_times: np.ndarray, take: int) -> np.ndarray:
    """Return a set of 'most critical' jobs based on completion slacks on last machine."""
    # Simple proxy: jobs with largest cumulative time on last two machines when scheduled
    # Not perfect CP detection, but cheap & effective.
    m = p_times.shape[0]
    last = p_times[m-1, order]
    penult = p_times[m-2, order] if m >= 2 else last
    score = last + penult
    idx_sorted = np.argsort(-score)  # descending
    take = max(1, min(take, order.size))
    return order[idx_sorted[:take]]


# ---------- Operators with granular neighborhoods & deadlines ----------
class Operators:
    def __init__(self, p_times: np.ndarray) -> None:
        self.p_times = np.ascontiguousarray(p_times, dtype=np.int64)

    # Best-improvement relocate (i -> j), but optionally granular
    def best_improvement_relocate(
        self,
        perm: np.ndarray,
        deadline: Optional[float] = None,
        granular_window: int = 10,
        critical_only: bool = True,
        critical_take_frac: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, int]]:
        n = perm.shape[0]
        base = makespan(perm, self.p_times)
        best_val = base
        best_perm: Optional[np.ndarray] = None

        # Build candidate i's
        if critical_only:
            k = max(1, int(round(critical_take_frac * n)))
            crit = set(int(x) for x in _critical_jobs(perm, self.p_times, k))
        else:
            crit = None

        for i in range(n):
            if deadline and (i % 4 == 0) and time.time() >= deadline:
                break
            job = perm[i]
            if crit is not None and job not in crit:
                continue
            rest = np.delete(perm, i)
            lo = max(0, i - granular_window)
            hi = min(rest.size, i + granular_window)
            for pos in range(lo, hi + 1):
                if pos == i:
                    continue
                if deadline and (pos % 12 == 0) and time.time() >= deadline:
                    break
                cand = np.insert(rest, pos, job)
                val = makespan(cand, self.p_times)
                if val < best_val:
                    best_val = val
                    best_perm = cand
        if best_perm is None:
            return None
        return best_perm, int(best_val)

    # Best-improvement adjacent swap (granular by nature)
    def best_improvement_adjacent_swap(
        self,
        perm: np.ndarray,
        deadline: Optional[float] = None,
        critical_only: bool = False,
        critical_take_frac: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, int]]:
        n = perm.shape[0]
        base = makespan(perm, self.p_times)
        best_val = base
        best_perm: Optional[np.ndarray] = None

        crit = None
        if critical_only:
            k = max(1, int(round(critical_take_frac * n)))
            crit = set(int(x) for x in _critical_jobs(perm, self.p_times, k))

        for i in range(n - 1):
            if deadline and (i % 12 == 0) and time.time() >= deadline:
                break
            if crit is not None and (perm[i] not in crit and perm[i+1] not in crit):
                continue
            cand = perm.copy()
            cand[i], cand[i+1] = cand[i+1], cand[i]
            val = makespan(cand, self.p_times)
            if val < best_val:
                best_val = val
                best_perm = cand
        if best_perm is None:
            return None
        return best_perm, int(best_val)

    # Best-improvement block reinsert (lengths 2/3), granular distance
    def best_improvement_block(
        self,
        perm: np.ndarray,
        block_lengths: Iterable[int] = (2, 3),
        deadline: Optional[float] = None,
        granular_window: int = 10,
        critical_only: bool = True,
        critical_take_frac: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, int]]:
        n = perm.shape[0]
        base = makespan(perm, self.p_times)
        best_val = base
        best_perm: Optional[np.ndarray] = None

        crit = None
        if critical_only:
            k = max(1, int(round(critical_take_frac * n)))
            crit = set(int(x) for x in _critical_jobs(perm, self.p_times, k))

        for L in block_lengths:
            if L >= n:
                continue
            for i in range(0, n - L + 1):
                if deadline and (i % 4 == 0) and time.time() >= deadline:
                    break
                block = perm[i:i+L]
                if crit is not None and not any(int(x) in crit for x in block):
                    continue
                remainder = np.concatenate([perm[:i], perm[i+L:]])
                lo = max(0, i - granular_window)
                hi = min(remainder.size, i + granular_window)
                for pos in range(lo, hi + 1):
                    if pos == i:
                        continue
                    if deadline and (pos % 12 == 0) and time.time() >= deadline:
                        break
                    cand = np.insert(remainder, pos, block)
                    val = makespan(cand, self.p_times)
                    if val < best_val:
                        best_val = val
                        best_perm = cand
        if best_perm is None:
            return None
        return best_perm, int(best_val)

    # Destroy-repair (greedy reinsertion with deadline)
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

    # Ruin–recreate around a reference
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
