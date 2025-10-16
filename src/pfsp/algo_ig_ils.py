"""Iterated Greedy + ILS with VND for PFSP (with optional progress callback)."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .operators import Operators, makespan
from .mechanisms import build_scheduler, get_mechanism


@dataclass
class IGILSResult:
    permutation: List[int]
    makespan: int
    iterations: int


def _neh(p_times: np.ndarray) -> np.ndarray:
    """NEH initialisation (descending sum with greedy insertion)."""
    m, n = p_times.shape
    order = np.argsort(-np.sum(p_times, axis=0))
    perm = np.empty(0, dtype=np.int64)
    for j in order:
        best_val = None
        best_perm = None
        for pos in range(perm.shape[0] + 1):
            cand = np.insert(perm, pos, j)
            val = makespan(cand, p_times)
            if best_val is None or val < best_val:
                best_val = val
                best_perm = cand
        perm = best_perm
    return perm


class IteratedGreedyILS:
    def __init__(
        self,
        p_times: np.ndarray,
        mechanism: str = "fixed",
        window_size: int = 50,
        p_min: float = 0.10,
        learning_rate: float = 0.30,
        block_lengths: Tuple[int, ...] = (2, 3),
        seed: Optional[int] = None,
    ) -> None:
        self.p_times = np.ascontiguousarray(p_times, dtype=np.int64)
        self.ops = Operators(self.p_times)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.mechanism_key = mechanism
        self.mech_spec = get_mechanism(mechanism)
        self.op_names = list(self.mech_spec.design.operators)
        self.scheduler = build_scheduler(
            mechanism,
            self.op_names,
            options={
                "window_size": window_size,
                "p_min": p_min,
                "learning_rate": learning_rate,
            },
        )
        self.block_lengths = block_lengths

    def _local_search(self, perm: np.ndarray, value: int) -> Tuple[np.ndarray, int]:
        current_perm = np.ascontiguousarray(perm, dtype=np.int64)
        current_val = value
        improved = True
        while improved:
            improved = False
            self.scheduler.start_iter()
            attempts = 0
            max_attempts = len(self.op_names)
            while attempts < max_attempts:
                op_name = self.scheduler.next_operator()
                if op_name is None:
                    break
                attempts += 1
                neighbour = None
                if op_name == "relocate":
                    neighbour = self.ops.first_improvement_relocate(current_perm)
                elif op_name == "swap":
                    neighbour = self.ops.first_improvement_swap(current_perm)
                elif op_name == "block":
                    neighbour = self.ops.first_improvement_block(current_perm, block_lengths=self.block_lengths)
                reward = 0.0
                if neighbour is not None:
                    new_perm, new_val = neighbour
                    reward = (current_val - new_val) / max(1, current_val)
                self.scheduler.update(op_name, reward)
                if neighbour is not None and new_val < current_val:
                    current_perm, current_val = new_perm, new_val
                    improved = True
                    break
        return current_perm, current_val

    def run(
        self,
        max_iter: int = 1000,
        max_no_improve: int = 50,
        time_limit: Optional[float] = None,
        verbose: bool = False,
        # NEW:
        progress_cb: Optional[Callable[[int, int], None]] = None,
        progress_every: int = 10,
    ) -> IGILSResult:
        perm = _neh(self.p_times)
        val = makespan(perm, self.p_times)
        best_perm = perm.copy()
        best_val = val
        no_improve = 0
        start = time.time()
        iterations = 0

        def _maybe_log(it: int) -> None:
            if progress_cb and (it % max(1, progress_every) == 0):
                progress_cb(it, best_val)

        for it in range(max_iter):
            iterations = it + 1
            if time_limit is not None and (time.time() - start) >= time_limit:
                if verbose:
                    print(f"[stop] time limit reached at iter {it}")
                break

            perm, val = self._local_search(perm, val)

            if val < best_val:
                best_perm, best_val = perm.copy(), val
                no_improve = 0
                if verbose:
                    print(f"[{it}] best={best_val}")
                if progress_cb:
                    progress_cb(iterations, best_val)
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    if verbose:
                        print(f"[stop] no improvement in {max_no_improve} iters")
                    break

            # IG/ILS perturbation
            perm, val = self.ops.perturb_block_insert(perm, block_lengths=self.block_lengths)
            _maybe_log(iterations)

        return IGILSResult(permutation=best_perm.tolist(), makespan=best_val, iterations=iterations)
