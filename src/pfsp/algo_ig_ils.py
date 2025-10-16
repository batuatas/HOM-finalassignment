"""Iterated Greedy / Local Search metaheuristic for PFSP.

This module implements a simple single‐solution metaheuristic combining
Iterated Greedy/Iterated Local Search (IG/ILS) with Variable Neighbourhood
Descent (VND) local search.  Two operator scheduling mechanisms are
supported: a fixed sequence (Mechanism 1A) and an adaptive probability
matching approach (Mechanism 2A).

The main entry point is the ``IteratedGreedyILS`` class.  Create an instance
with a processing time matrix and your choice of mechanism, then call
``run`` to search for high quality permutations.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .operators import Operators, makespan
from .mechanisms import build_scheduler


@dataclass
class IGILSResult:
    """Container holding the outcome of a solver run."""

    permutation: List[int]
    makespan: int
    iterations: int


class IteratedGreedyILS:
    """Iterated Greedy / Local Search metaheuristic for the PFSP.

    Parameters
    ----------
    p_times : np.ndarray
        Processing time matrix of shape ``(m, n)``.
    mechanism : str, optional
        The scheduling mechanism to use.  ``'fixed'`` selects the fixed
        sequence scheduler (Mechanism 1A), while ``'adaptive'`` selects
        the adaptive scheduler (Mechanism 2A).  Defaults to ``'fixed'``.
    window_size : int, optional
        Sliding window size for credit computation in the adaptive
        scheduler.  Ignored for the fixed scheduler.  Default is 50.
    p_min : float, optional
        Minimum probability for each operator in the adaptive scheduler.
        Ignored for the fixed scheduler.  Default is 0.1.
    block_lengths : Tuple[int, ...], optional
        Block lengths used for perturbation and the block operator.  Default
        is (2, 3).
    seed : Optional[int], optional
        Random seed for reproducibility.  Default is ``None`` (no seeding).
    """

    def __init__(
        self,
        p_times: np.ndarray,
        mechanism: str = "fixed",
        window_size: int = 50,
        p_min: float = 0.1,
        block_lengths: Tuple[int, ...] = (2, 3),
        seed: Optional[int] = None,
    ) -> None:
        self.p_times = p_times
        self.operators = Operators(p_times)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Define the list of operator names
        self.op_names = ["relocate", "swap", "block"]
        self.mechanism = mechanism
        scheduler_options = {"window_size": window_size, "p_min": p_min}
        self.scheduler = build_scheduler(mechanism, self.op_names, scheduler_options)
        self.block_lengths = block_lengths

    def _local_search(self, perm: List[int], value: int) -> Tuple[List[int], int]:
        """Perform VND local search until no operator yields an improvement.

        Parameters
        ----------
        perm : List[int]
            Starting permutation.
        value : int
            Makespan of the starting permutation.

        Returns
        -------
        Tuple[List[int], int]
            The locally optimal permutation and its makespan.
        """
        current_perm = perm
        current_val = value
        improved = True
        while improved:
            improved = False
            # Reset scheduler at the start of each VND sweep
            self.scheduler.start_iter()
            while True:
                op_name = self.scheduler.next_operator()
                if op_name is None:
                    break
                neighbour = None
                # Apply the chosen operator
                if op_name == "relocate":
                    neighbour = self.operators.first_improvement_relocate(current_perm)
                elif op_name == "swap":
                    neighbour = self.operators.first_improvement_swap(current_perm)
                elif op_name == "block":
                    neighbour = self.operators.first_improvement_block(current_perm, block_lengths=self.block_lengths)
                # Compute reward
                reward = 0.0
                if neighbour is not None:
                    new_perm, new_val = neighbour
                    reward = (current_val - new_val) / max(1, current_val)
                # Update scheduler with reward
                self.scheduler.update(op_name, reward)
                # If improvement found, accept and restart sweep
                if neighbour is not None and new_val < current_val:
                    current_perm, current_val = new_perm, new_val
                    improved = True
                    break
            # End of sweep
        return current_perm, current_val

    def run(
        self,
        max_iter: int = 1000,
        max_no_improve: int = 50,
        time_limit: Optional[float] = None,
        verbose: bool = False,
    ) -> IGILSResult:
        """Run the metaheuristic and return the best found permutation.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of ILS iterations (local optima followed by
            perturbations).  Defaults to 1000.
        max_no_improve : int, optional
            Maximum number of consecutive iterations without improvement
            allowed before terminating.  Defaults to 50.
        time_limit : float, optional
            Time limit in seconds.  If provided, the algorithm stops when
            ``time.time() - start_time`` exceeds this value.  Defaults to
            ``None`` (no time limit).
        verbose : bool, optional
            If True, print progress information to stdout.  Defaults to False.

        Returns
        -------
        IGILSResult
            Dataclass capturing the best permutation, its makespan and the
            number of ILS iterations performed.
        """
        m, n = self.p_times.shape
        # Start from a random permutation
        current_perm = list(range(n))
        random.shuffle(current_perm)
        current_val = makespan(current_perm, self.p_times)
        best_perm = current_perm.copy()
        best_val = current_val
        no_improve_count = 0
        start_time = time.time()
        iterations = 0
        for it in range(max_iter):
            iterations = it + 1
            # Check time limit
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                if verbose:
                    print(f"Time limit reached at iteration {it}")
                break
            # Perform VND local search
            current_perm, current_val = self._local_search(current_perm, current_val)
            # Update best solution
            if current_val < best_val:
                best_perm, best_val = current_perm.copy(), current_val
                no_improve_count = 0
                if verbose:
                    print(f"Iter {it}: improved makespan to {best_val}")
            else:
                no_improve_count += 1
            if no_improve_count >= max_no_improve:
                if verbose:
                    print(f"No improvement for {max_no_improve} iterations; stopping.")
                break
            # Apply perturbation (block insert) to escape local optimum
            current_perm, current_val = self.operators.perturb_block_insert(
                current_perm, block_lengths=self.block_lengths
            )
        return IGILSResult(permutation=best_perm, makespan=best_val, iterations=iterations)
