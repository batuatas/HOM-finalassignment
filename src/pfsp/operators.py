"""Local search and perturbation operators for PFSP.

This module implements three standard neighbourhood moves on permutations of jobs:

* ``relocate`` (1–insert): remove one job from position ``i`` and insert it at
  position ``k``.
* ``swap`` (2–swap): exchange the jobs at positions ``i`` and ``j``.
* ``block_insert``: remove a contiguous block of length 2 or 3 and reinsert it
  elsewhere.  This operator serves as a simple perturbation mechanism.

For each operator a first‐improvement strategy is provided.  Given a current
solution (permutation) and the processing time matrix, the method explores
neighbours in a deterministic order and returns the first strictly improving
neighbour found.  If no improving neighbour exists, ``None`` is returned.

The code uses a generic ``makespan`` function to compute objective values.
It is written in pure Python/NumPy and prioritises clarity over extreme
performance.  For larger instances you may wish to implement incremental
evaluation strategies to reduce the cost of neighbourhood searches.
"""

from __future__ import annotations

import numpy as np
import random
from typing import List, Optional, Sequence, Tuple


def makespan(permutation: Sequence[int], p_times: np.ndarray) -> int:
    """Compute the makespan of a given permutation.

    Parameters
    ----------
    permutation : Sequence[int]
        A permutation of job indices (length ``n``).
    p_times : np.ndarray
        Processing time matrix with shape ``(m, n)``.

    Returns
    -------
    int
        The completion time of the last job on the last machine (makespan).
    """
    m, n = p_times.shape
    # Completion times array: we only need previous machine and previous job
    completion_prev_machine = np.zeros(n, dtype=int)
    for i in range(m):
        completion_this_machine = np.zeros(n, dtype=int)
        for j in range(n):
            job = permutation[j]
            if j == 0:
                prev_job_completion = 0
            else:
                prev_job_completion = completion_this_machine[j - 1]
            if i == 0:
                prev_machine_completion = 0
            else:
                prev_machine_completion = completion_prev_machine[j]
            completion_this_machine[j] = max(prev_job_completion, prev_machine_completion) + p_times[i, job]
        completion_prev_machine = completion_this_machine
    return int(completion_prev_machine[-1])


class Operators:
    """Collection of local search and perturbation operators for PFSP."""

    def __init__(self, p_times: np.ndarray):
        self.p_times = p_times

    def first_improvement_relocate(self, perm: List[int]) -> Optional[Tuple[List[int], int]]:
        """Perform the 1–insert (relocate) move using first improvement.

        Parameters
        ----------
        perm : List[int]
            Current permutation of jobs.

        Returns
        -------
        Optional[Tuple[List[int], int]]
            A tuple ``(new_perm, new_makespan)`` if an improving neighbour is
            found; otherwise ``None``.
        """
        n = len(perm)
        best_value = makespan(perm, self.p_times)
        for i in range(n):
            job = perm[i]
            for k in range(n):
                if k == i:
                    continue
                # Construct neighbour by relocating job at pos i to pos k
                new_perm = perm.copy()
                # Remove job
                new_perm.pop(i)
                # Insert at new position
                new_perm.insert(k, job)
                val = makespan(new_perm, self.p_times)
                if val < best_value:
                    return new_perm, val
        return None

    def first_improvement_swap(self, perm: List[int]) -> Optional[Tuple[List[int], int]]:
        """Perform the 2–swap move using first improvement.

        Parameters
        ----------
        perm : List[int]
            Current permutation of jobs.

        Returns
        -------
        Optional[Tuple[List[int], int]]
            A tuple ``(new_perm, new_makespan)`` if an improving neighbour is
            found; otherwise ``None``.
        """
        n = len(perm)
        best_value = makespan(perm, self.p_times)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if perm[i] == perm[j]:
                    continue
                new_perm = perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                val = makespan(new_perm, self.p_times)
                if val < best_value:
                    return new_perm, val
        return None

    def perturb_block_insert(self, perm: List[int], block_lengths: Tuple[int, ...] = (2, 3)) -> Tuple[List[int], int]:
        """Apply the block insert perturbation.

        Removes a contiguous block of random length (2 or 3 by default) and
        reinserts it at a random position.  The resulting permutation is
        always returned, regardless of whether it improves the makespan.

        Parameters
        ----------
        perm : List[int]
            Current permutation of jobs.
        block_lengths : Tuple[int, ...]
            Possible block lengths to choose from.  Defaults to (2, 3).

        Returns
        -------
        Tuple[List[int], int]
            A tuple ``(new_perm, new_makespan)`` representing the perturbed
            solution and its makespan.
        """
        n = len(perm)
        if n <= 1:
            return perm, makespan(perm, self.p_times)
        length = random.choice(block_lengths)
        length = min(length, n)  # ensure block length does not exceed permutation
        start_idx = random.randint(0, n - length)
        block = perm[start_idx:start_idx + length]
        remainder = perm[:start_idx] + perm[start_idx + length:]
        insert_pos = random.randint(0, len(remainder))
        new_perm = remainder[:insert_pos] + block + remainder[insert_pos:]
        return new_perm, makespan(new_perm, self.p_times)

    def first_improvement_block(self, perm: List[int], block_lengths: Tuple[int, ...] = (2, 3)) -> Optional[Tuple[List[int], int]]:
        """Perform a block insert move using first improvement.

        Scans over possible block lengths and start/end positions, reinserting the
        block at every other position, and returns the first solution that
        improves the makespan.

        Parameters
        ----------
        perm : List[int]
            Current permutation of jobs.
        block_lengths : Tuple[int, ...]
            A tuple of block lengths to consider (typically (2, 3)).

        Returns
        -------
        Optional[Tuple[List[int], int]]
            ``(new_perm, new_makespan)`` if an improving neighbour is found;
            otherwise ``None``.
        """
        n = len(perm)
        best_value = makespan(perm, self.p_times)
        # Iterate over block lengths
        for L in block_lengths:
            if L > n:
                continue
            # Loop over start index of block
            for start in range(n - L + 1):
                block = perm[start:start + L]
                remainder = perm[:start] + perm[start + L:]
                # Loop over insertion positions in remainder
                for insert_pos in range(len(remainder) + 1):
                    if insert_pos == start:
                        # Reinserting at original position yields original perm
                        continue
                    new_perm = remainder[:insert_pos] + block + remainder[insert_pos:]
                    val = makespan(new_perm, self.p_times)
                    if val < best_value:
                        return new_perm, val
        return None
