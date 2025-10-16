"""Experiment runner for the PFSP metaheuristic.

This module provides a convenient function ``run_experiments`` that executes
multiple runs of either the fixed or adaptive pursuit IG/ILS metaheuristic on a
collection of instances.  Results are returned as a pandas DataFrame for
further analysis or plotting.

Usage
-----

```
from pfsp.instance import read_instances
from pfsp.runner import run_experiments

instances = read_instances("data/Instances.xlsx")
df = run_experiments(instances, mechanism="adaptive", runs=3, max_iter=500)
df.to_csv("results.csv", index=False)
```

Each row in the returned DataFrame corresponds to a single run on a single
instance and includes the algorithm, instance name, run index, makespan,
best known makespan (if provided in the Instance object), relative percent
deviation (RPD) if computable, elapsed time and number of iterations.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .instance import Instance
from .algo_ig_ils import IGILSResult, IteratedGreedyILS
from .mechanisms import get_mechanism


def run_experiments(
    instances: Dict[str, Instance],
    mechanism: str = "fixed",
    runs: int = 3,
    max_iter: int = 1000,
    max_no_improve: int = 50,
    time_limit: Optional[float] = None,
    window_size: int = 50,
    p_min: float = 0.1,
    learning_rate: float = 0.2,
    block_lengths: tuple = (2, 3),
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Execute multiple runs of the IG/ILS algorithm on a set of instances.

    Parameters
    ----------
    instances : Dict[str, Instance]
        A dictionary mapping instance names to ``Instance`` objects.
    mechanism : str, optional
        Scheduling mechanism to use: ``'fixed'`` or ``'adaptive'``.
        Defaults to ``'fixed'``.
    runs : int, optional
        Number of independent runs per instance.  Defaults to 3.
    max_iter : int, optional
        Maximum number of ILS iterations per run.  Defaults to 1000.
    max_no_improve : int, optional
        Maximum consecutive iterations without improvement.  Defaults to 50.
    time_limit : float, optional
        Time limit in seconds for each run.  If ``None``, no time limit is
        enforced.  Defaults to ``None``.
    window_size : int, optional
        Sliding window size for the adaptive scheduler.  Ignored for fixed
        scheduler.  Default is 50.
    p_min : float, optional
        Minimum probability for each operator in the adaptive scheduler.
        Ignored for fixed scheduler.  Default is 0.1.
    learning_rate : float, optional
        Learning rate for adaptive pursuit probability updates.  Ignored for
        fixed scheduler.  Default is 0.2.
    block_lengths : tuple, optional
        Block lengths for the block operator and perturbation.  Default is
        ``(2, 3)``.
    seed : Optional[int], optional
        Base random seed.  If provided, each run will be seeded with
        ``seed + run_index`` to ensure reproducibility across runs.  Defaults
        to ``None`` (non‐deterministic behaviour).

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per run per instance and the following
        columns: ``algorithm``, ``instance``, ``run``, ``makespan``,
        ``best_known``, ``rpd`` (relative percent deviation), ``elapsed`` and
        ``iterations`` (number of ILS iterations actually executed).
    """
    records: List[dict] = []
    spec = get_mechanism(mechanism)

    for inst_name, inst in instances.items():
        for run_idx in range(runs):
            # Seed per run (if base seed provided)
            run_seed = seed + run_idx if seed is not None else None
            solver = IteratedGreedyILS(
                inst.p_times,
                mechanism=mechanism,
                window_size=window_size,
                p_min=p_min,
                learning_rate=learning_rate,
                block_lengths=block_lengths,
                seed=run_seed,
            )
            start_time = time.time()
            result: IGILSResult = solver.run(
                max_iter=max_iter,
                max_no_improve=max_no_improve,
                time_limit=time_limit,
                verbose=False,
            )
            elapsed = time.time() - start_time
            best_val = result.makespan
            best_known = inst.best_makespan
            if best_known and best_known > 0:
                rpd = 100.0 * (best_val - best_known) / best_known
            else:
                rpd = None
            records.append(
                {
                    "algorithm": mechanism,
                    "mechanism_key": mechanism,
                    "mechanism_label": spec.design.identifier,
                    "instance": inst_name,
                    "run": run_idx,
                    "makespan": best_val,
                    "best_known": best_known,
                    "rpd": rpd,
                    "elapsed": elapsed,
                    "iterations": result.iterations,
                }
            )
    return pd.DataFrame.from_records(records)
