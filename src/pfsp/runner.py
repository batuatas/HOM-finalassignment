"""Experiment runner for the PFSP metaheuristic.

Returns a tidy DataFrame and (optionally) writes per-run convergence CSVs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
    # NEW: progress logging
    log_progress: bool = False,
    log_dir: Optional[str] = None,
    progress_every: int = 10,
    # NEW: extra Q-learning knobs (ignored by fixed)
    gamma: float = 0.60,
    episode_len: int = 10,
) -> pd.DataFrame:
    """Execute multiple runs of the IG/ILS algorithm on a set of instances.

    Returns a DataFrame with one row per run per instance, plus optional
    convergence CSVs under <log_dir>/convergence/<mechanism>/ if enabled.
    """
    records: List[dict] = []
    spec = get_mechanism(mechanism)
    conv_base: Optional[Path] = None
    if log_progress and log_dir:
        conv_base = Path(log_dir) / "convergence" / mechanism
        conv_base.mkdir(parents=True, exist_ok=True)

    for inst_name, inst in instances.items():
        for run_idx in range(runs):
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
            convergence_rows: List[dict] = []

            def _progress_cb(iter_no: int, best_val: int) -> None:
                # called by solver on every improvement and every N iterations
                convergence_rows.append(
                    {
                        "instance": inst_name,
                        "mechanism": mechanism,
                        "run": run_idx,
                        "iter": iter_no,
                        "elapsed": time.time() - start_time,
                        "best_makespan": int(best_val),
                        "seed": run_seed,
                    }
                )

            result: IGILSResult = solver.run(
                max_iter=max_iter,
                max_no_improve=max_no_improve,
                time_limit=time_limit,
                verbose=False,
                progress_cb=_progress_cb if log_progress else None,
                progress_every=max(1, int(progress_every)),
            )
            elapsed = time.time() - start_time
            best_val = result.makespan
            best_known = inst.best_makespan
            rpd = (
                100.0 * (best_val - best_known) / best_known
                if (best_known is not None and best_known > 0)
                else None
            )

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
                    "iterations": int(result.iterations),
                    "seed": run_seed,
                }
            )

            # Write convergence CSV for this run, if requested
            if log_progress and conv_base is not None and convergence_rows:
                out_path = conv_base / f"{inst_name}_run{run_idx}.csv"
                pd.DataFrame(convergence_rows).to_csv(out_path, index=False)

    return pd.DataFrame.from_records(records)
