#!/usr/bin/env python3
"""Compare multiple PFSP mechanisms in a single run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from pfsp.design import describe_design
from pfsp.instance import attach_best_known, load_best_known, read_instances
from pfsp.mechanisms import available_mechanisms
from pfsp.runner import run_experiments
from pfsp.reporting import add_rpd_column, summarise_by_instance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PFSP IG/ILS experiments for multiple mechanisms and aggregate results"
    )
    parser.add_argument("--instances-file", type=str, required=True,
                        help="Path to Excel file with PFSP instances")
    mechanisms = available_mechanisms()
    parser.add_argument("--mechanisms", type=str, nargs="*", default=list(mechanisms.keys()),
                        help="Mechanisms to evaluate")
    parser.add_argument("--runs", type=int, default=3, help="Runs per instance")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max ILS iterations per run")
    parser.add_argument("--max-no-improve", type=int, default=50,
                        help="Max consecutive non-improving iterations before stopping")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit per run (seconds)")
    parser.add_argument("--window-size", type=int, default=50, help="Adaptive/QL window size")
    parser.add_argument("--p-min", type=float, default=0.1, help="Min prob / epsilon")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.60, help="Discount factor (Q-learning)")
    parser.add_argument("--episode-len", type=int, default=10, help="Episode length (Q-learning)")
    parser.add_argument("--block-lengths", type=int, nargs="*", default=[2, 3],
                        help="Block lengths for perturbation")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--bks-file", type=str, default=None,
                        help="Optional CSV with columns 'instance' and 'best_makespan'")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where result CSVs will be written")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics and write summary.csv")
    parser.add_argument("--log-progress", action="store_true",
                        help="Write per-run convergence CSVs under <output-dir>/convergence/<mechanism>/")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Record a progress row every N iterations (and on improvements)")
    parser.add_argument("--describe", action="store_true",
                        help="Print the design description of the requested mechanisms and exit")
    args = parser.parse_args()

    requested = args.mechanisms or list(mechanisms.keys())
    invalid = sorted(set(requested).difference(mechanisms))
    if invalid:
        raise ValueError(
            f"Unknown mechanisms requested: {', '.join(invalid)}. Available: {', '.join(mechanisms)}"
        )

    if args.describe:
        for mech in requested:
            print(describe_design(mech))
            print()
        return

    instances = read_instances(args.instances_file)
    best_known = None
    if args.bks_file:
        best_known = load_best_known(args.bks_file)
        attach_best_known(instances, best_known)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[pd.DataFrame] = []
    for mech in requested:
        df = run_experiments(
            instances=instances,
            mechanism=mech,
            runs=args.runs,
            max_iter=args.max_iter,
            max_no_improve=args.max_no_improve,
            time_limit=args.time_limit,
            window_size=args.window_size,
            p_min=args.p_min,
            learning_rate=args.learning_rate,
            block_lengths=tuple(args.block_lengths),
            seed=args.seed,
            # new:
            log_progress=args.log_progress,
            log_dir=str(out_dir),
            progress_every=args.progress_every,
            gamma=args.gamma,
            episode_len=args.episode_len,
        )
        df = add_rpd_column(df, best_known)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = out_dir / "results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote combined results to {combined_path}")

    if args.summary:
        summary = summarise_by_instance(combined)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary table to {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
