#!/usr/bin/env python3
"""Run PFSP metaheuristic experiments from the command line.

This script reads instances from an Excel workbook and executes either the
fixed or adaptive IG/ILS algorithm on each instance for a specified
number of independent runs.  Results are aggregated into a CSV file in
the output directory.  Optionally, basic summary statistics are printed
to stdout.

Example
-------

```
python scripts/run_experiments.py --instances-file data/Instances.xlsx \
    --mechanism adaptive --runs 3 --output-dir results/adaptive
```
"""

import argparse
from pathlib import Path
import os
import pandas as pd

from pfsp.instance import read_instances
from pfsp.runner import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PFSP IG/ILS experiments")
    parser.add_argument(
        "--instances-file",
        type=str,
        required=True,
        help="Path to Excel file with PFSP instances (each sheet is an instance)",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Operator scheduling mechanism to use (fixed or adaptive)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs per instance",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of ILS iterations per run",
    )
    parser.add_argument(
        "--max-no-improve",
        type=int,
        default=50,
        help="Maximum consecutive non-improving iterations before stopping",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit in seconds per run (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write CSV results and optional plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Sliding window size for adaptive scheduler (ignored for fixed)",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=0.1,
        help="Minimum probability for adaptive scheduler (ignored for fixed)",
    )
    parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Block lengths for block operator and perturbation",
    )
    args = parser.parse_args()
    # Read instances
    instances = read_instances(args.instances_file)
    # Run experiments
    results = run_experiments(
        instances=instances,
        mechanism=args.mechanism,
        runs=args.runs,
        max_iter=args.max_iter,
        max_no_improve=args.max_no_improve,
        time_limit=args.time_limit,
        window_size=args.window_size,
        p_min=args.p_min,
        block_lengths=tuple(args.block_lengths),
        seed=args.seed,
    )
    # Ensure output directory exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Wrote results to {csv_path}")
    # Print summary
    if "rpd" in results.columns and results["rpd"].notna().any():
        summary = results.groupby("instance")["rpd"].mean()
        print("Average RPD by instance:")
        print(summary)


if __name__ == "__main__":
    main()