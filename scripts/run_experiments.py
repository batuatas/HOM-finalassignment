#!/usr/bin/env python3
"""Run PFSP metaheuristic experiments from the command line.

This script reads instances from an Excel workbook and executes the
Iterated Greedy / Local Search metaheuristic using one of the scheduling
mechanisms defined for the final assignment.  Compared to the starter
version, the script now supports loading best known makespans, printing
tabular summaries and writing enriched CSV outputs that include RPD and
iteration counts.

Example
-------

```
python scripts/run_experiments.py --instances-file data/Instances.xlsx \
    --mechanism adaptive --runs 3 --output-dir results/adaptive
```
"""

import argparse
from pathlib import Path

from pfsp.design import describe_design
from pfsp.instance import attach_best_known, load_best_known, read_instances
from pfsp.mechanisms import available_mechanisms
from pfsp.runner import run_experiments
from pfsp.reporting import add_rpd_column, summarise_by_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PFSP IG/ILS experiments")
    parser.add_argument(
        "--instances-file",
        type=str,
        required=True,
        help="Path to Excel file with PFSP instances (each sheet is an instance)",
    )
    mechanisms = available_mechanisms()
    parser.add_argument(
        "--mechanism",
        type=str,
        choices=sorted(mechanisms.keys()),
        default="fixed",
        help="Operator scheduling mechanism to use",
        default="fixed",
        help=(
            "Operator scheduling mechanism to use.  Supports 'fixed', 'adaptive' and the "
            "aliases understood by pfsp.mechanisms.build_mechanism."
        ),
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
        "--bks-file",
        type=str,
        default=None,
        help="Optional CSV with columns 'instance' and 'best_makespan'",
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
        "--learning-rate",
        type=float,
        default=0.2,
        help="Learning rate for adaptive pursuit scheduler (ignored for fixed)",
    )
    parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Block lengths for block operator and perturbation",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregated summary statistics after running experiments",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the design summary for the selected mechanism and exit",
    )
    args = parser.parse_args()
    if args.describe:
        print(describe_design(args.mechanism))
        return
    # Read instances
    instances = read_instances(args.instances_file)
    best_known = None
    if args.bks_file:
        best_known = load_best_known(args.bks_file)
        attach_best_known(instances, best_known)
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
        learning_rate=args.learning_rate,
        block_lengths=tuple(args.block_lengths),
        seed=args.seed,
    )
    # Ensure RPD column is up to date
    results = add_rpd_column(results, best_known)
    # Ensure output directory exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Wrote results to {csv_path}")
    # Print summary
    if args.summary:
        summary = summarise_by_instance(results)
        print("Summary by mechanism and instance:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()