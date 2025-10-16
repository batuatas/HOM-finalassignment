#!/usr/bin/env python3
"""Run PFSP metaheuristic experiments from the command line.

Reads instances from an Excel workbook and executes the IG/ILS metaheuristic
using one mechanism. Can optionally log convergence for every run as CSVs.

Example
-------
python scripts/run_experiments.py --instances-file data/Instances.xlsx \
    --mechanism adaptive --runs 3 --log-progress --output-dir results/adaptive
"""
import argparse
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run PFSP IG/ILS experiments")
    parser.add_argument("--instances-file", type=str, required=True,
                        help="Path to Excel file with PFSP instances (each sheet is an instance)")
    mechanisms = available_mechanisms()
    parser.add_argument("--mechanism", type=str, choices=sorted(mechanisms.keys()), default="fixed",
                        help="Operator scheduling mechanism to use")
    parser.add_argument("--runs", type=int, default=3, help="Independent runs per instance")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max ILS iterations per run")
    parser.add_argument("--max-no-improve", type=int, default=50,
                        help="Max consecutive non-improving iterations before stopping")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Time limit in seconds per run (optional)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write CSV results and logs")
    parser.add_argument("--bks-file", type=str, default=None,
                        help="Optional CSV with columns 'instance' and 'best_makespan'")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for reproducibility (optional)")
    # Mechanism 2B / scheduler knobs (ignored by fixed)
    parser.add_argument("--window-size", type=int, default=50,
                        help="Sliding window size for adaptive scheduler (ignored for fixed)")
    parser.add_argument("--p-min", type=float, default=0.1,
                        help="Minimum probability / epsilon (ignored for fixed)")
    parser.add_argument("--learning-rate", type=float, default=0.2,
                        help="Learning rate (ignored for fixed)")
    parser.add_argument("--gamma", type=float, default=0.60,
                        help="Discount factor for Q-learning (ignored for fixed)")
    parser.add_argument("--episode-len", type=int, default=10,
                        help="Steps per episode for Q-learning (ignored for fixed)")
    parser.add_argument("--block-lengths", type=int, nargs="*", default=[2, 3],
                        help="Block lengths for block operator and perturbation")
    # Progress logging
    parser.add_argument("--log-progress", action="store_true",
                        help="Write per-run convergence CSVs to <output-dir>/convergence/<mechanism>/")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Record a progress row every N iterations (and on improvements)")
    # Describe mode
    parser.add_argument("--describe", action="store_true",
                        help="Print the design summary for the selected mechanism and exit")
    args = parser.parse_args()

    if args.describe:
        print(describe_design(args.mechanism))
        return

    instances = read_instances(args.instances_file)
    best_known = None
    if args.bks_file:
        best_known = load_best_known(args.bks_file)
        attach_best_known(instances, best_known)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        # new:
        log_progress=args.log_progress,
        log_dir=str(out_dir),
        progress_every=args.progress_every,
        # Q-learning extras are forwarded through options dict by mechanisms.py
        gamma=args.gamma,
        episode_len=args.episode_len,
    )

    results = add_rpd_column(results, best_known)
    csv_path = out_dir / "results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Wrote results to {csv_path}")

    # Optional aggregated summary
    summary = summarise_by_instance(results)
    (out_dir / "summary.csv").write_text(summary.to_csv(index=False))
    print("Summary by mechanism and instance:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
