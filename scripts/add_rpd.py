#!/usr/bin/env python3
"""Augment a results CSV with RPD information."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pfsp.instance import load_best_known
from pfsp.reporting import add_rpd_column


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add relative percent deviation (RPD) values to an experiment CSV"
    )
    parser.add_argument("--results", type=str, required=True, help="Input results CSV")
    parser.add_argument(
        "--bks-file",
        type=str,
        required=True,
        help="CSV with columns 'instance' and 'best_makespan'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path (defaults to overwriting the input file)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    best_known = load_best_known(args.bks_file)
    enriched = add_rpd_column(df, best_known)
    output_path = Path(args.output) if args.output else Path(args.results)
    enriched.to_csv(output_path, index=False)
    print(f"Wrote enriched results to {output_path}")


if __name__ == "__main__":
    main()
