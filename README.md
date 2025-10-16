# HOM Final Assignment - Heuristic Optimization for PFSP

This repository contains a complete codebase for the heuristic optimisation final assignment.
The objective is to design and evaluate single‐solution metaheuristics for the Permutation Flow Shop
Problem (PFSP), using both deterministic and adaptive operator scheduling mechanisms.  

The project is organised to make it easy to reproduce experiments and extend the code for your own
research.  It includes scripts to parse provided instances, implement the proposed metaheuristic,
run controlled experiments and generate plots.  The repository is intentionally lightweight and
Pythonic – there are no heavy frameworks and the only dependencies are NumPy, pandas and matplotlib.

## Contents

* `data/raw/` – the original `.txt` instance files supplied with the assignment.
* `data/Instances.xlsx` – an Excel workbook where each sheet corresponds to a PFSP instance.
  A conversion script is provided to build this workbook from the raw instances.
* `src/` – all source code.  In particular, the `pfsp` package contains:
  * `instance.py` – functions for reading instances from the Excel file and applying
    best-known makespans loaded from CSV.
  * `operators.py` – definitions of the local search and perturbation operators (1‐insert,
    2‐swap, block‐insert).
  * `scheduler.py` – operator scheduling mechanisms: a fixed sequence (Mechanism 1A) and
    an adaptive scheduler (Mechanism 2A) with credit assignment and probability matching.
  * `mechanisms.py` – declarative configuration for the assignment mechanisms, providing
    factories that build the appropriate scheduler.
  * `algo_ig_ils.py` – an Iterated Greedy/Iterated Local Search metaheuristic for PFSP
    implementing Mechanism 1A or 2A and returning structured run statistics.
  * `runner.py` – a high‐level experiment runner that executes multiple runs on a set of
    instances, validates mechanism names and records the real iteration counts.
  * `reporting.py` – helpers for computing Relative Percent Deviation (RPD) and tabular
    summaries.
* `scripts/` – useful command line entry points:
  * `convert_instances.py` – parse the provided `.txt` files and generate
    `data/Instances.xlsx` with the required sheet names.
  * `run_experiments.py` – orchestrate a series of experiments on all instances in
    `data/Instances.xlsx` and produce enriched CSV files (including RPD, best known values
    and iteration counts).  It can optionally print per-instance summaries and accepts a
    best-known makespan CSV.
  * `compare_mechanisms.py` – evaluate multiple mechanisms in a single command and write a
    combined summary table to disk.
  * `add_rpd.py` – post-process a results CSV to append Relative Percent Deviation values
    given a file of best-known makespans.

## Quickstart

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Convert the raw `.txt` instances into the Excel workbook:

```bash
python scripts/convert_instances.py --input-dir data/raw --output data/Instances.xlsx
```

3. Run experiments (this may take several minutes depending on your hardware):

```bash
# Optionally provide a CSV (instance,best_makespan) to compute RPD values
python scripts/run_experiments.py \
    --instances-file data/Instances.xlsx \
    --bks-file data/best_known.csv \
    --mechanism adaptive --runs 5 \
    --summary --output-dir results/adaptive

# Evaluate both mechanisms and generate a combined summary
python scripts/compare_mechanisms.py \
    --instances-file data/Instances.xlsx \
    --mechanisms fixed adaptive \
    --runs 5 --summary --output-dir results/compare
```

4. The `results/` directory will contain CSV summaries, convergence logs and plot images
   comparing the deterministic and adaptive variants.

## Notes

* The code uses NumPy for efficient makespan calculation and incremental updates.  All
  random number generators are seeded for reproducibility when required.
* The adaptive scheduler follows the probability matching approach described in Lecture 6:
  after each operator application a reward is computed based on the relative improvement,
  credits are updated using a sliding window and operator selection probabilities are
  recomputed.  The `pfsp/mechanisms.py` module exposes the deterministic and adaptive
  variants declaratively so you can add new mechanisms without changing the rest of the
  codebase.
* No external optimisation libraries are used – the algorithms are implemented from
  scratch so that you can easily modify or extend them.

Please read the source code and inline comments for further details.  If you encounter
any issues or have suggestions for improvement, feel free to open an issue or submit a
pull request.