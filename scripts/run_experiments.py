# scripts/run_experiments.py
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# make src importable
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from pfsp.instance import read_instances, load_best_known, attach_best_known
from pfsp.algo_ig_ils import IteratedGreedyILS
from pfsp.reporting import add_rpd_column, summarise_by_instance


def run_single(instance_name: str, p_times, mechanism: str, seed: int, time_limit: float,
               algo_kwargs: dict, run_kwargs: dict):
    t0 = time.time()
    algo = IteratedGreedyILS(p_times, mechanism=mechanism, seed=seed, **algo_kwargs)
    res = algo.run(time_limit=time_limit, **run_kwargs)
    elapsed = time.time() - t0
    return {
        "instance": instance_name,
        "mechanism_key": mechanism,
        "seed": seed,
        "makespan": res.makespan,
        "iterations": res.iterations,
        "elapsed": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--xlsx", type=str, default="data/Instances.xlsx")
    p.add_argument("--best-known", type=str, default="data/best_known.csv")
    p.add_argument("--instances", type=str, default="", help="Comma-separated sheet names to run (subset)")
    p.add_argument("--list-instances", action="store_true")
    # mechanisms + runtime
    p.add_argument("--mechanisms", type=str, default="fixed,adaptive")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--time-limit", type=float, default=10.0, help="seconds per run")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--verbose", action="store_true")

    # ---- NEW: algorithm hyperparameters ----
    # scheduler (2B)
    p.add_argument("--window-size", type=int, default=50)
    p.add_argument("--p-min", type=float, default=0.10)
    p.add_argument("--learning-rate", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.60)
    p.add_argument("--episode-len", type=int, default=50)
    # search
    p.add_argument("--block-lengths", type=str, default="2,3")
    p.add_argument("--d-frac", type=float, default=0.25)
    p.add_argument("--ls-step-cap", type=int, default=2000)
    p.add_argument("--ls-stagnation", type=int, default=200)
    # run-level
    p.add_argument("--max-iter", type=int, default=10_000)
    p.add_argument("--max-no-improve", type=int, default=200)

    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.list_instances:
        xl = pd.ExcelFile(args.xlsx, engine="openpyxl")
        print("Sheets:", ", ".join(xl.sheet_names))
        print(f"Total: {len(xl.sheet_names)}")
        return

    if args.verbose:
        print("[*] Loading instances (verbose)…")
    insts = read_instances(args.xlsx, verbose=args.verbose)

    try:
        bk = load_best_known(args.best_known)
    except Exception as e:
        if args.verbose:
            print(f"[warn] Could not load best-known from {args.best_known}: {e}")
        bk = {}
    attach_best_known(insts, bk)

    mechanisms = [x.strip() for x in args.mechanisms.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    if args.instances.strip():
        wanted = {s.strip() for s in args.instances.split(",")}
        insts = {k: v for k, v in insts.items() if k in wanted}
        if args.verbose:
            print(f"[*] Subset selected: {', '.join(insts.keys())}")

    # Parse block lengths
    try:
        block_lengths = tuple(int(x) for x in args.block_lengths.split(",") if x.strip())
    except Exception:
        block_lengths = (2, 3)

    algo_kwargs = dict(
        window_size=args.window_size,
        p_min=args.p_min,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        episode_len=args.episode_len,
        block_lengths=block_lengths,
        d_frac=args.d_frac,
        ls_step_cap=args.ls_step_cap,
        ls_stagnation_limit=args.ls_stagnation,
    )
    run_kwargs = dict(
        max_iter=args.max_iter,
        max_no_improve=args.max_no_improve,
        verbose=args.verbose,
    )

    rows = []
    for name, inst in insts.items():
        for mech in mechanisms:
            for sd in seeds:
                if args.verbose:
                    print(f"-> {name} | {mech} | seed={sd} "
                          f"(t≤{args.time_limit}s, max_iter={args.max_iter}, max_no_improve={args.max_no_improve})")
                r = run_single(name, inst.p_times, mech, sd, time_limit=args.time_limit,
                               algo_kwargs=algo_kwargs, run_kwargs=run_kwargs)
                rows.append(r)
                print(f"{name} | {mech} | seed={sd} -> {r['makespan']} in {r['elapsed']:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "raw.csv", index=False)

    if bk:
        df["best_known"] = df["instance"].map(bk)
        df = add_rpd_column(df, best_known=bk)
        df.to_csv(outdir / "raw_with_rpd.csv", index=False)

    summ = summarise_by_instance(df)
    summ.to_csv(outdir / "summary_by_instance.csv", index=False)

    if "rpd" in df.columns:
        overall = df.groupby("mechanism_key").agg(
            makespan_mean=("makespan", "mean"),
            elapsed_mean=("elapsed", "mean"),
            rpd_mean=("rpd", "mean"),
        )
    else:
        overall = df.groupby("mechanism_key").agg(
            makespan_mean=("makespan", "mean"),
            elapsed_mean=("elapsed", "mean"),
        )
    overall.to_csv(outdir / "overall.csv")

    meta = {
        "args": vars(args),
        "n_instances": len(insts),
        "mechanisms": mechanisms,
        "seeds": seeds,
        "algo_defaults": algo_kwargs,
        "run_defaults": run_kwargs,
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
