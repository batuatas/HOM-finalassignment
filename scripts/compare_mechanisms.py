# compare_mechanisms.py
from __future__ import annotations
import pandas as pd

def compare(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    # pivot to see mean per instance x mechanism
    pt = df.pivot_table(index="instance", columns="mechanism_key", values="makespan", aggfunc="mean")
    if "fixed" in pt.columns and "adaptive" in pt.columns:
        pt["diff_adaptive_minus_fixed"] = pt["adaptive"] - pt["fixed"]
    return pt.reset_index()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="comparison.csv")
    args = ap.parse_args()
    comp = compare(args.raw)
    comp.to_csv(args.out, index=False)
    print("Saved", args.out)
