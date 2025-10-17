# src/pfsp/instance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Optional
import os
import numpy as np
import pandas as pd

@dataclass
class Instance:
    name: str
    p_times: np.ndarray  # shape: (machines, jobs)
    best_makespan: Optional[int] = None
    @property
    def m(self) -> int: return self.p_times.shape[0]
    @property
    def n(self) -> int: return self.p_times.shape[1]

def _looks_like_ints(values: list) -> bool:
    try:
        _ = [int(x) for x in values]
        return True
    except Exception:
        return False

def read_instances(xlsx_path: str, verbose: bool = False) -> Dict[str, Instance]:
    # Force engine to openpyxl to avoid hangs/ambiguous detection
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    out: Dict[str, Instance] = {}
    total = len(xl.sheet_names)
    for idx, sheet in enumerate(xl.sheet_names, start=1):
        if verbose:
            print(f"[read] {idx}/{total} {sheet}")
        df = xl.parse(sheet, header=None)
        header = [x for x in df.iloc[0].tolist() if pd.notna(x)]
        if len(header) < 2 or not _looks_like_ints(header[:2]):
            raise ValueError(f"Sheet '{sheet}' must start with integer header [M, N]. Got: {header}")
        m = int(header[0]); n = int(header[1])

        body = df.iloc[1:1+m]
        row_vals = [[x for x in row.tolist() if pd.notna(x)] for _, row in body.iterrows()]

        # Plain MxN matrix
        if all(len(rv) == n for rv in row_vals):
            p_times = np.zeros((m, n), dtype=int)
            for i in range(m):
                p_times[i, :] = np.array([int(v) for v in row_vals[i]], dtype=int)
            out[sheet] = Instance(name=sheet, p_times=p_times)
            continue

        # Pairs (2N)
        if not all(len(rv) == 2*n for rv in row_vals):
            raise ValueError(
                f"Sheet '{sheet}' has invalid row lengths. Expected {n} or {2*n} integers per machine row."
            )

        p_times = np.zeros((m, n), dtype=int)
        sample_job_ids = [int(row_vals[0][2*j]) for j in range(n)]
        one_based = (min(sample_job_ids) == 1) and (max(sample_job_ids) == n)
        for i in range(m):
            row = row_vals[i]
            for j in range(n):
                job_id = int(row[2*j])
                if one_based: job_id -= 1
                if job_id < 0 or job_id >= n:
                    raise ValueError(f"Invalid job id {job_id} in sheet '{sheet}', machine {i}")
                p_time = int(row[2*j + 1])
                p_times[i, job_id] = p_time

        out[sheet] = Instance(name=sheet, p_times=p_times)
    return out

def read_raw_instance(path: str) -> Instance:
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    m, n = map(int, lines[0].split()[:2])
    rows = [list(map(int, ln.split())) for ln in lines[1:1+m]]
    p_times = np.zeros((m, n), dtype=int)
    if all(len(r) == n for r in rows):
        for i in range(m): p_times[i, :] = rows[i]
    elif all(len(r) == 2*n for r in rows):
        ids = [rows[0][2*j] for j in range(n)]
        one_based = (min(ids) == 1 and max(ids) == n)
        for i in range(m):
            r = rows[i]
            for j in range(n):
                job = r[2*j] - 1 if one_based else r[2*j]
                p_times[i, job] = r[2*j + 1]
    else:
        raise ValueError(f"Unrecognized raw instance row length in {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    return Instance(name=name, p_times=p_times)

def load_best_known(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    if not {"instance", "best_makespan"} <= set(df.columns):
        raise ValueError("best_known.csv must have columns: instance,best_makespan")
    return (
        df[["instance", "best_makespan"]]
        .dropna()
        .set_index("instance")["best_makespan"]
        .astype(int)
        .to_dict()
    )

def attach_best_known(instances: Dict[str, Instance], best_known: Mapping[str, int]) -> None:
    for name, val in best_known.items():
        if name in instances:
            instances[name].best_makespan = int(val)
