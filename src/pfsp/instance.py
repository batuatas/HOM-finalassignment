"""Instance parsing utilities for the Permutation Flow Shop Problem.

The assignment requires reading a collection of PFSP instances from a single
Excel workbook where each sheet corresponds to one instance.  Each sheet
contains a processing time matrix with dimensions (machines × jobs), along
with the header specifying the number of machines and jobs.

This module defines a simple ``Instance`` dataclass to store the processing
time matrix and optional metadata, and a function ``read_instances`` to
load all instances from an Excel file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional
import numpy as np
import pandas as pd


@dataclass
class Instance:
    """Represents a PFSP instance.

    Attributes
    ----------
    name : str
        The name of the instance (typically the sheet name).
    p_times : np.ndarray
        Processing time matrix of shape (machines, jobs).  ``p_times[i, j]``
        stores the processing time of job ``j`` on machine ``i``.
    best_makespan : Optional[int]
        The best known makespan value for this instance, if available.  If
        unknown, this value is ``None``.  The assignment uses ``gap``
        suffixes (e.g. ``_Gap.txt``) to indicate known optimal solutions,
        but those values are not directly contained in the raw instance
        files; thus by default this field is ``None``.
    """

    name: str
    p_times: np.ndarray
    best_makespan: Optional[int] = None

    @property
    def m(self) -> int:
        """Number of machines."""
        return self.p_times.shape[0]

    @property
    def n(self) -> int:
        """Number of jobs."""
        return self.p_times.shape[1]


def read_instances(xlsx_path: str) -> Dict[str, Instance]:
    """Read all PFSP instances from an Excel workbook.

    Parameters
    ----------
    xlsx_path : str
        Path to the Excel file containing the instances.  Each sheet is
        expected to have the following structure:

        * The first row contains two integers: ``M`` (number of machines)
          and ``N`` (number of jobs).
        * The next ``M`` rows each contain ``2N`` integers: pairs of job
          indices and processing times.  Job indices should be in
          ascending order from 0 to ``N-1``.

    Returns
    -------
    Dict[str, Instance]
        A mapping from sheet names to ``Instance`` objects.
    """
    xl = pd.ExcelFile(xlsx_path)
    instances: Dict[str, Instance] = {}
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, header=None)
        # The header row should contain two integers M and N
        header = df.iloc[0].dropna().tolist()
        if len(header) < 2:
            raise ValueError(
                f"Sheet '{sheet_name}' does not contain valid header (M N)")
        m = int(header[0])
        n = int(header[1])
        # Prepare processing time matrix
        p_times = np.zeros((m, n), dtype=int)
        for i in range(m):
            row = df.iloc[i + 1].dropna().tolist()
            if len(row) != 2 * n:
                raise ValueError(
                    f"Row {i + 1} in sheet '{sheet_name}' has {len(row)} values, expected {2 * n}")
            # row contains pairs: job_id, processing_time
            for j in range(n):
                job_id = int(row[2 * j])
                p_time = int(row[2 * j + 1])
                # job ids are zero-based according to the instance format
                if job_id < 0 or job_id >= n:
                    raise ValueError(
                        f"Invalid job id {job_id} in sheet '{sheet_name}', row {i + 1}")
                p_times[i, job_id] = p_time
        instances[sheet_name] = Instance(name=sheet_name, p_times=p_times)
    return instances


def read_raw_instance(path: str) -> Instance:
    """Parse a single raw instance file in the VFR format.

    Parameters
    ----------
    path : str
        Path to the `.txt` file containing the raw instance.

    Returns
    -------
    Instance
        An ``Instance`` object representing the parsed instance.
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid header line in {path}")
    m = int(header[0])
    n = int(header[1])
    p_times = np.zeros((m, n), dtype=int)
    # There should be m lines with 2n integers each
    if len(lines) - 1 < m:
        raise ValueError(f"Expected at least {m} lines of data in {path}")
    for i in range(m):
        parts = lines[i + 1].split()
        if len(parts) != 2 * n:
            raise ValueError(
                f"Line {i+2} in {path} has {len(parts)} integers, expected {2*n}")
        for j in range(n):
            job_id = int(parts[2 * j])
            p_time = int(parts[2 * j + 1])
            if job_id < 0 or job_id >= n:
                raise ValueError(
                    f"Invalid job id {job_id} on line {i+2} in {path}")
            p_times[i, job_id] = p_time
    # Use file stem as instance name (without extension)
    import os
    name = os.path.splitext(os.path.basename(path))[0]
    return Instance(name=name, p_times=p_times)


def load_best_known(csv_path: str) -> Dict[str, int]:
    """Load best known makespans from a CSV file.

    The CSV is expected to contain at least two columns: ``instance`` and
    ``best_makespan``.  Additional columns are ignored.
    """

    df = pd.read_csv(csv_path)
    if "instance" not in df.columns or "best_makespan" not in df.columns:
        raise ValueError(
            "Best known CSV must contain 'instance' and 'best_makespan' columns"
        )
    mapping = (
        df[["instance", "best_makespan"]]
        .dropna()
        .set_index("instance")["best_makespan"]
        .astype(int)
        .to_dict()
    )
    return mapping


def attach_best_known(instances: Dict[str, Instance], best_known: Mapping[str, int]) -> None:
    """Attach best known makespans to the provided instances in-place."""

    for name, value in best_known.items():
        if name in instances:
            instances[name].best_makespan = int(value)