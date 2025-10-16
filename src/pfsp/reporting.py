"""Reporting helpers for the PFSP final assignment."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def add_rpd_column(df: pd.DataFrame, best_known: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Return a copy of *df* with a normalised ``rpd`` column.

    Parameters
    ----------
    df:
        DataFrame with at least ``instance`` and ``makespan`` columns.
    best_known:
        Optional mapping from instance name to best known makespan.  When
        provided, the ``best_known`` column will be filled/overwritten with the
        mapped values before computing the RPD.
    """

    if "instance" not in df.columns:
        raise ValueError("Input DataFrame must contain an 'instance' column")
    if "makespan" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'makespan' column")

    result = df.copy()
    if best_known is not None:
        result["best_known"] = result["instance"].map(best_known)
    if "best_known" not in result.columns:
        result["best_known"] = pd.NA
    mask = result["best_known"].notna() & (result["best_known"].astype(float) > 0)
    result.loc[mask, "rpd"] = (
        (result.loc[mask, "makespan"] - result.loc[mask, "best_known"].astype(float))
        / result.loc[mask, "best_known"].astype(float)
        * 100.0
    )
    return result


def summarise_by_instance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by instance and mechanism."""

    required = {"instance", "makespan", "elapsed"}
    if "mechanism_label" not in df.columns and "mechanism_key" not in df.columns:
        required.add("algorithm")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {', '.join(sorted(missing))}")
    if "mechanism_label" in df.columns:
        group_cols = ["mechanism_label", "instance"]
    elif "mechanism_key" in df.columns:
        group_cols = ["mechanism_key", "instance"]
    else:
        group_cols = ["algorithm", "instance"]
    agg_dict: dict[str, object] = {
        "makespan": ["mean", "min", "std"],
        "elapsed": "mean",
    }
    if "iterations" in df.columns:
        agg_dict["iterations"] = "mean"
    if "rpd" in df.columns:
        agg_dict["rpd"] = "mean"
    grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
    # Flatten MultiIndex columns produced by aggregation
    grouped.columns = [
        "_".join(filter(None, map(str, col))).rstrip("_") for col in grouped.columns.values
    ]
    return grouped
