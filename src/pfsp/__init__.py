"""Permutation Flow Shop heuristics package."""

This package contains modules for reading PFSP instances, defining local
search operators, implementing operator scheduling mechanisms and running
Iterated Greedy/Local Search algorithms.
"""

from .design import DESIGNS, describe_design, get_design
from .instance import read_instances, Instance
from .operators import Operators
from .scheduler import FixedScheduler, AdaptiveScheduler
from .mechanisms import MECHANISMS, available_mechanisms, build_scheduler, get_mechanism
from .algo_ig_ils import IteratedGreedyILS
from .runner import run_experiments
from .reporting import add_rpd_column, summarise_by_instance

__all__ = [
    "DESIGNS",
    "Instance",
    "read_instances",
    "Operators",
    "FixedScheduler",
    "AdaptiveScheduler",
    "MECHANISMS",
    "available_mechanisms",
    "get_mechanism",
    "get_design",
    "describe_design",
    "build_scheduler",
    "IteratedGreedyILS",
    "run_experiments",
    "add_rpd_column",
    "summarise_by_instance",
]
