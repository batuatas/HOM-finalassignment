"""Compatibility layer for legacy imports.

The project has been redesigned to separate operator scheduling mechanisms
into the :mod:`pfsp.mechanisms` package, aligned with the final assignment
rubric.  Import from :mod:`pfsp.mechanisms` directly in new code; the names
below are provided for backwards compatibility with older notebooks.
"""

from __future__ import annotations

from .mechanisms import AdaptiveMechanism as AdaptiveScheduler
from .mechanisms import FixedMechanism as FixedScheduler

__all__ = ["AdaptiveScheduler", "FixedScheduler"]
