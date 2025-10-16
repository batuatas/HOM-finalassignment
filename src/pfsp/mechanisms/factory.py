"""Factory helpers for constructing mechanisms from configuration values."""

from __future__ import annotations

from typing import Iterable

from .adaptive import AdaptiveMechanism
from .base import Mechanism
from .fixed import FixedMechanism


def build_mechanism(
    name: str,
    operators: Iterable[str],
    **kwargs,
) -> Mechanism:
    """Instantiate a scheduling mechanism by symbolic name."""

    normalised = name.lower()
    if normalised in {"fixed", "deterministic", "mechanism1a"}:
        if kwargs:
            # Avoid silent acceptance of irrelevant parameters.
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Fixed mechanism does not accept parameters: {unexpected}")
        return FixedMechanism(operators)
    if normalised in {"adaptive", "probability_matching", "mechanism2a"}:
        return AdaptiveMechanism(operators, **kwargs)
    raise ValueError(f"Unknown mechanism '{name}'")


__all__ = ["build_mechanism"]
