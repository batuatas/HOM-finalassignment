# mechanisms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Mapping
from .design import get_design, MechanismDesign
from .scheduler import FixedOrderScheduler, QLearningScheduler

@dataclass(frozen=True)
class MechanismSpec:
    key: str
    design: MechanismDesign

_MECHS: Dict[str, MechanismSpec] = {
    "fixed": MechanismSpec(key="fixed", design=get_design("fixed")),
    "adaptive": MechanismSpec(key="adaptive", design=get_design("adaptive")),
}

def get_mechanism(key: str) -> MechanismSpec:
    k = key.lower()
    if k not in _MECHS:
        raise KeyError(f"Unknown mechanism '{key}'. Use one of: {', '.join(sorted(_MECHS))}")
    return _MECHS[k]

def build_scheduler(key: str, op_names: List[str], options: Mapping[str, float] | None = None):
    options = dict(options or {})
    if key == "fixed":
        return FixedOrderScheduler(op_names=list(op_names))
    elif key == "adaptive":
        return QLearningScheduler(
            op_names=list(op_names),
            window_size=int(options.get("window_size", 50)),
            eps=float(options.get("p_min", 0.10)),
            alpha=float(options.get("learning_rate", 0.30)),
            gamma=float(options.get("gamma", 0.60)),
            episode_len=int(options.get("episode_len", 50)),
        )
    else:
        raise KeyError(f"Unknown mechanism '{key}'")
