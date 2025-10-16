"""Operator scheduling mechanisms for the PFSP metaheuristic.

This package provides concrete mechanism implementations (in this
subpackage) but the repository also contains a top-level module
``pfsp/mechanisms.py`` which defines a registry (``MECHANISMS``) and
factory helpers used by the experiment scripts.  Python's import system
would normally choose this package when importing ``pfsp.mechanisms``,
causing a name collision with the module file.  To preserve the public API
expected by the rest of the codebase we dynamically load the registry
module from the file and re-export its symbols here.
"""

from .adaptive import AdaptiveMechanism
from .base import Mechanism
from .factory import build_mechanism
from .fixed import FixedMechanism

import importlib.machinery
import importlib.util
import os

# Load the sibling module file `mechanisms.py` (the registry) under a
# private name to avoid clashing with this package. Use SourceFileLoader so
# that the loaded module has a proper __spec__ and __package__ set which
# dataclasses rely on when they are created.
_registry_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mechanisms.py"))
loader = importlib.machinery.SourceFileLoader("pfsp._mechanisms_registry", _registry_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
_registry = importlib.util.module_from_spec(spec)
loader.exec_module(_registry)

# Re-export registry symbols expected by the rest of the codebase
MECHANISMS = getattr(_registry, "MECHANISMS")
available_mechanisms = getattr(_registry, "available_mechanisms")
get_mechanism = getattr(_registry, "get_mechanism")
build_scheduler = getattr(_registry, "build_scheduler")
normalise_mechanism_options = getattr(_registry, "normalise_mechanism_options", None)

__all__ = [
    "AdaptiveMechanism",
    "FixedMechanism",
    "Mechanism",
    "build_mechanism",
    "MECHANISMS",
    "available_mechanisms",
    "get_mechanism",
    "build_scheduler",
    "normalise_mechanism_options",
]
