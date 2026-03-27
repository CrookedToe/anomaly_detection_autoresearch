"""Adapters load lazily so metrics-only use does not require docker, durations, or numpyencoder."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "JarAdapter",
    "DistributedAdapter",
    "MultivarAdapter",
    "DockerAdapter",
    "FunctionAdapter",
]

_LAZY = {
    "JarAdapter": ".jar",
    "DistributedAdapter": ".distributed",
    "MultivarAdapter": ".multivar",
    "DockerAdapter": ".docker",
    "FunctionAdapter": ".function",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        submodule = importlib.import_module(_LAZY[name], __name__)
        return getattr(submodule, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
