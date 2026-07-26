"""Unified local knowledge-base package.

The package owns ingestion, document preparation, indexing, retrieval, and
provider clients.  Desktop and agent integrations share the public backend
facade exposed here.
"""
from importlib import import_module


def __getattr__(name):
    if name == "backend":
        module = import_module(f"{__name__}.backend")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["backend"]
