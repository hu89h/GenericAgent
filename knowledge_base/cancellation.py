"""Cooperative cancellation primitives for knowledge-base mutations."""
from __future__ import annotations

import time
from typing import Callable


class KnowledgeBaseCancelled(RuntimeError):
    """Raised when the user cancels an import or reindex operation."""

    code = "kb_operation_cancelled"

    def __init__(self) -> None:
        super().__init__(self.code)


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if callable(cancelled) and cancelled():
        raise KnowledgeBaseCancelled()


def wait_with_cancellation(
    seconds: float,
    cancelled: Callable[[], bool] | None,
    *,
    interval: float = 0.1,
) -> None:
    """Wait for a retry/poll delay while remaining responsive to cancellation."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        check_cancelled(cancelled)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(max(0.01, float(interval)), remaining))


__all__ = [
    "KnowledgeBaseCancelled",
    "check_cancelled",
    "wait_with_cancellation",
]
