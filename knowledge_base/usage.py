"""Build usage accounting for the knowledge-base indexer.

This module deliberately does not import ``backend``.  Storage paths are
supplied by :class:`UsageTracker` callbacks so accounting can evolve
independently from document extraction and Zvec operations.

Only compact per-build counters are tracked here.  The fine-grained cost
estimation and per-model token breakdown that used to live in this module
were removed: their sole consumer was ``kb --status`` printing the raw JSON,
while the Desktop bridge and frontend never read the cost/model fields.

Token counts are the *real* values reported by the provider APIs — image
prompt/completion tokens come from the VLM ``usage`` block, and embedding
``api_tokens`` come from the embedding endpoint ``usage``.  They therefore
only cover text that actually hit the API this build; cache hits contribute
no tokens (because no call was made).  Cost in currency is intentionally not
computed here: the APIs never return an amount, and any local estimate would
depend on hand-maintained unit prices that drift when the provider repriced.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict


class UsageTracker:
    """Track per-thread build usage counters and persist a compact report."""

    def __init__(
        self,
        *,
        index_dir_fn: Callable[[str], str],
        usage_path_fn: Callable[[str], str],
    ) -> None:
        self._index_dir_fn = index_dir_fn
        self._usage_path_fn = usage_path_fn
        self._local = threading.local()
        self._lock = threading.Lock()

    def empty(self) -> Dict[str, Any]:
        return {
            "image_analysis": {
                "calls": 0, "cached": 0, "failed": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
            },
            "embedding": {"calls": 0, "texts": 0, "api_tokens": 0},
            "sparse_embedding": {"calls": 0, "texts": 0, "api_tokens": 0},
        }

    def current(self) -> Dict[str, Any]:
        value = getattr(self._local, "current", None)
        if value is None:
            value = self.empty()
            self._local.current = value
        return value

    def set_current(self, value: Dict[str, Any]) -> None:
        self._local.current = value

    def merge_image_analysis(self, usage_delta: Dict[str, Any] | None) -> None:
        if not usage_delta:
            return
        with self._lock:
            destination = self.current()["image_analysis"]
            for key in ("calls", "cached", "failed", "prompt_tokens", "completion_tokens"):
                destination[key] += int(usage_delta.get(key) or 0)

    def write(self, kb_path: str, usage: Dict[str, Any] | None) -> None:
        os.makedirs(self._index_dir_fn(kb_path), exist_ok=True)
        value = dict(usage or {})
        with open(self._usage_path_fn(kb_path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    def load(self, kb_path: str) -> Dict[str, Any]:
        try:
            with open(self._usage_path_fn(kb_path), encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
