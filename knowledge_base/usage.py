"""Compact provider-reported usage counters for one complete index build."""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict


class UsageTracker:
    """Track one mutation-locked build and persist its compact usage report."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current = self.empty()

    @staticmethod
    def index_dir(kb_path: str) -> str:
        return os.path.join(kb_path, ".kb_index")

    @classmethod
    def usage_path(cls, kb_path: str) -> str:
        return os.path.join(cls.index_dir(kb_path), "build_usage.json")

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
        return self._current

    def set_current(self, value: Dict[str, Any]) -> None:
        with self._lock:
            self._current = value

    def merge_image_analysis(self, usage_delta: Dict[str, Any] | None) -> None:
        if not usage_delta:
            return
        with self._lock:
            destination = self.current()["image_analysis"]
            for key in ("calls", "cached", "failed", "prompt_tokens", "completion_tokens"):
                destination[key] += int(usage_delta.get(key) or 0)

    def write(self, kb_path: str, usage: Dict[str, Any] | None) -> None:
        os.makedirs(self.index_dir(kb_path), exist_ok=True)
        value = dict(usage or {})
        with open(self.usage_path(kb_path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    def load(self, kb_path: str) -> Dict[str, Any]:
        try:
            with open(self.usage_path(kb_path), encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
