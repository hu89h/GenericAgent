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
            # Keep the provider model names alongside the counters so the
            # usage page describes the maintenance that actually produced
            # the data, rather than whatever happens to be configured now.
            "models": {
                "image": "",
                "embedding": "",
            },
            "image_analysis": {
                "calls": 0, "cached": 0, "failed": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "token_usage_reported": False,
            },
            "embedding": {
                "calls": 0, "texts": 0, "api_tokens": 0,
                "input_tokens": 0, "output_tokens": 0,
                "cache_hits": 0,
                "api_calls": 0,
                "token_usage_reported": False,
                "input_token_usage_reported": False,
                "output_token_usage_reported": False,
            },
            "sparse_embedding": {
                "calls": 0, "texts": 0, "api_tokens": 0,
                "input_tokens": 0, "output_tokens": 0,
                "cache_hits": 0,
                "api_calls": 0,
                "token_usage_reported": False,
                "input_token_usage_reported": False,
                "output_token_usage_reported": False,
            },
        }

    def current(self) -> Dict[str, Any]:
        return self._current

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._current, ensure_ascii=False))

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
            destination["token_usage_reported"] = bool(
                destination.get("token_usage_reported")
                or usage_delta.get("token_usage_reported")
            )

    @staticmethod
    def summary(usage: Dict[str, Any] | None) -> Dict[str, Any]:
        """Return a stable, display-safe summary of provider usage.

        Token values are meaningful only when the provider returned a usage
        block.  Keep those values as ``None`` when the endpoint omitted the
        block, instead of making an unknown value look like zero consumption.
        """
        raw = usage if isinstance(usage, dict) else {}

        models = raw.get("models") if isinstance(raw.get("models"), dict) else {}

        def section(name: str) -> Dict[str, Any]:
            value = raw.get(name) if isinstance(raw.get(name), dict) else {}
            reported = bool(value.get("token_usage_reported"))
            return value, reported

        image, image_reported = section("image_analysis")
        dense, dense_reported = section("embedding")
        sparse, sparse_reported = section("sparse_embedding")
        dense_api_active = int(dense.get("api_calls") or 0) > 0
        sparse_api_active = int(sparse.get("api_calls") or 0) > 0
        dense_active = any(
            int(dense.get(key) or 0) > 0
            for key in ("calls", "texts", "cache_hits")
        )
        sparse_active = any(
            int(sparse.get(key) or 0) > 0
            for key in ("calls", "texts", "cache_hits")
        )
        embedding_reported = (
            (not dense_active or dense_reported)
            and (not sparse_active or sparse_reported)
        )
        embedding_input_reported = (
            (not dense_api_active or bool(dense.get("input_token_usage_reported")))
            and (not sparse_api_active or bool(sparse.get("input_token_usage_reported")))
        )
        embedding_output_reported = (
            (not dense_api_active or bool(dense.get("output_token_usage_reported")))
            and (not sparse_api_active or bool(sparse.get("output_token_usage_reported")))
        )
        embedding_api_active = dense_api_active or sparse_api_active
        return {
            "available": bool(raw),
            "image_model": str(models.get("image") or ""),
            "embedding_model": str(models.get("embedding") or ""),
            "image_calls": int(image.get("calls") or 0),
            "image_cached": int(image.get("cached") or 0),
            "image_failed": int(image.get("failed") or 0),
            "image_prompt_tokens": int(image.get("prompt_tokens") or 0) if image_reported else None,
            "image_completion_tokens": int(image.get("completion_tokens") or 0) if image_reported else None,
            "image_token_usage_reported": image_reported,
            # Dense and sparse vectors are an implementation detail.  The
            # user configures one embedding service, so expose one combined
            # line while retaining the raw per-field counters on disk.
            "embedding_calls": max(
                int(dense.get("calls") or 0), int(sparse.get("calls") or 0)
            ),
            "embedding_texts": max(
                int(dense.get("texts") or 0), int(sparse.get("texts") or 0)
            ),
            "embedding_cache_hits": max(
                int(dense.get("cache_hits") or 0), int(sparse.get("cache_hits") or 0)
            ),
            "embedding_api_calls": int(dense.get("api_calls") or 0) + int(
                sparse.get("api_calls") or 0
            ),
            "embedding_api_tokens": (
                int(dense.get("api_tokens") or 0)
                + int(sparse.get("api_tokens") or 0)
                if embedding_reported else None
            ),
            "embedding_token_usage_reported": embedding_reported,
            "embedding_input_tokens": (
                int(dense.get("input_tokens") or 0)
                + int(sparse.get("input_tokens") or 0)
                if embedding_api_active and embedding_input_reported else None
            ),
            "embedding_output_tokens": (
                int(dense.get("output_tokens") or 0)
                + int(sparse.get("output_tokens") or 0)
                if embedding_api_active and embedding_output_reported else None
            ),
            "embedding_input_token_usage_reported": bool(
                embedding_api_active and embedding_input_reported
            ),
            "embedding_output_token_usage_reported": bool(
                embedding_api_active and embedding_output_reported
            ),
        }

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
