"""Build usage accounting and cost estimation for the knowledge-base indexer.

This module deliberately does not import ``backend``.  Runtime metadata and
storage paths are supplied by :class:`UsageTracker` callbacks so accounting
can evolve independently from document extraction and Zvec operations.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict


class UsageTracker:
    """Track per-thread build usage and persist its final report."""

    def __init__(
        self,
        *,
        image_meta_fn: Callable[[], Dict[str, Any]],
        embedding_meta_fn: Callable[[], Dict[str, Any]],
        sparse_embedding_meta_fn: Callable[[], Dict[str, Any]],
        embedding_provider_fn: Callable[[], str],
        index_dir_fn: Callable[[str], str],
        usage_path_fn: Callable[[str], str],
    ) -> None:
        self._image_meta_fn = image_meta_fn
        self._embedding_meta_fn = embedding_meta_fn
        self._sparse_embedding_meta_fn = sparse_embedding_meta_fn
        self._embedding_provider_fn = embedding_provider_fn
        self._index_dir_fn = index_dir_fn
        self._usage_path_fn = usage_path_fn
        self._local = threading.local()
        self._lock = threading.Lock()

        # These are environment-configured because provider pricing can vary
        # by region and change independently from the index format.
        self.currency = os.environ.get("GA_KB_COST_CURRENCY", "CNY")
        self.image_input_price = float(
            os.environ.get("GA_KB_IMAGE_INPUT_PRICE_PER_MTOKENS", "0.3")
        )
        self.image_output_price = float(
            os.environ.get("GA_KB_IMAGE_OUTPUT_PRICE_PER_MTOKENS", "0.5")
        )
        self.embedding_input_price = float(
            os.environ.get("GA_KB_EMBED_INPUT_PRICE_PER_KTOKENS", "0.0005")
        )

    def empty(self, kb_id: str = "", kb_path: str = "") -> Dict[str, Any]:
        try:
            image_meta = self._image_meta_fn()
        except Exception:
            image_meta = {
                "enabled": os.environ.get("GA_KB_IMAGE_ANALYSIS", "0")
                .strip()
                .lower()
                in ("1", "true", "yes", "on")
            }
        try:
            embedding_meta = self._embedding_meta_fn()
        except Exception:
            embedding_meta = {"provider": self._embedding_provider_fn(), "model": ""}
        try:
            sparse_meta = self._sparse_embedding_meta_fn()
        except Exception:
            sparse_meta = {"provider": self._embedding_provider_fn(), "model": ""}
        return {
            "schema_version": 2,
            "kb_id": kb_id,
            "kb_path": kb_path,
            "started_at": int(time.time()),
            "finished_at": None,
            "image_analysis": {
                "enabled": bool(image_meta.get("enabled")),
                "meta": image_meta,
                "calls": 0,
                "cached": 0,
                "failed": 0,
                "input_images": 0,
                "input_image_bytes": 0,
                "input_text_chars": 0,
                "output_chars": 0,
                "models": {},
                "cached_models": {},
            },
            "embedding": {
                "provider": embedding_meta.get("provider") or self._embedding_provider_fn(),
                "model": embedding_meta.get("model") or "",
                "meta": embedding_meta,
                "calls": 0,
                "texts": 0,
                "input_chars": 0,
                "estimated_input_tokens": 0,
                "failed": 0,
            },
            "sparse_embedding": {
                "provider": sparse_meta.get("provider") or self._embedding_provider_fn(),
                "model": sparse_meta.get("model") or "",
                "meta": sparse_meta,
                "calls": 0,
                "texts": 0,
                "input_chars": 0,
                "estimated_input_tokens": 0,
                "failed": 0,
            },
        }

    def current(self) -> Dict[str, Any]:
        value = getattr(self._local, "current", None)
        if value is None:
            value = self.empty()
            self._local.current = value
        return value

    def set_current(self, value: Dict[str, Any]) -> None:
        self._local.current = value

    @staticmethod
    def _empty_model_usage_row() -> Dict[str, int]:
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "image_tokens": 0,
            "text_tokens": 0,
            "output_chars": 0,
        }

    def _merge_model_usage_rows(self, dst_models: Dict[str, Any], src_models: Dict[str, Any] | None) -> None:
        for model, row in (src_models or {}).items():
            bucket = dst_models.setdefault(model or "unknown", self._empty_model_usage_row())
            for key in (
                "calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "image_tokens",
                "text_tokens",
                "output_chars",
            ):
                bucket[key] += int(row.get(key) or 0)

    def model_usage_delta(
        self, model: str, usage: Dict[str, Any] | None, output_chars: int = 0
    ) -> Dict[str, Dict[str, int]]:
        row = self._empty_model_usage_row()
        row["calls"] = 1
        row["output_chars"] = int(output_chars or 0)
        if isinstance(usage, dict):
            row["prompt_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            row["completion_tokens"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            row["total_tokens"] += int(usage.get("total_tokens") or 0)
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                row["image_tokens"] += int(prompt_details.get("image_tokens") or 0)
                row["text_tokens"] += int(prompt_details.get("text_tokens") or 0)
        return {model or "unknown": row}

    def add_model_usage(self, model: str, usage: Dict[str, Any] | None, output_chars: int = 0) -> None:
        bucket = self.current()["image_analysis"]["models"].setdefault(
            model or "unknown", self._empty_model_usage_row()
        )
        bucket["calls"] += 1
        bucket["output_chars"] += int(output_chars or 0)
        if isinstance(usage, dict):
            bucket["prompt_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            bucket["completion_tokens"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            bucket["total_tokens"] += int(usage.get("total_tokens") or 0)
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                bucket["image_tokens"] += int(prompt_details.get("image_tokens") or 0)
                bucket["text_tokens"] += int(prompt_details.get("text_tokens") or 0)

    def merge_image_analysis(self, usage_delta: Dict[str, Any] | None) -> None:
        if not usage_delta:
            return
        with self._lock:
            destination = self.current()["image_analysis"]
            for key in (
                "calls",
                "cached",
                "failed",
                "input_images",
                "input_image_bytes",
                "input_text_chars",
                "output_chars",
            ):
                destination[key] += int(usage_delta.get(key) or 0)
            self._merge_model_usage_rows(destination["models"], usage_delta.get("models"))
            self._merge_model_usage_rows(
                destination.setdefault("cached_models", {}), usage_delta.get("cached_models")
            )

    def write(self, kb_path: str, usage: Dict[str, Any] | None) -> None:
        os.makedirs(self._index_dir_fn(kb_path), exist_ok=True)
        value = dict(usage or {})
        value["finished_at"] = int(time.time())
        value["pricing"] = {
            "currency": self.currency,
            "image_input_per_million_tokens": self.image_input_price,
            "image_output_per_million_tokens": self.image_output_price,
            "embedding_input_per_thousand_tokens": self.embedding_input_price,
            "source": "configurable defaults; billing is subject to provider console",
        }
        value["cost"] = self.calculate_cost(value)
        with open(self._usage_path_fn(kb_path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    def load(self, kb_path: str) -> Dict[str, Any]:
        try:
            with open(self._usage_path_fn(kb_path), encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _image_model_cost(self, models: Dict[str, Any] | None):
        image_cost = 0.0
        image_prompt = image_completion = image_calls = 0
        for row in (models or {}).values():
            prompt = int(row.get("prompt_tokens") or 0)
            completion = int(row.get("completion_tokens") or 0)
            image_calls += int(row.get("calls") or 0)
            image_prompt += prompt
            image_completion += completion
            image_cost += prompt / 1_000_000.0 * self.image_input_price
            image_cost += completion / 1_000_000.0 * self.image_output_price
        return image_cost, image_prompt, image_completion, image_calls

    def calculate_cost(self, value: Dict[str, Any]) -> Dict[str, Any]:
        image_analysis = value.get("image_analysis") or {}
        image_cost, image_prompt, image_completion, image_calls = self._image_model_cost(
            image_analysis.get("models") or {}
        )
        cached_cost, cached_prompt, cached_completion, cached_calls = self._image_model_cost(
            image_analysis.get("cached_models") or {}
        )
        embedding = value.get("embedding") or {}
        embedding_tokens = int(embedding.get("estimated_input_tokens") or 0)
        sparse = value.get("sparse_embedding") or {}
        sparse_tokens = int(sparse.get("estimated_input_tokens") or 0)
        embedding_cost = (embedding_tokens + sparse_tokens) / 1000.0 * self.embedding_input_price
        lifetime_image_cost = image_cost + cached_cost
        return {
            "currency": self.currency,
            "image_actual": {
                "amount": round(image_cost, 8),
                "calls": image_calls,
                "prompt_tokens": image_prompt,
                "completion_tokens": image_completion,
                "note": "本次构建新增发生的图片模型调用费用",
            },
            "image_cached_historical": {
                "amount": round(cached_cost, 8),
                "calls": cached_calls,
                "prompt_tokens": cached_prompt,
                "completion_tokens": cached_completion,
                "note": "本次构建复用缓存所对应的历史图片模型调用费用",
            },
            "image_lifetime_actual": {
                "amount": round(lifetime_image_cost, 8),
                "calls": image_calls + cached_calls,
                "prompt_tokens": image_prompt + cached_prompt,
                "completion_tokens": image_completion + cached_completion,
                "note": "当前索引使用到的图片理解结果对应的累计实际调用费用",
            },
            "embedding_estimated": {
                "amount": round(embedding_cost, 8),
                "estimated_input_tokens": embedding_tokens + sparse_tokens,
                "dense_estimated_input_tokens": embedding_tokens,
                "sparse_estimated_input_tokens": sparse_tokens,
                "note": "DashScope embedding 响应未统一返回 usage；按字符粗估 token 后计费",
            },
            "total_estimated": {
                "amount": round(image_cost + embedding_cost, 8),
                "note": "本次构建新增成本估算；缓存命中不作为本次新增费用",
            },
            "total_with_cached_image_estimated": {
                "amount": round(lifetime_image_cost + embedding_cost, 8),
                "note": "当前索引使用到的图片历史实际费用，加上本次 embedding 估算费用",
            },
        }
