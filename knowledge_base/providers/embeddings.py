"""Embedding provider for GenericAgent's Zvec knowledge-base index.

This module provides the useful part of a DashScope/OpenAI-compatible embedding
client:
DashScope/OpenAI-compatible ``/embeddings`` requests, basic token trimming, and
batching.  It is synchronous because GA's kb indexer is synchronous.
"""
from __future__ import annotations

import os
import json
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List

from . import provider_http, provider_settings, rate_limit


# Real token usage reported by the embedding endpoint, accumulated per
# ``output_type``.  Batches may run concurrently (see ``_run_batches``), so the
# accumulator is guarded by a lock.  The KB indexer drains it around each flush
# to record true ``api_tokens``; cache hits never call the API and so add
# nothing here.
_usage_lock = threading.Lock()
_usage_acc: Dict[str, int] = {"dense": 0, "sparse": 0}


def _add_api_tokens(output_type: str, body: Dict) -> None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return
    tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens")
                 or usage.get("input_tokens") or 0)
    if tokens <= 0:
        return
    with _usage_lock:
        _usage_acc[output_type] = _usage_acc.get(output_type, 0) + tokens


def drain_usage() -> Dict[str, int]:
    """Return accumulated embedding token usage and reset the accumulator."""
    with _usage_lock:
        drained = dict(_usage_acc)
        for key in _usage_acc:
            _usage_acc[key] = 0
    return drained


def _runtime_config() -> dict:
    """Load one provider snapshot for the current embedding operation.

    Settings can be edited while the Bridge stays alive.  A snapshot keeps one
    batch internally consistent while allowing the next operation to use the
    new configuration without reloading this module or racing mutable globals.
    """
    raw = provider_settings.embedding_config()
    return {
        "base_url": str(raw.get("apibase") or raw.get("base_url") or "").rstrip("/"),
        "api_key": str(raw.get("apikey") or raw.get("api_key") or ""),
        "model": str(raw.get("model") or ""),
        "dimension": int(raw.get("dimension") or 1024),
        "max_tokens": int(raw.get("max_tokens") or 8192),
        "timeout": int(raw.get("timeout") or 60),
        "retries": int(raw.get("max_retries") or 4),
        "batch_size": int(raw.get("batch_size") or 10),
        "concurrency": max(1, int(raw.get("concurrency") or 10)),
        "request_interval": max(0.0, float(raw.get("request_interval") or 0)),
        "rpm_limit": max(1, int(os.environ.get("GA_KB_EMBED_RPM", "1800"))),
        "tpm_limit": max(1, int(os.environ.get("GA_KB_EMBED_TPM", "1200000"))),
        "rate_headroom": min(
            0.95,
            max(0.1, float(os.environ.get("GA_KB_EMBED_RATE_HEADROOM", "0.8"))),
        ),
        "cache_enabled": os.environ.get("GA_KB_EMBED_CACHE", "1").strip().lower()
        not in ("0", "false", "no", "off"),
    }


def _cache_dir() -> str:
    return provider_settings.embedding_cache_dir()


def _rough_token_count(text: str) -> int:
    # RAGAPI uses tiktoken.  Avoid adding a hard dependency here; this estimate
    # is conservative for Chinese-heavy text and good enough for trimming.
    return max(1, len(text or "") // 2)


def _rate_limit_token_estimate(texts: Iterable[str]) -> int:
    """Conservative input-token estimate for mixed Chinese/Latin embedding text."""

    total = 0.0
    for value in texts:
        text = str(value or "")
        cjk = sum(
            "\u3400" <= char <= "\u9fff"
            or "\uf900" <= char <= "\ufaff"
            for char in text
        )
        other = max(0, len(text) - cjk)
        total += cjk + other / 4.0 + 8
    return max(1, int(total * 1.15 + 0.999))


def _usage_input_tokens(body: dict) -> int | None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    return int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("total_tokens")
        or 0
    ) or None


def _embedding_limiter(config: dict):
    return rate_limit.get_limiter(
        "kb_embedding",
        rpm=config["rpm_limit"],
        tpm=config["tpm_limit"],
        headroom=config["rate_headroom"],
    )


def _request_token_budget(config: dict) -> int:
    return max(
        1,
        int(
            int(config["tpm_limit"])
            * float(config["rate_headroom"])
            / 60.0
        ),
    )


def _trim_text(text: str, max_tokens: int | None = None) -> str:
    if max_tokens is None:
        max_tokens = _runtime_config()["max_tokens"]
    text = str(text or "")
    while text and _rough_token_count(text) > max_tokens:
        ratio = max_tokens / float(_rough_token_count(text))
        text = text[:max(1, int(len(text) * ratio) - 50)]
    return text


def _prepared_inputs(batch: List[str], config: dict) -> List[str]:
    inputs = [_trim_text(t, config["max_tokens"]) for t in batch]
    budget = _request_token_budget(config)
    estimate = _rate_limit_token_estimate(inputs)
    if estimate <= budget:
        return inputs
    if len(inputs) != 1:
        raise ValueError(
            f"embedding batch token estimate {estimate} exceeds TPS budget {budget}"
        )
    text = inputs[0]
    while text and _rate_limit_token_estimate([text]) > budget:
        estimate = _rate_limit_token_estimate([text])
        ratio = budget / float(estimate)
        text = text[: max(1, int(len(text) * ratio) - 16)]
    return [text]


def _post_embeddings(batch: List[str], config: dict) -> List[List[float]]:
    if not (config["api_key"] and config["base_url"] and config["model"]):
        raise RuntimeError("mykey.py 需要配置 kb_embedding_config.apikey/apibase/model")
    inputs = _prepared_inputs(batch, config)
    body = provider_http.embeddings(
        model=config["model"],
        inputs=inputs,
        base=config["base_url"],
        key=config["api_key"],
        timeout=config["timeout"],
        retries=config["retries"],
        rate_limiter=_embedding_limiter(config),
        estimated_tokens=_rate_limit_token_estimate(inputs),
        usage_tokens=_usage_input_tokens,
    )
    _add_api_tokens("dense", body)
    rows = sorted(body.get("data") or [], key=lambda x: x.get("index", 0))
    vectors = [list(map(float, r["embedding"])) for r in rows]
    if len(vectors) != len(batch):
        raise RuntimeError(f"embedding 返回数量不匹配：请求 {len(batch)}，返回 {len(vectors)}")
    for vec in vectors:
        if len(vec) != config["dimension"]:
            raise RuntimeError(f"embedding 维度不匹配：期望 {config['dimension']}，返回 {len(vec)}")
    return vectors


def embed_texts(texts: Iterable[str], batch_size: int | None = None) -> List[List[float]]:
    config = _runtime_config()
    texts = list(texts)
    return _embed_cached(
        texts,
        batch_size=config["batch_size"] if batch_size is None else batch_size,
        output_type="dense",
        text_type="document",
        request_fn=lambda batch: _post_embeddings(batch, config),
        normalize_fn=lambda x: list(map(float, x)),
        config=config,
    )


def _post_sparse_embeddings(batch: List[str], text_type: str, config: dict) -> List[Dict[int, float]]:
    if not (config["api_key"] and config["base_url"] and config["model"]):
        raise RuntimeError("mykey.py 需要配置 kb_embedding_config.apikey/apibase/model")
    inputs = _prepared_inputs(batch, config)
    body = provider_http.post_json(
        "/embeddings",
        {
            "model": config["model"],
            "input": inputs,
            "dimension": config["dimension"],
            "output_type": "sparse",
            "text_type": text_type,
        },
        base=config["base_url"],
        key=config["api_key"],
        timeout=config["timeout"],
        retries=config["retries"],
        error_prefix="sparse embedding endpoint",
        rate_limiter=_embedding_limiter(config),
        estimated_tokens=_rate_limit_token_estimate(inputs),
        usage_tokens=_usage_input_tokens,
    )
    _add_api_tokens("sparse", body)
    rows = sorted(body.get("data") or [], key=lambda x: x.get("index", 0))
    vectors: List[Dict[int, float]] = []
    for row in rows:
        items = row.get("embedding") or row.get("sparse_embedding") or []
        sparse: Dict[int, float] = {}
        for item in items:
            try:
                idx = int(item.get("index"))
                val = float(item.get("value"))
            except Exception:
                continue
            if val > 0:
                sparse[idx] = val
        vectors.append(dict(sorted(sparse.items())))
    if len(vectors) != len(batch):
        raise RuntimeError(f"sparse embedding 返回数量不匹配：请求 {len(batch)}，返回 {len(vectors)}")
    return vectors


def embed_sparse_texts(
    texts: Iterable[str],
    *,
    text_type: str = "document",
    batch_size: int | None = None,
) -> List[Dict[int, float]]:
    config = _runtime_config()
    texts = list(texts)
    text_type = "query" if text_type == "query" else "document"
    return _embed_cached(
        texts,
        batch_size=config["batch_size"] if batch_size is None else batch_size,
        output_type="sparse",
        text_type=text_type,
        request_fn=lambda batch: _post_sparse_embeddings(batch, text_type, config),
        normalize_fn=_normalize_sparse_cached,
        config=config,
    )


def _embed_cached(texts, *, batch_size, output_type, text_type, request_fn, normalize_fn, config):
    if not config["cache_enabled"]:
        out = []
        for _, vectors in _run_batches(
            _make_batches(texts, batch_size, config=config),
            request_fn,
            concurrency=config["concurrency"],
            request_interval=config["request_interval"],
        ):
            out.extend(vectors)
        return out
    os.makedirs(_cache_dir(), exist_ok=True)
    out = [None] * len(texts)
    missing = []
    for i, text in enumerate(texts):
        key = _cache_key(text, output_type=output_type, text_type=text_type, config=config)
        cached = _cache_get(key)
        if cached is None:
            missing.append((i, text, key))
        else:
            out[i] = normalize_fn(cached)
    if missing:
        size = max(1, int(batch_size))
        missing_batches = _partition_batches(
            missing,
            size,
            config=config,
            text_getter=lambda row: row[1],
        )
        for _, rows in _run_batches(
            [(i, [text for _idx, text, _key in rows]) for i, rows in missing_batches],
            request_fn,
            concurrency=config["concurrency"],
            request_interval=config["request_interval"],
        ):
            batch_rows = dict(missing_batches)[_]
            for (idx, _text, key), vector in zip(batch_rows, rows):
                normalized = normalize_fn(vector)
                out[idx] = normalized
                _cache_put(key, normalized)
    return out


def _cache_key(text: str, *, output_type: str, text_type: str, config: dict) -> str:
    payload = {
        "base_url": config["base_url"],
        "model": config["model"],
        "dimension": config["dimension"],
        "output_type": output_type,
        "text_type": text_type,
        "text": _trim_text(text, config["max_tokens"]),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(_cache_dir(), key[:2], f"{key}.json")


def _cache_get(key: str):
    try:
        with open(_cache_path(key), encoding="utf-8") as f:
            return json.load(f).get("embedding")
    except Exception:
        return None


def _cache_put(key: str, embedding) -> None:
    path = _cache_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding}, f, ensure_ascii=False)
    os.replace(tmp, path)


def _normalize_sparse_cached(value) -> Dict[int, float]:
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items() if float(v) > 0}
    return {}


def _partition_batches(items, batch_size: int, *, config: dict, text_getter):
    size = max(1, int(batch_size))
    budget = _request_token_budget(config)
    batches = []
    current = []
    current_tokens = 0
    start = 0
    for position, item in enumerate(items):
        text = _trim_text(text_getter(item), config["max_tokens"])
        item_tokens = _rate_limit_token_estimate([text])
        if current and (
            len(current) >= size or current_tokens + item_tokens > budget
        ):
            batches.append((start, current))
            current = []
            current_tokens = 0
            start = position
        if not current:
            start = position
        current.append(item)
        current_tokens += min(item_tokens, budget)
    if current:
        batches.append((start, current))
    return batches


def _make_batches(
    texts: List[str],
    batch_size: int,
    *,
    config: dict | None = None,
) -> List[tuple[int, List[str]]]:
    if config is None:
        size = max(1, int(batch_size))
        return [(i, texts[i:i + size]) for i in range(0, len(texts), size)]
    return _partition_batches(
        texts,
        batch_size,
        config=config,
        text_getter=lambda text: text,
    )


def _run_batches(batches, fn, *, concurrency=1, request_interval=0.0):
    if not batches:
        return []
    if concurrency <= 1 or len(batches) == 1:
        results = []
        for pos, (i, batch) in enumerate(batches):
            if pos and request_interval:
                time.sleep(request_interval)
            try:
                vectors = fn(batch)
            except Exception:
                # DashScope occasionally returns a transient 403/429 on an
                # otherwise valid batch. Wait longer than the per-request retry
                # backoff, then retry the same batch once before surfacing it.
                time.sleep(10)
                vectors = fn(batch)
            results.append((i, vectors))
        return results
    results = {}
    failed = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as ex:
        futures = {ex.submit(fn, batch): i for i, batch in batches}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                failed[idx] = exc
    if failed:
        # DashScope may reject a burst even when the same batch succeeds moments
        # later. Keep global concurrency high, but retry only failed batches
        # serially so one transient 403/429 does not abort a full KB rebuild.
        time.sleep(2)
        batch_map = dict(batches)
        for pos, idx in enumerate(sorted(failed)):
            if pos and request_interval:
                time.sleep(request_interval)
            results[idx] = fn(batch_map[idx])
    return [(i, results[i]) for i, _batch in sorted(batches, key=lambda x: x[0])]


def embedding_meta() -> dict:
    config = _runtime_config()
    return {
        "provider": "dashscope",
        "base_url": config["base_url"],
        "model": config["model"],
        "dimension": config["dimension"],
        "batch_size": config["batch_size"],
        "concurrency": config["concurrency"],
        "rpm_limit": config["rpm_limit"],
        "tpm_limit": config["tpm_limit"],
        "rate_headroom": config["rate_headroom"],
    }


def sparse_embedding_meta() -> dict:
    config = _runtime_config()
    return {
        "provider": "dashscope",
        "base_url": config["base_url"],
        "model": config["model"],
        "dimension": config["dimension"],
        "output_type": "sparse",
        "batch_size": config["batch_size"],
        "concurrency": config["concurrency"],
        "rpm_limit": config["rpm_limit"],
        "tpm_limit": config["tpm_limit"],
        "rate_headroom": config["rate_headroom"],
    }
