"""Shared synchronous DashScope/OpenAI-compatible HTTP helpers."""
from __future__ import annotations

import json
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Optional

from ..cancellation import KnowledgeBaseCancelled, check_cancelled, wait_with_cancellation


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def base_url(*names: str, default: str = DEFAULT_BASE_URL) -> str:
    for name in names:
        value = os.environ.get(str(name), "").strip()
        if value:
            return value.rstrip("/")
    return str(default or DEFAULT_BASE_URL).rstrip("/")


def api_key(*names: str) -> str:
    for name in names:
        value = os.environ.get(str(name), "").strip()
        if value:
            return value
    return ""


def endpoint_url(base: str, path: str) -> str:
    base = (base or DEFAULT_BASE_URL).rstrip("/")
    clean_path = "/" + str(path or "").strip("/")
    if base.endswith("$"):
        return base[:-1].rstrip("/")
    if base.endswith(clean_path):
        return base
    return base + clean_path


def _retryable_status(status: int) -> bool:
    return int(status) in (408, 409, 425, 429) or int(status) >= 500


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    raw = str((error.headers or {}).get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    exponential = min(30.0, float(2 ** max(0, int(attempt) - 1)))
    base = max(exponential, float(retry_after or 0))
    return base + random.uniform(0, min(1.0, base * 0.2))


def _request_once(
    request: urllib.request.Request,
    *,
    timeout: int,
    cancelled: Callable[[], bool] | None = None,
):
    """Run one HTTP request, allowing the caller to stop waiting for it.

    ``urllib`` cannot interrupt a blocking ``urlopen`` from another thread.
    When cancellation is supplied, run that one network operation in a daemon
    helper and poll it from the task thread.  Cancellation therefore stops
    retries and releases the import worker immediately; the in-flight socket
    is bounded by the normal request timeout and cannot keep the process alive.
    """

    def operation():
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    if not callable(cancelled):
        return operation()

    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result.put((True, operation()))
        except BaseException as error:  # propagate HTTPError and cancellation-safe errors
            result.put((False, error))

    thread = threading.Thread(
        target=worker,
        name="ga-kb-http",
        daemon=True,
    )
    thread.start()
    while True:
        check_cancelled(cancelled)
        try:
            ok, value = result.get(timeout=0.2)
        except queue.Empty:
            continue
        if ok:
            return value
        raise value


def post_json(
    path: str,
    payload: Dict[str, Any],
    *,
    base: Optional[str] = None,
    key: Optional[str] = None,
    timeout: int = 60,
    retries: int = 3,
    error_prefix: str = "dashscope request",
    require_key: bool = True,
    rate_limiter=None,
    estimated_tokens: int = 1,
    usage_tokens=None,
    auth_mode: str = "bearer",
    extra_headers: Optional[Dict[str, str]] = None,
    cancelled: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    base = (base or base_url()).rstrip("/")
    key = key if key is not None else api_key()
    if require_key and not key:
        raise RuntimeError("API key 未设置")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"accept": "application/json", "content-type": "application/json"}
    if key:
        if str(auth_mode or "bearer").strip().lower() in {"x-api-key", "x_api_key", "anthropic"}:
            headers["x-api-key"] = key
        else:
            headers["authorization"] = f"Bearer {key}"
    for name, value in (extra_headers or {}).items():
        if value is not None and str(value) != "":
            headers[str(name)] = str(value)
    req = urllib.request.Request(endpoint_url(base, path), data=data, headers=headers, method="POST")
    last_error = None
    retry_after = None
    for attempt in range(1, max(1, int(retries)) + 1):
        check_cancelled(cancelled)
        if rate_limiter is None:
            reservation = None
        elif callable(cancelled):
            reservation = rate_limiter.acquire(
                estimated_tokens,
                cancelled=cancelled,
            )
        else:
            # Keep third-party/test limiter implementations that only expose
            # the original one-argument acquire() contract working.
            reservation = rate_limiter.acquire(estimated_tokens)
        try:
            body = _request_once(req, timeout=timeout, cancelled=cancelled)
            if not isinstance(body, dict):
                raise RuntimeError(f"{error_prefix} 返回格式异常: {body}")
            if reservation is not None and callable(usage_tokens):
                try:
                    rate_limiter.reconcile(reservation, usage_tokens(body))
                except Exception:
                    pass
            return body
        except KnowledgeBaseCancelled:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            last_error = RuntimeError(f"{error_prefix} HTTP {exc.code}: {detail}")
            if not _retryable_status(exc.code):
                raise last_error from exc
            retry_after = _retry_after_seconds(exc)
        except Exception as exc:
            last_error = exc
            retry_after = None
        if attempt < max(1, int(retries)):
            delay = _retry_delay(attempt, retry_after)
            if callable(cancelled):
                wait_with_cancellation(delay, cancelled)
            else:
                # Preserve the original synchronous path and its injectable
                # sleep behavior for callers that do not need cancellation.
                time.sleep(delay)
    raise RuntimeError(f"{error_prefix} 失败: {last_error}")


def embeddings(
    *,
    model: str,
    inputs: list[str],
    base: Optional[str] = None,
    key: Optional[str] = None,
    timeout: int = 60,
    retries: int = 3,
    rate_limiter=None,
    estimated_tokens: int = 1,
    usage_tokens=None,
    cancelled: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    return post_json(
        "/embeddings",
        {"model": model, "input": inputs},
        base=base,
        key=key,
        timeout=timeout,
        retries=retries,
        error_prefix="embedding endpoint",
        rate_limiter=rate_limiter,
        estimated_tokens=estimated_tokens,
        usage_tokens=usage_tokens,
        cancelled=cancelled,
    )


def chat_completions(
    *,
    model: str,
    messages: list,
    base: Optional[str] = None,
    key: Optional[str] = None,
    timeout: int = 120,
    retries: int = 2,
    extra: Optional[Dict[str, Any]] = None,
    rate_limiter=None,
    estimated_tokens: int = 1,
    usage_tokens=None,
    cancelled: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if extra:
        payload.update(extra)
    return post_json(
        "/chat/completions",
        payload,
        base=base,
        key=key,
        timeout=timeout,
        retries=retries,
        error_prefix="chat completion endpoint",
        rate_limiter=rate_limiter,
        estimated_tokens=estimated_tokens,
        usage_tokens=usage_tokens,
        cancelled=cancelled,
    )


def anthropic_messages(
    *,
    model: str,
    messages: list,
    base: Optional[str] = None,
    key: Optional[str] = None,
    timeout: int = 120,
    retries: int = 2,
    max_tokens: int = 8192,
    extra: Optional[Dict[str, Any]] = None,
    auth_mode: str = "bearer",
    rate_limiter=None,
    estimated_tokens: int = 1,
    usage_tokens=None,
    cancelled: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
    }
    if extra:
        payload.update(extra)
    return post_json(
        "/messages",
        payload,
        base=base,
        key=key,
        timeout=timeout,
        retries=retries,
        error_prefix="anthropic messages endpoint",
        auth_mode=auth_mode,
        extra_headers={"anthropic-version": "2023-06-01"},
        rate_limiter=rate_limiter,
        estimated_tokens=estimated_tokens,
        usage_tokens=usage_tokens,
        cancelled=cancelled,
    )


