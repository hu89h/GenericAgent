"""Shared synchronous DashScope/OpenAI-compatible HTTP helpers."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


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
) -> Dict[str, Any]:
    base = (base or base_url()).rstrip("/")
    key = key if key is not None else api_key()
    if require_key and not key:
        raise RuntimeError("API key 未设置")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"accept": "application/json", "content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    req = urllib.request.Request(endpoint_url(base, path), data=data, headers=headers, method="POST")
    last_error = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise RuntimeError(f"{error_prefix} 返回格式异常: {body}")
            return body
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            last_error = RuntimeError(f"{error_prefix} HTTP {exc.code}: {detail}")
            if not _retryable_status(exc.code):
                raise last_error from exc
        except Exception as exc:
            last_error = exc
        if attempt < max(1, int(retries)):
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{error_prefix} 失败: {last_error}")


def embeddings(
    *,
    model: str,
    inputs: list[str],
    base: Optional[str] = None,
    key: Optional[str] = None,
    timeout: int = 60,
    retries: int = 3,
) -> Dict[str, Any]:
    return post_json(
        "/embeddings",
        {"model": model, "input": inputs},
        base=base,
        key=key,
        timeout=timeout,
        retries=retries,
        error_prefix="embedding endpoint",
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
    )


