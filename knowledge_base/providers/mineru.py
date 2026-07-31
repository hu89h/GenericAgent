"""Small synchronous client for the MinerU v4 document parsing API.

The client owns only HTTP interaction.  Knowledge-base persistence lives in
:mod:`knowledge_base.importer`; PDF page-limit validation happens before a
remote job is submitted.
"""
from __future__ import annotations

import os
import hashlib
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
import zipfile

try:
    from . import provider_settings
except ImportError:  # pragma: no cover - supports direct CLI execution
    import provider_settings
try:
    from ..cancellation import (
        KnowledgeBaseCancelled,
        check_cancelled,
        wait_with_cancellation,
    )
except ImportError:  # pragma: no cover - supports direct CLI execution
    from knowledge_base.cancellation import (
        KnowledgeBaseCancelled,
        check_cancelled,
        wait_with_cancellation,
    )


SUPPORTED_EXTS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"}
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_BATCH_FILES = 50
DEFAULT_BASE_URL = "https://mineru.net/api/v4"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_TIMEOUT = (15, 300)
MINERU_CACHE_VERSION = 1
MINERU_CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
MINERU_CACHE_MAX_BYTES = 4 * 1024 * 1024 * 1024


class MinerUError(RuntimeError):
    """Raised when MinerU cannot accept or process an import file."""


class _RetryableMinerUError(MinerUError):
    pass


@dataclass
class MinerUFile:
    source_path: Path
    relative_path: str
    data_id: str
    upload_url: str = ""
    zip_url: str = ""
    state: str = "queued"
    error: str = ""
    result_path: Path | None = None
    cache_key: str = ""
    cache_hit: bool = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configured_endpoint() -> tuple[str, str]:
    """Read cache-relevant MinerU settings without requiring an API key."""
    try:
        raw = provider_settings.mineru_config()
    except Exception:
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    base_url = str(raw.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")
    model_version = (
        str(raw.get("model_version") or DEFAULT_MODEL_VERSION).strip()
        or DEFAULT_MODEL_VERSION
    )
    return base_url, model_version


def _cache_key(path: Path, relative_path: str, *, base_url: str, model_version: str) -> str:
    payload = "\x1f".join(
        (
            str(MINERU_CACHE_VERSION),
            str(base_url),
            str(model_version),
            str(relative_path).replace("\\", "/"),
            _file_sha256(path),
        )
    )
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.zip"


def _valid_zip(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _store_cache(source: Path, destination: Path) -> bool:
    """Persist a complete MinerU ZIP, never exposing a partial cache entry."""
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.part"
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, temporary)
        if not _valid_zip(temporary):
            return False
        os.replace(temporary, destination)
        return True
    except (OSError, zipfile.BadZipFile):
        return False
    finally:
        temporary.unlink(missing_ok=True)


def cleanup_cache(
    cache_dir: str | Path,
    *,
    max_age_seconds: int = MINERU_CACHE_MAX_AGE_SECONDS,
    max_bytes: int = MINERU_CACHE_MAX_BYTES,
) -> dict[str, int]:
    """Remove incomplete, expired, and over-quota MinerU cache artifacts."""
    root = Path(cache_dir)
    if not root.is_dir():
        return {"removed": 0, "bytes": 0}
    now = time.time()
    removed = 0
    removed_bytes = 0
    entries: list[tuple[float, int, Path]] = []
    for path in root.iterdir():
        if path.is_dir() or path.suffix.lower() != ".zip":
            if path.name.startswith(".") and path.is_file():
                try:
                    removed_bytes += path.stat().st_size
                except OSError:
                    pass
                path.unlink(missing_ok=True)
                removed += 1
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if now - stat.st_mtime > max(0, int(max_age_seconds)) or not _valid_zip(path):
            removed_bytes += stat.st_size
            path.unlink(missing_ok=True)
            removed += 1
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _mtime, size, _path in entries)
    for _mtime, size, path in sorted(entries):
        if total <= max(0, int(max_bytes)):
            break
        path.unlink(missing_ok=True)
        total -= size
        removed += 1
        removed_bytes += size
    return {"removed": removed, "bytes": removed_bytes}


def load_config() -> dict[str, str]:
    """Load and validate MinerU settings without exposing the API key."""
    raw = provider_settings.mineru_config()
    api_key = str(raw.get("api_key") or "").strip()
    if not api_key:
        raise MinerUError(
            "未配置 MinerU API-Key。请在 mykey.py 的 mineru_config 中填写 api_key，"
            "或设置 MINERU_API_KEY。"
        )
    return {
        "api_key": api_key,
        "base_url": str(raw.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/"),
        "model_version": (
            str(raw.get("model_version") or DEFAULT_MODEL_VERSION).strip()
            or DEFAULT_MODEL_VERSION
        ),
    }


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _retry(
    label: str,
    operation: Callable[[], Any],
    attempts: int = 3,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        check_cancelled(cancelled)
        try:
            return operation()
        except _RetryableMinerUError as error:
            last_error = error
        except requests.RequestException as error:
            last_error = error
        if attempt + 1 < max(1, int(attempts)):
            wait_with_cancellation(min(2 ** attempt, 8), cancelled)
    raise MinerUError(f"{label}失败：{last_error}") from last_error


def _json_response(response: requests.Response, context: str) -> dict[str, Any]:
    status = int(response.status_code)
    if status == 408 or status == 409 or status == 425 or status == 429 or status >= 500:
        raise _RetryableMinerUError(f"{context} HTTP {status}")
    try:
        payload = response.json()
    except ValueError as error:
        raise MinerUError(f"{context} 返回了非 JSON 响应（HTTP {status}）") from error
    if not isinstance(payload, dict):
        raise MinerUError(f"{context} 返回格式异常")
    if status >= 400 or (payload.get("code") not in (0, None)):
        message = payload.get("msg") or payload.get("message") or str(payload)[:500]
        raise MinerUError(f"{context}失败：{message}")
    return payload


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    context: str,
    json_body: dict[str, Any] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        with requests.request(
            method,
            url,
            headers=headers,
            json=json_body,
            timeout=DEFAULT_TIMEOUT,
        ) as response:
            return _json_response(response, context)

    return _retry(
        context,
        operation,
        cancelled=cancelled,
    )


def _request_upload_urls(
    config: dict[str, str],
    files: list[MinerUFile],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    payload = {
        "files": [
            {"name": item.relative_path, "data_id": item.data_id}
            for item in files
        ],
        "model_version": config["model_version"],
    }
    result = _request_json(
        "POST",
        f"{config['base_url']}/file-urls/batch",
        headers=_headers(config),
        context="申请 MinerU 上传地址",
        json_body=payload,
        cancelled=cancelled,
    )
    data = result.get("data") or {}
    batch_id = str(data.get("batch_id") or "")
    urls = data.get("file_urls") or []
    if not batch_id or len(urls) != len(files):
        raise MinerUError("MinerU 上传地址响应不完整")
    for item, value in zip(files, urls):
        item.upload_url = (
            str(value.get("url") or value.get("upload_url") or "")
            if isinstance(value, dict)
            else str(value or "")
        )
        if not item.upload_url:
            raise MinerUError(f"MinerU 未返回 {item.relative_path} 的上传地址")
    return batch_id


def _upload(
    item: MinerUFile,
    cancelled: Callable[[], bool] | None = None,
) -> MinerUFile:
    def operation() -> MinerUFile:
        try:
            with item.source_path.open("rb") as handle, requests.put(
                    item.upload_url,
                    data=handle,
                    timeout=DEFAULT_TIMEOUT,
            ) as response:
                status = int(response.status_code)
        except requests.RequestException:
            raise
        if status in (408, 409, 425, 429) or status >= 500:
            raise _RetryableMinerUError(f"上传 {item.relative_path} HTTP {status}")
        if status < 200 or status >= 300:
            raise MinerUError(f"上传 {item.relative_path} 失败（HTTP {status}）")
        return item

    return _retry(
        f"上传 {item.relative_path}",
        operation,
        cancelled=cancelled,
    )


def _poll_batch(
    config: dict[str, str],
    batch_id: str,
    files: list[MinerUFile],
    on_update: Callable[[MinerUFile], None],
    cancelled: Callable[[], bool] | None = None,
) -> None:
    deadline = time.monotonic() + 60 * 60
    url = f"{config['base_url']}/extract-results/batch/{batch_id}"
    while time.monotonic() < deadline:
        check_cancelled(cancelled)
        result = _request_json(
            "GET",
            url,
            headers=_headers(config),
            context="查询 MinerU 解析状态",
            cancelled=cancelled,
        )
        rows = (result.get("data") or {}).get("extract_result") or []
        by_data_id = {
            str(row.get("data_id") or ""): row
            for row in rows
            if isinstance(row, dict)
        }
        by_name = {
            str(row.get("file_name") or ""): row
            for row in rows
            if isinstance(row, dict)
        }
        complete = True
        for item in files:
            row = by_data_id.get(item.data_id) or by_name.get(item.relative_path)
            if not row:
                complete = False
                continue
            state = str(row.get("state") or "processing").strip().lower()
            if state in {"done", "success", "succeeded"}:
                item.zip_url = str(row.get("full_zip_url") or "")
                item.state = "done" if item.zip_url else "failed"
                if not item.zip_url:
                    item.error = "MinerU 未返回结果下载地址"
            elif state in {"failed", "error"}:
                item.state = "failed"
                item.error = str(row.get("err_msg") or row.get("message") or "MinerU 解析失败")
                if "200" in item.error and ("page" in item.error.lower() or "页" in item.error):
                    item.error += "；当前导入器会自动分片，若仍失败请检查该 PDF 是否损坏或加密"
            else:
                item.state = state or "processing"
                complete = False
            on_update(item)
        if complete and all(item.state in {"done", "failed"} for item in files):
            return
        wait_with_cancellation(2, cancelled)
    for item in files:
        if item.state not in {"done", "failed"}:
            item.state = "failed"
            item.error = "等待 MinerU 解析结果超时"
            on_update(item)


def _download(
    url: str,
    target: Path,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    # CDN downloads are more prone to a transient TLS EOF than the API calls
    # above.  Never expose a partial ZIP as the final artifact: every attempt
    # writes to its own temporary file and only replaces ``target`` after the
    # complete response has been consumed.
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")

    def operation() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        received = 0
        try:
            with requests.get(
                url,
                stream=True,
                timeout=DEFAULT_TIMEOUT,
                headers={"Accept": "application/zip", "User-Agent": "GenericAgent/KB"},
            ) as response:
                status = int(response.status_code)
                if status in (408, 409, 425, 429) or status >= 500:
                    raise _RetryableMinerUError(f"下载解析结果 HTTP {status}")
                if status >= 400:
                    raise MinerUError(f"下载解析结果失败（HTTP {status}）")
                expected = str(response.headers.get("Content-Length") or "").strip()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        check_cancelled(cancelled)
                        if chunk:
                            handle.write(chunk)
                            received += len(chunk)
                if expected and not str(response.headers.get("Content-Encoding") or "").strip():
                    try:
                        if received != int(expected):
                            raise requests.exceptions.ConnectionError(
                                f"下载结果不完整：收到 {received} / {expected} bytes"
                            )
                    except ValueError:
                        pass
                if received <= 0:
                    raise requests.exceptions.ConnectionError("下载结果为空")
            check_cancelled(cancelled)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    try:
        _retry(
            "下载 MinerU 解析结果",
            operation,
            attempts=5,
            cancelled=cancelled,
        )
    finally:
        temporary.unlink(missing_ok=True)


def process_batches(
    files: Iterable[tuple[Path, str]],
    download_dir: Path,
    *,
    on_update: Callable[[MinerUFile], None],
    cancelled: Callable[[], bool] | None = None,
    cache_dir: Path | None = None,
) -> list[MinerUFile]:
    """Upload, poll, and download files, reusing complete cached results."""
    base_url, model_version = _configured_endpoint()
    cache_root = Path(cache_dir).resolve() if cache_dir else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cleanup_cache(cache_root)

    jobs: list[MinerUFile] = []
    remote_jobs: list[MinerUFile] = []
    for path, relative_path in files:
        check_cancelled(cancelled)
        path = Path(path).resolve()
        if not path.is_file():
            raise MinerUError(f"待处理文件不存在：{path}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise MinerUError(
                f"文件过大：{relative_path}（{size} bytes，限制 {MAX_FILE_BYTES} bytes）"
            )
        relative = str(relative_path).replace("\\", "/")
        cache_key = (
            _cache_key(
                path,
                relative,
                base_url=base_url,
                model_version=model_version,
            )
            if cache_root is not None
            else ""
        )
        cached_path = _cache_path(cache_root, cache_key) if cache_root else None
        if cached_path is not None and _valid_zip(cached_path):
            item = MinerUFile(
                path,
                relative,
                cache_key,
                state="downloaded",
                result_path=cached_path,
                cache_key=cache_key,
                cache_hit=True,
            )
            jobs.append(item)
            on_update(item)
            continue
        if cached_path is not None and cached_path.exists():
            cached_path.unlink(missing_ok=True)
        item = MinerUFile(
            path,
            relative,
            uuid.uuid4().hex,
            cache_key=cache_key,
        )
        jobs.append(item)
        remote_jobs.append(item)

    if not remote_jobs:
        return jobs

    config = load_config()
    download_dir.mkdir(parents=True, exist_ok=True)

    for first in range(0, len(remote_jobs), MAX_BATCH_FILES):
        check_cancelled(cancelled)
        batch = remote_jobs[first:first + MAX_BATCH_FILES]
        batch_id = _request_upload_urls(config, batch, cancelled=cancelled)
        for item in batch:
            item.state = "uploading"
            on_update(item)
        pool = ThreadPoolExecutor(max_workers=min(4, len(batch)))
        futures = {}
        try:
            futures = {
                pool.submit(_upload, item, cancelled): item
                for item in batch
            }
            for future in as_completed(futures):
                check_cancelled(cancelled)
                item = futures[future]
                try:
                    future.result()
                    item.state = "processing"
                except KnowledgeBaseCancelled:
                    raise
                except Exception as error:
                    item.state = "failed"
                    item.error = str(error)
                on_update(item)
        except KnowledgeBaseCancelled:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        pending = [item for item in batch if item.state != "failed"]
        if pending:
            _poll_batch(
                config,
                batch_id,
                pending,
                on_update,
                cancelled=cancelled,
            )
        for item in batch:
            check_cancelled(cancelled)
            if item.state != "done":
                continue
            item.state = "downloading"
            on_update(item)
            try:
                _download(
                    item.zip_url,
                    download_dir / f"{item.data_id}.zip",
                    cancelled=cancelled,
                )
                item.result_path = download_dir / f"{item.data_id}.zip"
                if cache_root is not None and item.cache_key:
                    cached_path = _cache_path(cache_root, item.cache_key)
                    if _store_cache(item.result_path, cached_path):
                        item.result_path = cached_path
                item.state = "downloaded"
            except Exception as error:
                item.state = "failed"
                item.error = str(error)
            on_update(item)
    return jobs
