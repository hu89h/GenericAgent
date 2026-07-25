"""Small synchronous client for the MinerU v4 document parsing API.

The client owns only HTTP interaction.  Knowledge-base persistence lives in
:mod:`knowledge_base.importer`; PDF page-limit validation happens before a
remote job is submitted.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

try:
    from . import provider_settings
except ImportError:  # pragma: no cover - supports direct CLI execution
    import provider_settings


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


def _retry(label: str, operation: Callable[[], Any], attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            return operation()
        except _RetryableMinerUError as error:
            last_error = error
        except requests.RequestException as error:
            last_error = error
        if attempt + 1 < max(1, int(attempts)):
            time.sleep(min(2 ** attempt, 8))
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
    )


def _request_upload_urls(config: dict[str, str], files: list[MinerUFile]) -> str:
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


def _upload(item: MinerUFile) -> MinerUFile:
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

    return _retry(f"上传 {item.relative_path}", operation)


def _poll_batch(
    config: dict[str, str],
    batch_id: str,
    files: list[MinerUFile],
    on_update: Callable[[MinerUFile], None],
) -> None:
    deadline = time.monotonic() + 60 * 60
    url = f"{config['base_url']}/extract-results/batch/{batch_id}"
    while time.monotonic() < deadline:
        result = _request_json(
            "GET",
            url,
            headers=_headers(config),
            context="查询 MinerU 解析状态",
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
        time.sleep(2)
    for item in files:
        if item.state not in {"done", "failed"}:
            item.state = "failed"
            item.error = "等待 MinerU 解析结果超时"
            on_update(item)


def _download(url: str, target: Path) -> None:
    def operation() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException:
            raise
        status = int(response.status_code)
        if status in (408, 409, 425, 429) or status >= 500:
            response.close()
            raise _RetryableMinerUError(f"下载解析结果 HTTP {status}")
        if status >= 400:
            response.close()
            raise MinerUError(f"下载解析结果失败（HTTP {status}）")
        try:
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        finally:
            response.close()

    _retry("下载 MinerU 解析结果", operation)


def process_batches(
    files: Iterable[tuple[Path, str]],
    download_dir: Path,
    *,
    on_update: Callable[[MinerUFile], None],
) -> list[MinerUFile]:
    """Upload, poll, and download all supplied files in MinerU batches."""
    config = load_config()
    jobs = []
    for path, relative_path in files:
        path = Path(path).resolve()
        if not path.is_file():
            raise MinerUError(f"待处理文件不存在：{path}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise MinerUError(
                f"文件过大：{relative_path}（{size} bytes，限制 {MAX_FILE_BYTES} bytes）"
            )
        jobs.append(MinerUFile(path, str(relative_path).replace("\\", "/"), uuid.uuid4().hex))
    download_dir.mkdir(parents=True, exist_ok=True)

    for first in range(0, len(jobs), MAX_BATCH_FILES):
        batch = jobs[first:first + MAX_BATCH_FILES]
        batch_id = _request_upload_urls(config, batch)
        for item in batch:
            item.state = "uploading"
            on_update(item)
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
            futures = {pool.submit(_upload, item): item for item in batch}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    item.state = "processing"
                except Exception as error:
                    item.state = "failed"
                    item.error = str(error)
                on_update(item)
        pending = [item for item in batch if item.state != "failed"]
        if pending:
            _poll_batch(config, batch_id, pending, on_update)
        for item in batch:
            if item.state != "done":
                continue
            item.state = "downloading"
            on_update(item)
            try:
                _download(item.zip_url, download_dir / f"{item.data_id}.zip")
                item.state = "downloaded"
            except Exception as error:
                item.state = "failed"
                item.error = str(error)
            on_update(item)
    return jobs
