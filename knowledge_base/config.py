"""Knowledge-base registry and deterministic managed paths.

The registry stores identity only.  Every runtime path is derived from
``DATA_ROOT/<kb_id>/active`` so import, retrieval, and deletion cannot disagree
about which files belong to the application.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CLIENT_DIR)
CONFIG_PATH = os.environ.get("GA_KB_CONFIG") or os.path.join(ROOT, "data", "kb.yaml")
DATA_ROOT = os.environ.get("GA_KB_DATA_ROOT") or os.path.join(ROOT, "data", "kbs")
_KB_ID_BAD = set(' \t\r\n/\\:[]{}"\'')


def canonical_source_path(source_dir: str) -> str:
    return os.path.normpath(os.path.realpath(os.path.expanduser(str(source_dir or ""))))


def kb_id_for_source(source_dir: str) -> str:
    canonical = canonical_source_path(source_dir)
    identity = os.path.normcase(canonical).replace("/", "\\")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"kb-{digest}"


def valid_kb_id(kb_id: str) -> bool:
    value = str(kb_id or "").strip()
    if not value or len(value) > 64 or ".." in value:
        return False
    return not any(char in _KB_ID_BAD for char in value)


def kb_root(kb_id: str) -> str:
    if not valid_kb_id(kb_id):
        raise ValueError("知识库 ID 不合法")
    root = os.path.realpath(DATA_ROOT)
    unresolved = os.path.abspath(os.path.join(root, str(kb_id)))
    if os.path.islink(unresolved):
        raise ValueError("拒绝使用符号链接知识库目录")
    target = os.path.realpath(unresolved)
    if os.path.commonpath((root, target)) != root or target == root:
        raise ValueError("知识库路径越界")
    return target


def active_root(kb_id: str) -> str:
    return os.path.join(kb_root(kb_id), "active")


def processed_path(kb_id: str) -> str:
    return os.path.join(active_root(kb_id), "processed")


def staging_root(kb_id: str) -> str:
    return os.path.join(kb_root(kb_id), "staging")


def manifest_path(kb_id: str) -> str:
    return os.path.join(active_root(kb_id), "manifest.json")


def records_path(kb_id: str) -> str:
    return os.path.join(active_root(kb_id), "records.jsonl")


def _parse_scalar(value: str):
    value = str(value or "").strip().strip("'\"")
    lowered = value.lower()
    if lowered in ("true", "yes", "on", "1"):
        return True
    if lowered in ("false", "no", "off", "0"):
        return False
    return value


def _parse_config_text(text: str) -> dict:
    text = str(text or "").strip()
    if not text:
        return {"knowledge_base": {}}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"knowledge_base": {}}
    except Exception:
        pass

    block: dict[str, dict] = {}
    current = None
    in_kb = False
    for line in text.splitlines():
        raw = line.rstrip()
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if re.match(r"^knowledge_base\s*:\s*(\{\})?\s*$", raw):
            in_kb = True
            continue
        if not in_kb:
            continue
        match = re.match(r"^\s{2}([^:#]+)\s*:\s*$", raw)
        if match:
            current = match.group(1).strip().strip("'\"")
            block[current] = {}
            continue
        match = re.match(r"^\s{4}([A-Za-z0-9_]+)\s*:\s*(.*)$", raw)
        if match and current:
            # Unknown fields are parsed but never persisted.  This is not a
            # compatibility path: the runtime only consumes name/source_path.
            block[current][match.group(1)] = _parse_scalar(match.group(2))
    return {"knowledge_base": block}


def _load_raw_config(path: str | None = None) -> dict:
    path = path or CONFIG_PATH
    if not os.path.isfile(path):
        return {"knowledge_base": {}}
    try:
        with open(path, encoding="utf-8") as handle:
            data = _parse_config_text(handle.read())
        if not isinstance(data.get("knowledge_base"), dict):
            return {"knowledge_base": {}}
        return data
    except Exception:
        return {"knowledge_base": {}}


def _kb_block(data: dict | None) -> dict[str, dict]:
    raw = (data or {}).get("knowledge_base")
    if not isinstance(raw, dict):
        return {}
    return {
        str(kb_id): dict(value)
        for kb_id, value in raw.items()
        if valid_kb_id(str(kb_id)) and isinstance(value, dict)
    }


def _serialize_config(data: dict | None) -> str:
    block = _kb_block(data)
    if not block:
        return "knowledge_base: {}\n"
    lines = ["knowledge_base:"]
    for kb_id, config in block.items():
        lines.append(f"  {kb_id}:")
        source = str(config.get("source_path") or "").replace(os.sep, "/")
        if source:
            lines.append(f"    source_path: {source}")
        name = str(config.get("name") or "").strip()
        if name:
            lines.append(f"    name: {name}")
    return "\n".join(lines) + "\n"


def _dump_raw_config(data: dict, path: str | None = None) -> None:
    """Atomically persist the small project-owned YAML subset."""
    path = path or CONFIG_PATH
    directory = os.path.dirname(os.path.realpath(path)) or os.curdir
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".kb-config-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_serialize_config(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _runtime_row(kb_id: str, config: dict) -> dict:
    path = processed_path(kb_id)
    active = active_root(kb_id)
    return {
        "id": kb_id,
        "name": str(config.get("name") or kb_id),
        "source_path": str(config.get("source_path") or ""),
        "root": kb_root(kb_id),
        "active_path": active,
        "path": path,
        "manifest_path": os.path.join(active, "manifest.json"),
        "records_path": os.path.join(active, "records.jsonl"),
        "exists": os.path.isdir(path),
    }


def load_config(path: str | None = None) -> list[dict]:
    path = path or CONFIG_PATH
    block = _kb_block(_load_raw_config(path))
    return [_runtime_row(kb_id, config) for kb_id, config in block.items()]


def kb_by_id(kb_id: str, path: str | None = None) -> dict | None:
    path = path or CONFIG_PATH
    value = str(kb_id or "").strip()
    return next((kb for kb in load_config(path) if kb["id"] == value), None)


def upsert_kb(
    kb_id: str,
    *,
    name: str = "",
    source_path: str = "",
    config_path: str | None = None,
) -> list[dict]:
    config_path = config_path or CONFIG_PATH
    kb_id = str(kb_id or "").strip()
    if not valid_kb_id(kb_id):
        raise ValueError("知识库 ID 不合法")
    source = canonical_source_path(source_path) if source_path else ""
    data = _load_raw_config(config_path)
    block = _kb_block(data)
    block[kb_id] = {
        "source_path": source,
        "name": str(name or Path(source).name or kb_id).strip(),
    }
    data["knowledge_base"] = block
    _dump_raw_config(data, config_path)
    return load_config(config_path)


def create_kb(name: str, config_path: str | None = None) -> dict:
    """Create an empty user-named knowledge base ready for document additions."""
    import uuid

    label = str(name or "").strip()
    if not label:
        raise ValueError("知识库名称不能为空")
    if len(label) > 120:
        raise ValueError("知识库名称过长")
    config_path = config_path or CONFIG_PATH
    data = _load_raw_config(config_path)
    block = _kb_block(data)
    while True:
        kb_id = f"kb-{uuid.uuid4().hex[:16]}"
        if kb_id not in block:
            break
    block[kb_id] = {"name": label, "source_path": ""}
    data["knowledge_base"] = block
    _dump_raw_config(data, config_path)
    return next(row for row in load_config(config_path) if row["id"] == kb_id)


def remove_kb(kb_id: str, config_path: str | None = None) -> bool:
    config_path = config_path or CONFIG_PATH
    value = str(kb_id or "").strip()
    data = _load_raw_config(config_path)
    block = _kb_block(data)
    if value not in block:
        return False
    block.pop(value)
    data["knowledge_base"] = block
    _dump_raw_config(data, config_path)
    return True


__all__ = [
    "CLIENT_DIR",
    "CONFIG_PATH",
    "DATA_ROOT",
    "ROOT",
    "active_root",
    "canonical_source_path",
    "create_kb",
    "kb_by_id",
    "kb_id_for_source",
    "kb_root",
    "load_config",
    "manifest_path",
    "processed_path",
    "records_path",
    "remove_kb",
    "staging_root",
    "upsert_kb",
    "valid_kb_id",
]
