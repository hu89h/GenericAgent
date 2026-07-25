"""Knowledge-base registration and ``kb.yaml`` persistence.

The registry is intentionally independent from indexing.  It only resolves
paths and reads/writes the small project-owned YAML subset, so listing or
editing a KB does not import Zvec or the embedding clients.
"""
from __future__ import annotations

import json
import hashlib
import os
import re


CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CLIENT_DIR)
CONFIG_PATH = os.environ.get("GA_KB_CONFIG") or os.path.join(ROOT, "data", "kb.yaml")
DATA_ROOT = os.path.join(ROOT, "data", "kbs")
_KB_ID_BAD = set(' \t\r\n/\\:[]{}"\'')


def resolve_path(path):
    """Resolve a configured path, preferring project and package roots."""
    path = os.path.expanduser(str(path or "").strip())
    if os.path.isabs(path):
        return os.path.normpath(path)
    for base in (ROOT, CLIENT_DIR):
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.isdir(candidate):
            return candidate
    return os.path.normpath(os.path.join(ROOT, path))


def canonical_source_path(source_dir):
    """Return the normalized absolute path used for imported KB identity."""
    return os.path.normpath(os.path.realpath(os.path.expanduser(str(source_dir or ""))))


def kb_id_for_source(source_dir):
    """Return a stable package-safe ID derived from the complete source path."""
    canonical = canonical_source_path(source_dir)
    identity = os.path.normcase(canonical).replace("/", "\\")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"kb-{digest}"


def _parse_scalar(value):
    value = str(value or "").strip().strip("'\"")
    lowered = value.lower()
    if lowered in ("true", "yes", "on", "1"):
        return True
    if lowered in ("false", "no", "off", "0"):
        return False
    return value


def _parse_config_text(text):
    text = str(text or "").strip()
    if not text:
        return {"knowledge_base": {}}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"knowledge_base": {}}
    except Exception:
        pass

    block = {}
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
            block[current][match.group(1)] = _parse_scalar(match.group(2))
    return {"knowledge_base": block}


def _load_raw_config(path=CONFIG_PATH):
    """Read the project-owned YAML/JSON subset without a PyYAML dependency."""
    if not os.path.isfile(path):
        return {"knowledge_base": {}}
    try:
        with open(path, encoding="utf-8") as handle:
            data = _parse_config_text(handle.read())
        if not isinstance(data.get("knowledge_base"), dict):
            data["knowledge_base"] = {}
        return data
    except Exception:
        return {"knowledge_base": {}}


def _kb_block(data):
    """Normalize dict/list registry forms to an ordered ``id -> config`` dict."""
    knowledge_bases = (data or {}).get("knowledge_base")
    result = {}
    if isinstance(knowledge_bases, dict):
        for key, value in knowledge_bases.items():
            result[str(key)] = dict(value) if isinstance(value, dict) else {}
    elif isinstance(knowledge_bases, list):
        for index, config in enumerate(knowledge_bases):
            if isinstance(config, dict):
                kb_id = str(config.get("id") or f"kb{index + 1}")
                result[kb_id] = {key: value for key, value in config.items() if key != "id"}
    return result


def load_config(path=CONFIG_PATH):
    """Return normalized registry rows with resolved paths and existence flags."""
    data = _load_raw_config(path)
    knowledge_bases = (data or {}).get("knowledge_base") or {}
    if isinstance(knowledge_bases, dict):
        items = knowledge_bases.items()
    elif isinstance(knowledge_bases, list):
        items = [
            (config.get("id") or f"kb{index + 1}", config)
            for index, config in enumerate(knowledge_bases)
            if isinstance(config, dict)
        ]
    else:
        items = []

    result = []
    for kb_id, config in items:
        if not isinstance(config, dict):
            continue
        raw_path = str(config.get("path", "") or "")
        absolute_path = resolve_path(raw_path)
        result.append({
            "id": str(kb_id),
            "name": str(config.get("name") or kb_id),
            "path": absolute_path,
            "raw_path": raw_path,
            "source_path": str(config.get("source_path") or ""),
            "exists": os.path.isdir(absolute_path),
        })
    return result


def kb_by_id(kb_id, path=CONFIG_PATH):
    for kb in load_config(path):
        if kb["id"] == kb_id:
            return kb
    return None


def valid_kb_id(kb_id):
    """Return whether an ID is safe for references and local registry keys."""
    value = str(kb_id or "").strip()
    if not value or len(value) > 64 or ".." in value:
        return False
    return not any(char in _KB_ID_BAD for char in value)


def _dump_raw_config(data, path=CONFIG_PATH):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    block = _kb_block(data)
    lines = ["knowledge_base:"]
    if not block:
        lines[0] = "knowledge_base: {}"
    else:
        for kb_id, config in block.items():
            lines.append(f"  {kb_id}:")
            lines.append(f"    path: {str(config.get('path', '')).replace(os.sep, '/')}")
            if config.get("source_path"):
                lines.append(f"    source_path: {str(config.get('source_path')).replace(os.sep, '/')}")
            if config.get("name"):
                lines.append(f"    name: {config.get('name')}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def upsert_kb(kb_id, path=None, name=None, old_id=None,
              config_path=CONFIG_PATH, source_path=None):
    """Create/update a registry entry and return normalized config rows."""
    kb_id = str(kb_id or "").strip()
    if not valid_kb_id(kb_id):
        raise ValueError("知识库 ID 不合法（不能含空格与 / \\ : [ ] 等，且不能包含 ..）")
    data = _load_raw_config(config_path)
    block = _kb_block(data)
    source_id = old_id if (old_id and old_id in block) else (kb_id if kb_id in block else None)
    entry = dict(block.get(source_id, {}) or {}) if source_id else {}
    if path is not None:
        entry["path"] = str(path).strip()
    if source_path is not None:
        source = str(source_path).strip()
        if source:
            entry["source_path"] = source
        else:
            entry.pop("source_path", None)
    if name is not None:
        display_name = str(name).strip()
        if display_name and display_name != kb_id:
            entry["name"] = display_name
        else:
            entry.pop("name", None)
    entry.setdefault("path", "")

    new_block, placed = {}, False
    for key, value in block.items():
        if key == source_id:
            new_block[kb_id] = entry
            placed = True
        elif key == kb_id and source_id != kb_id:
            continue
        else:
            new_block[key] = value
    if not placed:
        new_block[kb_id] = entry
    data["knowledge_base"] = new_block
    _dump_raw_config(data, config_path)
    return load_config(config_path)


def remove_kb(kb_id, config_path=CONFIG_PATH):
    """Remove a registry entry without deleting any user source files."""
    kb_id = str(kb_id or "").strip()
    data = _load_raw_config(config_path)
    block = _kb_block(data)
    if kb_id not in block:
        return False
    block.pop(kb_id, None)
    data["knowledge_base"] = block
    _dump_raw_config(data, config_path)
    return True


__all__ = [
    "CLIENT_DIR",
    "CONFIG_PATH",
    "DATA_ROOT",
    "ROOT",
    "_dump_raw_config",
    "_kb_block",
    "_load_raw_config",
    "kb_by_id",
    "load_config",
    "remove_kb",
    "resolve_path",
    "upsert_kb",
    "valid_kb_id",
]
