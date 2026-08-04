"""Canonical knowledge scope shared by Desktop and Agent enforcement."""
from __future__ import annotations

from typing import Any


MODES = frozenset({"none", "all", "kb", "document", "selection"})


def _text(value: Any) -> str:
    return str(value or "").strip()


def _document(value: Any, kb_id: str) -> dict | None:
    if not isinstance(value, dict):
        return None
    data_id = _text(value.get("data_id"))
    if not data_id or not data_id.startswith(f"{kb_id}::"):
        return None
    result = {"data_id": data_id}
    title = _text(value.get("title"))
    if title:
        result["title"] = title
    return result


def normalize_scope(value: Any) -> dict:
    """Return the only scope shape accepted inside the Python runtime.

    Missing scope means the product default (all knowledge bases).  A scope
    that explicitly declares an unknown mode or omits a required identifier
    fails closed to ``none`` rather than widening access.
    """
    if value is None or value == {}:
        return {"mode": "all", "origin": "chat"}
    if not isinstance(value, dict):
        return {"mode": "none", "origin": "chat"}
    mode = _text(value.get("mode")).lower()
    if mode not in MODES:
        return {"mode": "none", "origin": "chat"}
    origin = _text(value.get("origin")).lower()
    if origin not in {"chat", "knowledge"}:
        origin = "knowledge" if mode in {"kb", "document"} else "chat"
    if mode in {"none", "all"}:
        return {"mode": mode, "origin": origin}

    kb_id = _text(value.get("kb_id"))
    if mode in {"kb", "document"}:
        if not kb_id:
            return {"mode": "none", "origin": origin}
        result = {"mode": mode, "origin": origin, "kb_id": kb_id}
        kb_name = _text(value.get("kb_name"))
        if kb_name:
            result["kb_name"] = kb_name
        if mode == "document":
            document = _document(value, kb_id)
            if document is None:
                return {"mode": "none", "origin": origin}
            result.update(document)
        return result

    targets = value.get("targets")
    if not isinstance(targets, list):
        return {"mode": "none", "origin": origin}
    normalized_targets = []
    seen_kbs = set()
    for raw in targets:
        if not isinstance(raw, dict):
            continue
        target_kb = _text(raw.get("kb_id"))
        if not target_kb or target_kb in seen_kbs:
            continue
        all_documents = raw.get("all_documents") is True
        documents = []
        seen_documents = set()
        if not all_documents:
            for item in raw.get("documents") or []:
                document = _document(item, target_kb)
                if document is None or document["data_id"] in seen_documents:
                    continue
                seen_documents.add(document["data_id"])
                documents.append(document)
            if not documents:
                continue
        target = {"kb_id": target_kb, "all_documents": all_documents}
        kb_name = _text(raw.get("kb_name"))
        if kb_name:
            target["kb_name"] = kb_name
        if documents:
            target["documents"] = documents
        normalized_targets.append(target)
        seen_kbs.add(target_kb)
    if not normalized_targets:
        return {"mode": "none", "origin": origin}
    return {"mode": "selection", "origin": origin, "targets": normalized_targets}


__all__ = ["MODES", "normalize_scope"]
