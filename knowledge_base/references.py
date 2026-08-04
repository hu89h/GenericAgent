"""Stable, user-facing knowledge-base reference fields.

The retrieval layer may carry storage details such as processed file names and
internal source references.  This module keeps the small public reference
contract in one place without introducing a stateful reference object.
"""
from __future__ import annotations

import os
import html
import re
from typing import Any


REFERENCE_FIELDS = (
    "kind",
    "kb_id",
    "data_id",
    "source_data_id",
    "chunk_index",
    "source_chunk_index",
    "image_id",
    "ref_key",
    "title",
    "file_name",
    "source_file_name",
    "source_section",
    "citation_label",
    "ref",
)

_CHUNK_CONTEXT_RE = re.compile(r"(?m)^\s*章节路径：/[^\r\n]*\r?\n+")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^\r\n)]*(?:\)|$)")
_GENERATED_IMAGE_PATH_RE = re.compile(
    r"\S*?\.assets-[^\s()]*[\\/][^\s()]+\.(?:png|jpe?g|jp2|webp|gif|bmp|tiff?)(?:\?[^\s)]*)?\)?",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _relative(value: Any) -> str:
    return _text(value).replace("\\", "/").lstrip("/")


def _source_name(value: Any) -> str:
    value = _relative(value)
    name = value.rsplit("/", 1)[-1] if value else ""
    # Imported processed names may carry the deterministic content prefix.
    return re.sub(r"^[0-9a-f]{12,16}-", "", name, flags=re.IGNORECASE)


def _label_text(value: Any) -> str:
    value = html.unescape(_text(value))
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" *_`")


def _section_name(value: Any) -> str:
    parts = [part.strip() for part in _text(value).replace("\\", "/").split("/") if part.strip()]
    return parts[-1] if len(parts) > 1 else ""


def section_label(value: Any) -> str:
    """Return only the final human-readable heading from a header path."""
    return _section_name(value)


def clean_public_text(value: Any) -> str:
    """Remove generated chunk context before text reaches the model/UI."""
    text = _CHUNK_CONTEXT_RE.sub("", _text(value))
    text = _MARKDOWN_IMAGE_RE.sub(
        lambda match: f"[图片：{match.group(1).strip()}]" if match.group(1).strip() else "[图片]",
        text,
    )
    text = _GENERATED_IMAGE_PATH_RE.sub("[图片]", text)
    return text.strip()


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _image_label(item: dict[str, Any]) -> str:
    for key in ("display_label", "caption", "ref_key", "title", "description"):
        value = _label_text(item.get(key))
        if value and value.lower() != "image":
            return value
    return "图片"


def _citation_label(item: dict[str, Any], kind: str, source_name: str) -> str:
    if kind == "image":
        label = _image_label(item)
        return f"{label} · {source_name}" if source_name else label

    label = source_name or _text(item.get("title")) or "知识库内容"
    section = _section_name(item.get("header_path"))
    if section and section not in label:
        label = f"{label} · {section}"
    return label


def reference_fields(item: dict[str, Any] | None, *, kind: str | None = None) -> dict[str, Any]:
    """Return the stable reference fields shared by all knowledge surfaces."""
    raw = dict(item or {})
    resolved_kind = _text(kind or raw.get("kind"))
    if resolved_kind in {"text", "document_chunk", "source"}:
        resolved_kind = "document"
    elif resolved_kind not in {"document", "image"}:
        resolved_kind = "image" if raw.get("image_id") or raw.get("image_path") else "document"

    source_file_name = _source_name(
        raw.get("source_file_name") or raw.get("source_name") or raw.get("file_name")
    )
    source_section = section_label(raw.get("header_path"))
    title = _label_text(raw.get("title"))
    if resolved_kind == "document" and os.path.splitext(title)[1]:
        title = _source_name(title)
    fields = {
        "kind": resolved_kind,
        "kb_id": _text(raw.get("kb_id")),
        "data_id": _text(raw.get("data_id")),
        "source_data_id": _text(raw.get("source_data_id")),
        "chunk_index": _integer(raw.get("chunk_index"), 0),
        "source_chunk_index": _integer(raw.get("source_chunk_index"), -1),
        "image_id": _text(raw.get("image_id")),
        "ref_key": _text(raw.get("ref_key")),
        "title": title,
        "file_name": _relative(raw.get("file_name")),
        "source_file_name": source_file_name,
        "source_section": source_section,
        "ref": _relative(raw.get("ref") or raw.get("source_ref")),
    }
    fields["citation_label"] = _citation_label(raw | fields, resolved_kind, source_file_name)
    return fields


def with_reference(item: dict[str, Any] | None, *, kind: str | None = None) -> dict[str, Any]:
    """Copy an item and replace its reference fields with normalized values."""
    result = dict(item or {})
    result.update(reference_fields(result, kind=kind))
    return result


def public_reference(item: dict[str, Any] | None, *, kind: str | None = None) -> dict[str, Any]:
    """Return only stable fields safe for agent/session/UI metadata."""
    normalized = reference_fields(item, kind=kind)
    result = {key: normalized[key] for key in REFERENCE_FIELDS}
    # The resolver uses the internal relative path, but model/session metadata
    # should expose only the original file name.
    result["file_name"] = result["source_file_name"] or result["title"]
    return result
