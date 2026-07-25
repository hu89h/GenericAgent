"""Markdown image assets, contextual metadata, caching, and VLM analysis.

The processor is dependency-injected so it can be used by the KB builder
without importing the backend module.  In particular, image analysis usage is
reported through callbacks rather than reaching into backend globals.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from urllib.parse import unquote


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
_REF_CANDIDATE_RE = re.compile(
    r"(?:图|表)\s*[0-9０-９]{1,3}(?:\s*[-－–—.．·]\s*[0-9０-９]{1,3}){0,3}"
)
_SOURCE_LINE_RE = re.compile(r"^(?:资料)?来源\b|^source\b", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


@dataclass(slots=True)
class ImageOccurrence:
    """One occurrence of an image reference inside one source document."""

    occurrence_id: int
    path: str
    start: int
    end: int
    alt: str = ""
    title: str = ""
    section: str = ""
    near_text: str = ""
    ref_candidates: list[str] = field(default_factory=list)
    ref_key: str = ""
    related_text: str = ""
    related_text_refs: list[dict] = field(default_factory=list)
    chunk_index: int = -1


@dataclass(slots=True)
class ImageContent:
    """Unique image content shared by all occurrences with the same hash."""

    image_sha: str
    image_path: str
    image_abspath: str
    focus: str
    title: str
    near_text: str
    ref_candidates: list[str]
    analysis_meta: dict[str, Any]
    focus_rank: int = 0
    contexts: list[str] = field(default_factory=list)


class DocumentImageIndex:
    """Precompute image context once for one Markdown document.

    The index owns occurrence-level information only.  File hashes and VLM
    results are added later by :class:`ImageAssetProcessor`, so parsing a
    document never performs file IO or network work.
    """

    def __init__(self, processor: "ImageAssetProcessor", body: str) -> None:
        self.body = str(body or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n").strip()
        self._processor = processor
        self._lines = self.body.splitlines()
        self._line_starts = []
        offset = 0
        for line in self._lines:
            self._line_starts.append(offset)
            offset += len(line) + 1
        self._line_clean: list[str] = []
        self._line_ref_candidates: list[list[str]] = []
        self._line_has_image: list[bool] = []
        self._line_is_source: list[bool] = []
        self._line_headings: list[tuple[str, int]] = []
        self._paragraphs: list[dict[str, Any]] = []
        self._body_ref_candidates: list[str] = []
        self.occurrences: list[ImageOccurrence] = []
        self._by_start: dict[int, int] = {}
        self.related_index: dict[str, Any] = {}
        self._prepare_context()
        self._build()

    @classmethod
    def build(cls, processor: "ImageAssetProcessor", body: str) -> "DocumentImageIndex":
        return cls(processor, body)

    def _line_index(self, position: int) -> int:
        if not self._line_starts:
            return 0
        return max(0, min(
            len(self._lines) - 1,
            bisect_right(self._line_starts, max(0, position)) - 1,
        ))

    def _prepare_context(self) -> None:
        """Cache line and paragraph metadata shared by every occurrence."""
        self._body_ref_candidates = self._processor.extract_ref_candidates(self.body)
        heading = ""
        heading_line = -1
        for line_index, line in enumerate(self._lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                heading_line = line_index
            cleaned = self._processor._clean_caption_line(line)
            self._line_clean.append(cleaned)
            self._line_ref_candidates.append(
                self._processor.extract_ref_candidates(cleaned)
            )
            self._line_has_image.append(bool(_MD_IMAGE_RE.search(line)))
            self._line_is_source.append(bool(_SOURCE_LINE_RE.match(cleaned)))
            self._line_headings.append((heading, heading_line))

        for paragraph_index, (offset, paragraph) in enumerate(
            self._processor.paragraphs_with_offsets(self.body)
        ):
            self._paragraphs.append({
                "paragraph_index": paragraph_index,
                "offset": offset,
                "raw": paragraph,
                "text": re.sub(r"\s+", " ", paragraph).strip(),
                "has_image": bool(_MD_IMAGE_RE.search(paragraph)),
                "ref_candidates": self._processor.extract_ref_candidates(paragraph),
            })

    def occurrence_ids_between(self, start: int, end: int) -> list[int]:
        return [
            occurrence.occurrence_id
            for occurrence in self.occurrences
            if occurrence.start < end and occurrence.end > start
        ]

    def occurrence_id_at(self, position: int) -> int | None:
        occurrence_id = self._by_start.get(position)
        return occurrence_id

    def _caption(self, position: int, max_scan: int = 12) -> str:
        line_index = self._line_index(position)
        scanned = 0
        for candidate_line in range(line_index + 1, len(self._lines)):
            scanned += 1
            if scanned > 6:
                break
            if self._line_has_image[candidate_line]:
                break
            value = self._line_clean[candidate_line]
            if not value:
                continue
            if self._line_is_source[candidate_line]:
                continue
            if self._line_ref_candidates[candidate_line]:
                return value

        scanned = 0
        for candidate_line in range(line_index - 1, max(-1, line_index - max_scan - 1), -1):
            scanned += 1
            value = self._line_clean[candidate_line]
            if not value or self._line_is_source[candidate_line]:
                continue
            if self._line_ref_candidates[candidate_line]:
                return value
            if scanned >= max_scan:
                break
        return ""

    def _heading(self, position: int) -> str:
        line_index = self._line_index(position)
        heading, heading_line = self._line_headings[line_index]
        return heading if heading and line_index - heading_line <= 80 else ""

    def _near_text(self, position: int, window: int = 300) -> str:
        start = max(0, position - window)
        end = min(len(self.body), position + window)
        near = _MD_IMAGE_RE.sub(
            lambda match: f"[图片:{(match.group(1) or 'image').strip()}]",
            self.body[start:end],
        )
        return re.sub(r"\s+", " ", near).strip()

    def _build_related_index(self) -> dict[str, Any]:
        candidates = list(self._body_ref_candidates)
        for occurrence in self.occurrences:
            for value in (occurrence.title, occurrence.near_text):
                for candidate in self._processor.extract_ref_candidates(value):
                    if candidate not in candidates:
                        candidates.append(candidate)
        mapping = {candidate: self._processor.local_ref_key(candidate) for candidate in candidates}
        index: dict[str, Any] = {"__mapping": mapping}
        for paragraph in self._paragraphs:
            if paragraph["has_image"]:
                continue
            keys = []
            for candidate in paragraph["ref_candidates"]:
                key = mapping.get(candidate) or self._processor.local_ref_key(candidate)
                if key and key not in keys:
                    keys.append(key)
            for key in keys:
                if self._processor.is_caption_like(paragraph["raw"], key):
                    continue
                index.setdefault(key, []).append({
                    "paragraph_index": paragraph["paragraph_index"],
                    "offset": paragraph["offset"],
                    "text": paragraph["text"],
                })
        return index

    def _build(self) -> None:
        refs = self._processor.scan_image_refs(self.body)
        for occurrence_id, ref in enumerate(refs):
            caption = self._caption(ref["start"])
            alt = (ref.get("alt") or "").strip()
            title = alt if alt and alt.lower() != "image" else (caption or alt or "image")
            occurrence = ImageOccurrence(
                occurrence_id=occurrence_id,
                path=ref.get("path", ""),
                start=int(ref.get("start", 0)),
                end=int(ref.get("end", 0)),
                alt=alt,
                title=title,
                section=self._heading(ref["start"]),
                near_text=self._near_text(ref["start"]),
            )
            occurrence.ref_candidates = self._processor.collect_ref_candidates(
                occurrence.title, caption, occurrence.near_text
            )
            occurrence.ref_key = next(
                (
                    self._processor.local_ref_key(candidate)
                    for candidate in occurrence.ref_candidates
                    if self._processor.local_ref_key(candidate)
                ),
                "",
            )
            self.occurrences.append(occurrence)
            self._by_start[occurrence.start] = occurrence_id

        self.related_index = self._build_related_index()
        for occurrence in self.occurrences:
            if occurrence.ref_key:
                related_text, related_refs = self._processor.related_text_for_ref_key(
                    occurrence.ref_key, self.related_index
                )
            else:
                related_text, related_refs = self._processor.related_text_for_image(
                    occurrence.title, self.related_index
                )
            occurrence.related_text = related_text
            occurrence.related_text_refs = related_refs

    def assign_chunks(self, chunks: list[dict]) -> None:
        for chunk_index, chunk in enumerate(chunks):
            occurrence_ids = chunk.pop("_image_occurrence_ids", []) or []
            for occurrence_id in occurrence_ids:
                if 0 <= int(occurrence_id) < len(self.occurrences):
                    self.occurrences[int(occurrence_id)].chunk_index = chunk_index


class ImageAssetProcessor:
    def __init__(
        self,
        *,
        image_client_fn: Callable[[], Any],
        image_meta_fn: Callable[[], Dict[str, Any]],
        image_cache_dir_fn: Callable[[str], str],
        image_assets_path_fn: Callable[[str], str],
        index_dir_fn: Callable[[str], str],
        merge_usage_fn: Callable[[Dict[str, Any]], None],
        model_usage_delta_fn: Callable[[str, Dict[str, Any] | None, int], Dict[str, Any]],
        concurrency: int = 1,
    ) -> None:
        self._image_client_fn = image_client_fn
        self._image_meta_fn = image_meta_fn
        self._image_cache_dir_fn = image_cache_dir_fn
        self._image_assets_path_fn = image_assets_path_fn
        self._index_dir_fn = index_dir_fn
        self._merge_usage_fn = merge_usage_fn
        self._model_usage_delta_fn = model_usage_delta_fn
        self._concurrency = max(1, int(concurrency))

    def build_document_index(self, body: str) -> DocumentImageIndex:
        return DocumentImageIndex.build(self, body)

    def scan_image_refs(self, body: str):
        out = []
        for match in _MD_IMAGE_RE.finditer(body or ""):
            raw = html.unescape(unquote((match.group(2) or "").strip()))
            if not raw or re.match(r"^[a-z][a-z0-9+.-]*:", raw, re.I):
                continue
            out.append({"alt": (match.group(1) or "").strip(), "path": raw, "start": match.start(), "end": match.end()})
        seen = {(row["start"], row["end"], row["path"]) for row in out}
        for match in _MD_LINK_RE.finditer(body or ""):
            raw = html.unescape(unquote((match.group(2) or "").strip()))
            if not raw or re.match(r"^[a-z][a-z0-9+.-]*:", raw, re.I):
                continue
            path_part = raw.split("?", 1)[0].split("#", 1)[0].lower()
            if not path_part.endswith(_IMAGE_EXTS):
                continue
            key = (match.start(), match.end(), raw)
            if key in seen:
                continue
            out.append({"alt": (match.group(1) or "").strip(), "path": raw, "start": match.start(), "end": match.end()})
        out.sort(key=lambda row: (row["start"], row["end"]))
        return out

    @staticmethod
    def _clean_caption_line(value: str) -> str:
        value = _MD_IMAGE_RE.sub("", str(value or ""))
        value = re.sub(r"\\([~*_#])", r"\1", value)
        value = re.sub(r"^[*_`]+|[*_`]+$", "", value.strip())
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def to_half_width(value: str) -> str:
        out = []
        for char in str(value or ""):
            code = ord(char)
            if code == 0x3000:
                out.append(" ")
            elif 0xFF01 <= code <= 0xFF5E:
                out.append(chr(code - 0xFEE0))
            else:
                out.append(char)
        return "".join(out)

    def local_ref_key(self, value: str) -> str:
        value = self.to_half_width(value)
        value = re.sub(r"[－–—]", "-", value)
        value = re.sub(r"．|·", ".", value)
        match = re.search(r"(图|表)\s*([0-9]{1,3}(?:\s*[-.]\s*[0-9]{1,3}){0,3})", value)
        if not match:
            return ""
        number = re.sub(r"\s+", "", match.group(2))
        return f"{match.group(1)}{number}"

    def extract_ref_candidates(self, text: str):
        out, seen = [], set()
        for match in _REF_CANDIDATE_RE.finditer(self.to_half_width(text or "")):
            raw = match.group(0).strip()
            if raw and raw not in seen:
                out.append(raw)
                seen.add(raw)
        return out

    def collect_ref_candidates(self, *values):
        out = []
        for value in values:
            for candidate in self.extract_ref_candidates(str(value or "")):
                if candidate not in out:
                    out.append(candidate)
            raw = str(value or "").strip()
            if raw and self.local_ref_key(raw) and raw not in out:
                out.append(raw)
        return out

    @staticmethod
    def paragraphs_with_offsets(text: str):
        out = []
        for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text or "", re.S):
            paragraph = match.group(0).strip()
            if paragraph:
                out.append((match.start(), paragraph))
        return out

    def is_caption_like(self, paragraph: str, ref_key: str) -> bool:
        stripped = _MD_IMAGE_RE.sub("", paragraph or "").strip()
        stripped = _MD_HEADING_RE.sub("", stripped).strip()
        if not stripped or stripped.startswith("!["):
            return True
        match = re.match(r"\s*((?:图|表)\s*[0-9０-９]{1,3}(?:\s*[-－–—.．·]\s*[0-9０-９]{1,3}){0,3})", stripped)
        if not match:
            return False
        normalized = self.local_ref_key(match.group(1))
        return bool(normalized and normalized == ref_key and len(stripped) <= 80)

    def compact_ref_text(self, value: str) -> str:
        value = self.to_half_width(value)
        value = re.sub(r"[－–—]", "-", value)
        value = re.sub(r"．|·", ".", value)
        return re.sub(r"\s+", "", value)

    def ref_title_prefix_match(self, title: str, ref_key: str) -> bool:
        compact_title = self.compact_ref_text(title or "")
        compact_key = self.compact_ref_text(ref_key or "")
        if not compact_title or not compact_key:
            return False
        if compact_title.startswith(compact_key):
            return True
        folded_title = re.sub(r"[-.]", "", compact_title)
        folded_key = re.sub(r"[-.]", "", compact_key)
        return bool(folded_key and folded_title.startswith(folded_key))

    def related_text_for_ref_key(self, ref_key: str, related_index, limit: int = 5, max_chars: int = 1800):
        key = self.local_ref_key(ref_key)
        if not key:
            return "", []
        return self.related_text_for_key(key, related_index, limit=limit, max_chars=max_chars)

    def related_text_for_image(self, title: str, related_index, limit: int = 5, max_chars: int = 1800):
        mapping = (related_index or {}).get("__mapping") or {}
        key = ""
        for candidate in [str(title or "").strip()] + self.extract_ref_candidates(title):
            key = mapping.get(candidate) or self.local_ref_key(candidate)
            if key:
                break
        return self.related_text_for_key(key, related_index, title=title, limit=limit, max_chars=max_chars)

    def related_text_for_key(self, key: str, related_index, *, title: str = "", limit: int = 5, max_chars: int = 1800):
        if key not in (related_index or {}):
            matches = [
                candidate_key for candidate_key in (related_index or {})
                if title and candidate_key != "__mapping" and self.ref_title_prefix_match(title, candidate_key)
            ]
            if matches:
                key = max(matches, key=lambda value: len(self.compact_ref_text(value)))
        if not key:
            return "", []
        refs = []
        total = 0
        for row in related_index.get(key, []) or []:
            text = row.get("text", "")
            if not text:
                continue
            remain = max_chars - total
            if remain <= 0 or len(refs) >= limit:
                break
            clipped = text[:remain]
            refs.append({
                "ref_key": key,
                "paragraph_index": row.get("paragraph_index", -1),
                "offset": row.get("offset", -1),
                "text": clipped,
            })
            total += len(clipped)
        return "\n".join(row["text"] for row in refs), refs

    @staticmethod
    def asset_body(asset: Dict[str, Any]) -> str:
        parts = []
        for label, key in (
            ("章节", "section"),
            ("图题", "title"),
            ("图表编号", "ref_key"),
            ("图片描述", "description"),
            ("表格", "table_markdown"),
            ("正文引用", "related_text"),
            ("邻近正文", "near_text"),
        ):
            value = asset.get(key)
            if isinstance(value, list):
                value = "；".join(str(item) for item in value if str(item).strip())
            if value:
                parts.append(f"{label}: {value}")
        return "\n".join(parts).strip()

    def analysis_cache_path(self, kb_path: str, image_sha: str, analysis_meta, focus: str = "general") -> str:
        version = analysis_meta.get("prompt_version", 1)
        model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(analysis_meta.get("model") or "image"))
        focus_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(focus or "general"))
        return os.path.join(self._image_cache_dir_fn(kb_path), f"{image_sha}.v{version}.{model}.{focus_part}.json")

    def load_cached_analysis(self, kb_path: str, image_sha: str, analysis_meta, focus: str = "general"):
        try:
            with open(self.analysis_cache_path(kb_path, image_sha, analysis_meta, focus), encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def save_cached_analysis(self, kb_path: str, image_sha: str, analysis_meta, payload, focus: str = "general") -> None:
        os.makedirs(self._image_cache_dir_fn(kb_path), exist_ok=True)
        path = self.analysis_cache_path(kb_path, image_sha, analysis_meta, focus)
        temp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp, path)

    def _pending_assets_path(self, kb_path: str) -> str:
        return self._image_assets_path_fn(kb_path) + ".pending"

    def write_assets(self, kb_path: str, assets, validation=None, pending: bool = False) -> None:
        """Serialise image assets.

        When ``pending`` is true the payload is written to a
        ``image_assets.json.pending`` sidecar instead of the final path.
        The build coordinator promotes it with :meth:`commit_pending_assets`
        only after the Zvec index has published successfully, so a failed /
        rolled-back index build never leaves the final asset file
        overwritten out of sync with the index (bug S3).
        """
        os.makedirs(self._index_dir_fn(kb_path), exist_ok=True)
        content_fields = ("description", "table_markdown", "uncertain", "analysis_error")
        contents = []
        content_by_id = {}
        occurrences = []
        for raw_asset in assets or []:
            asset = dict(raw_asset or {})
            image_id = str(asset.get("image_id") or "")
            if image_id and image_id not in content_by_id:
                content = {"image_id": image_id}
                for key in content_fields:
                    content[key] = asset.get(key, "" if key != "uncertain" else [])
                content_by_id[image_id] = content
                contents.append(content)
            for key in content_fields:
                asset.pop(key, None)
            occurrences.append(asset)
        payload = {
            "schema_version": 6,
            "built_at": int(time.time()),
            "analysis": self._image_meta_fn(),
            "n_assets": len(occurrences),
            "n_contents": len(contents),
            "validation": validation if isinstance(validation, dict) else {},
            "contents": contents,
            "assets": occurrences,
        }
        path = self._pending_assets_path(kb_path) if pending else self._image_assets_path_fn(kb_path)
        temp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temp, path)
        finally:
            try:
                os.remove(temp)
            except OSError:
                pass

    def commit_pending_assets(self, kb_path: str) -> bool:
        """Atomically promote a pending asset file to the final path.

        Returns True if a pending file existed and was promoted, False if
        there was nothing to commit (e.g. an up-to-date build that never
        regenerated assets)."""
        pending = self._pending_assets_path(kb_path)
        if not os.path.exists(pending):
            return False
        os.replace(pending, self._image_assets_path_fn(kb_path))
        return True

    def discard_pending_assets(self, kb_path: str) -> None:
        """Drop a pending asset file left by a failed index build."""
        pending = self._pending_assets_path(kb_path)
        try:
            os.remove(pending)
        except OSError:
            pass

    def load_assets(self, kb_path: str, prefer_pending: bool = False):
        path = self._image_assets_path_fn(kb_path)
        if prefer_pending:
            pending = self._pending_assets_path(kb_path)
            if os.path.exists(pending):
                path = pending
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            assets = payload.get("assets") if isinstance(payload, dict) else None
            if not isinstance(assets, list):
                raise RuntimeError(f"图片资产文件格式无效：{path}")
            contents = payload.get("contents") if isinstance(payload, dict) else None
            if not isinstance(contents, list):
                return assets
            content_by_id = {
                str(content.get("image_id") or ""): content
                for content in contents
                if isinstance(content, dict) and content.get("image_id")
            }
            expanded = []
            for asset in assets:
                item = dict(asset or {})
                content = content_by_id.get(str(item.get("image_id") or ""))
                if content:
                    for key in ("description", "table_markdown", "uncertain", "analysis_error"):
                        item[key] = content.get(key, "" if key != "uncertain" else [])
                expanded.append(item)
            return expanded
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as error:
            raise RuntimeError(f"读取图片资产失败：{path}: {error}") from error

    def image_source_fingerprint(self, kb_path: str, scanned, image_indexes=None):
        if image_indexes is None:
            raise ValueError("image source fingerprint requires prebuilt document image indexes")
        fingerprint = {}
        for rel, ap, _mt, _size in scanned:
            if os.path.splitext(rel)[1].lower() not in (".md", ".markdown"):
                continue
            image_index = image_indexes.get(rel)
            if image_index is None:
                continue
            for occurrence in image_index.occurrences:
                image_rel = os.path.normpath(
                    os.path.join(os.path.dirname(rel), occurrence.path)
                ).replace(os.sep, "/")
                image_abs = os.path.realpath(os.path.join(kb_path, image_rel))
                root = os.path.realpath(kb_path)
                if not (image_abs == root or image_abs.startswith(root + os.sep)) or not os.path.isfile(image_abs):
                    continue
                try:
                    stat = os.stat(image_abs)
                    fingerprint[image_rel] = {"mtime": int(stat.st_mtime), "size": stat.st_size}
                except OSError:
                    continue
        return fingerprint

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _display_label(ref_key: str, caption: str, title: str) -> str:
        caption = str(caption or "").strip()
        if caption:
            return caption
        ref_key = str(ref_key or "").strip()
        if ref_key:
            return ref_key
        return str(title or "图片").strip() or "图片"

    def image_records_for_document(
        self, kb, rel, data_id, body, title, log,
        image_jobs=None, image_index=None,
    ):
        """Build one canonical asset record for every image reference in a document.

        ``image_index`` is built once from the complete Markdown document.  The
        builder passes the same index to chunking and asset creation so context
        and chunk ownership cannot drift apart.
        """
        if os.path.splitext(rel)[1].lower() not in (".md", ".markdown"):
            return {"assets": [], "missing": [], "referenced": 0}
        image_index = image_index or self.build_document_index(body)
        occurrences = image_index.occurrences
        if not occurrences:
            return {"assets": [], "missing": [], "referenced": 0}
        try:
            client = self._image_client_fn()
        except Exception as exc:
            client = None
            log(f"  [warn] 图片模块不可用，仅建立基础图片资产：{exc}")
        assets = []
        missing = []
        kb_root = os.path.realpath(kb["path"])
        analysis_meta = self._image_meta_fn()
        analysis_enabled = client is not None and getattr(client, "enabled", lambda: False)()
        local_jobs = image_jobs if image_jobs is not None else {}
        for occurrence in occurrences:
            image_rel = os.path.normpath(
                os.path.join(os.path.dirname(rel), occurrence.path)
            ).replace(os.sep, "/")
            image_abs = os.path.realpath(os.path.join(kb["path"], image_rel))
            if not (image_abs == kb_root or image_abs.startswith(kb_root + os.sep)) or not os.path.isfile(image_abs):
                missing.append({
                    "path": image_rel,
                    "alt_text": occurrence.alt,
                    "caption": occurrence.title,
                })
                log(f"  [warn] 图片引用未找到：{rel} -> {image_rel}")
                continue
            try:
                image_sha = self._sha256_file(image_abs)
            except Exception:
                image_sha = hashlib.sha1(
                    f"{rel}:{occurrence.occurrence_id}:{occurrence.path}".encode("utf-8")
                ).hexdigest()
            focus = (
                client.understanding_focus(
                    occurrence.title,
                    occurrence.near_text,
                    occurrence.ref_candidates,
                )
                if client is not None
                else "general"
            )
            if analysis_enabled:
                job = ImageContent(
                    image_sha=image_sha,
                    image_path=image_rel,
                    image_abspath=image_abs,
                    focus=focus,
                    title=occurrence.title,
                    near_text=occurrence.near_text,
                    ref_candidates=list(occurrence.ref_candidates),
                    analysis_meta=analysis_meta,
                )
                existing = local_jobs.get(image_sha)
                if existing is None:
                    job.focus_rank = {"general": 0, "figure": 1, "table": 2}.get(focus, 0)
                    job.contexts = [occurrence.near_text] if occurrence.near_text else []
                    local_jobs[image_sha] = job
                else:
                    current_rank = int(existing.focus_rank)
                    new_rank = {"general": 0, "figure": 1, "table": 2}.get(focus, 0)
                    if new_rank > current_rank:
                        existing.focus = focus
                        existing.title = occurrence.title
                        existing.near_text = occurrence.near_text
                        existing.ref_candidates = list(occurrence.ref_candidates)
                        existing.focus_rank = new_rank
                    else:
                        merged = list(existing.ref_candidates or [])
                        for candidate in occurrence.ref_candidates:
                            if candidate not in merged:
                                merged.append(candidate)
                        existing.ref_candidates = merged
                    contexts = existing.contexts
                    if occurrence.near_text and occurrence.near_text not in contexts and len(contexts) < 3:
                        contexts.append(occurrence.near_text)
                        existing.near_text = "\n".join(contexts)[:1200]
            ref_sig = f"occ{occurrence.occurrence_id:06d}"
            image_data_id = f"{data_id}::image::{image_sha[:16]}::{ref_sig}"
            ref_key = occurrence.ref_key
            display_label = self._display_label(ref_key, occurrence.title, title)
            asset = {
                "kind": "image",
                "image_id": image_sha,
                "occurrence_id": occurrence.occurrence_id,
                "data_id": image_data_id,
                "chunk_index": 0,
                "source_data_id": data_id,
                "source_chunk_index": int(occurrence.chunk_index),
                "title": occurrence.title,
                "caption": occurrence.title if occurrence.title.lower() != "image" else "",
                "display_label": display_label,
                "source_file_name": title,
                "source_ref": f"{kb['id']}/{rel}",
                "file_name": rel,
                "image_path": image_rel,
                "image_abspath": image_abs,
                "alt_text": occurrence.alt,
                "section": occurrence.section,
                "understanding_focus": focus,
                "ref_key": ref_key,
                "near_text": occurrence.near_text,
                "related_text": occurrence.related_text,
                "related_text_refs": occurrence.related_text_refs,
                "description": "",
                "table_markdown": "",
                "uncertain": [],
                "analysis_error": "",
            }
            asset["body"] = self.asset_body(asset)
            assets.append(asset)
        if image_jobs is None and local_jobs:
            analyses = self.analyze_image_jobs(kb, local_jobs, log)
            for asset in assets:
                self.apply_image_analysis(asset, analyses.get(asset.get("image_id")))
        return {"assets": assets, "missing": missing, "referenced": len(occurrences)}

    @staticmethod
    def analysis_output_chars(analysis) -> int:
        if not isinstance(analysis, dict):
            return 0
        return sum(len(str(analysis.get(key) or "")) for key in ("description", "table_markdown", "ref_key"))

    @staticmethod
    def cached_analysis_model(cached, result, analysis_meta) -> str:
        if isinstance(result, dict) and result.get("model"):
            return str(result.get("model") or "")
        return str(analysis_meta.get("model") or "")

    def analyze_image_job(self, kb_path: str, job: ImageContent):
        delta = {
            "calls": 0,
            "cached": 0,
            "failed": 0,
            "input_images": 0,
            "input_image_bytes": 0,
            "input_text_chars": 0,
            "output_chars": 0,
            "models": {},
            "cached_models": {},
        }
        image_sha = job.image_sha
        analysis_meta = job.analysis_meta
        focus = str(job.focus or "general")
        cached = self.load_cached_analysis(kb_path, image_sha, analysis_meta, focus)
        if cached:
            delta["cached"] += 1
            result = cached.get("result", cached)
            usage = cached.get("usage") if isinstance(cached, dict) else None
            model = self.cached_analysis_model(cached, result, analysis_meta) if isinstance(cached, dict) else ""
            delta["cached_models"] = self._model_usage_delta_fn(model, usage, self.analysis_output_chars(result))
            return result, delta
        if os.environ.get("GA_KB_IMAGE_ANALYSIS_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
            return {"error": "image analysis cache missing", "uncertain": ["image analysis cache missing"]}, delta
        try:
            client = self._image_client_fn()
            image_abs = job.image_abspath
            image_size = os.path.getsize(image_abs)
            delta["calls"] += 1
            delta["input_images"] += 1
            delta["input_image_bytes"] += image_size
            delta["input_text_chars"] += len(job.title or "") + len(job.near_text or "")
            analysis = client.analyze_image(
                image_abs,
                focus=focus,
                title=job.title or "",
                near_text=job.near_text or "",
                ref_candidates=job.ref_candidates or [],
            )
            usage = analysis.pop("_usage", None)
            request_id = analysis.pop("_request_id", None)
            output_chars = self.analysis_output_chars(analysis)
            delta["output_chars"] += output_chars
            delta["models"] = self._model_usage_delta_fn(str(analysis.get("model") or ""), usage, output_chars)
            # S1: a failed parse/analysis (error-marked) must NOT be cached —
            # otherwise garbage freezes into the permanent VLM cache and is
            # never retried.  The API call still happened, so usage above is
            # kept; here we just flag it failed and skip persisting.
            if analysis.get("error"):
                delta["calls"] -= 1
                delta["failed"] += 1
                return analysis, delta
            self.save_cached_analysis(kb_path, image_sha, analysis_meta, {
                "image_sha256": image_sha,
                "image_path": job.image_path or "",
                "focus": focus,
                "analysis": analysis_meta,
                "usage": usage,
                "request_id": request_id,
                "result": analysis,
            }, focus)
            return analysis, delta
        except Exception as exc:
            delta["failed"] += 1
            return {"error": str(exc), "uncertain": [str(exc)]}, delta

    def apply_image_analysis(self, asset: Dict[str, Any], analysis):
        analysis = analysis or {}
        asset["description"] = analysis.get("description", "")
        asset["table_markdown"] = analysis.get("table_markdown", "")
        asset["uncertain"] = analysis.get("uncertain", [])
        asset["analysis_error"] = analysis.get("error", "")
        # S2: the occurrence's own caption-derived ref_key (set per
        # occurrence at ingestion, from the caption next to THIS
        # placement) is authoritative for the exact-image-ref channel.
        # The VLM sees only the shared image content (one analysis per
        # image_sha) and must not overwrite a captioned occurrence's
        # figure/table number — otherwise a figure reused under two
        # numbers gets both occurrences pinned to whatever the VLM
        # guessed.  VLM ref_key is therefore a fallback only.
        asset["ref_key"] = asset.get("ref_key", "") or self.local_ref_key(analysis.get("ref_key") or "")
        asset["display_label"] = self._display_label(
            asset.get("ref_key", ""), asset.get("caption", ""), asset.get("title", "")
        )
        asset["body"] = self.asset_body(asset)
        return asset

    def analyze_image_jobs(self, kb, image_jobs: dict[str, ImageContent], log):
        if not image_jobs:
            return {}
        jobs = list(image_jobs.values())
        try:
            client = self._image_client_fn()
            enabled = getattr(client, "enabled", lambda: False)()
        except Exception:
            enabled = False
        if not enabled:
            return {}
        workers = max(1, int(os.environ.get("GA_KB_IMAGE_CONCURRENCY", str(self._concurrency))))
        workers = min(workers, len(jobs))
        log(f"  图片分析任务 {len(jobs)} 个，并发 {workers}...")
        results = {}
        done = 0
        if workers <= 1:
            for job in jobs:
                analysis, delta = self.analyze_image_job(kb["path"], job)
                self._merge_usage_fn(delta)
                results[job.image_sha] = analysis
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    log(f"  图片分析进度 {done}/{len(jobs)}")
            return results
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.analyze_image_job, kb["path"], job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    analysis, delta = future.result()
                except Exception as exc:
                    analysis, delta = {"error": str(exc), "uncertain": [str(exc)]}, {"failed": 1}
                self._merge_usage_fn(delta)
                results[job.image_sha] = analysis
                done += 1
                if analysis.get("error"):
                    log(f"  [warn] 图片分析失败 {job.image_path}: {analysis.get('error')}")
                if done % 50 == 0 or done == len(jobs):
                    log(f"  图片分析进度 {done}/{len(jobs)}")
        return results
