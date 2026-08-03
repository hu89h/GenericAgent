"""Markdown image assets, contextual metadata, caching, and VLM analysis."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from urllib.parse import unquote

from .cancellation import KnowledgeBaseCancelled, check_cancelled


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK_RE = re.compile(
    r"(?<!!)\[([^\]\r\n]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)"
)
_IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tif", ".tiff"
)
_REF_CANDIDATE_RE = re.compile(
    r"(?:图|表)\s*[0-9０-９]{1,3}(?:\s*[-－–—.．·]\s*[0-9０-９]{1,3}){0,3}"
)
_SOURCE_LINE_RE = re.compile(r"^(?:资料)?来源\b|^source\b", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")

IMAGE_CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
IMAGE_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024


def cleanup_image_cache(
    cache_dir: str,
    *,
    max_age_seconds: int = IMAGE_CACHE_MAX_AGE_SECONDS,
    max_bytes: int = IMAGE_CACHE_MAX_BYTES,
) -> dict[str, int]:
    """Remove partial, expired, and over-quota VLM cache files."""
    root = os.path.abspath(str(cache_dir or ""))
    if not os.path.isdir(root):
        return {"removed": 0, "bytes": 0}
    now = time.time()
    removed = 0
    removed_bytes = 0
    entries: list[tuple[float, int, str]] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            continue
        if not name.lower().endswith(".json"):
            try:
                removed_bytes += os.path.getsize(path)
            except OSError:
                pass
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if now - stat.st_mtime > max(0, int(max_age_seconds)):
            try:
                os.remove(path)
                removed += 1
                removed_bytes += stat.st_size
            except OSError:
                pass
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _mtime, size, _path in entries)
    for _mtime, size, path in sorted(entries):
        if total <= max(0, int(max_bytes)):
            break
        try:
            os.remove(path)
            total -= size
            removed += 1
            removed_bytes += size
        except OSError:
            pass
    return {"removed": removed, "bytes": removed_bytes}


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
    origins: list[dict[str, str]] = field(default_factory=list)


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
        # Sorted spans for binary-search interval queries (see
        # occurrence_ids_between).  Filled by _build once occurrences exist.
        self._occ_starts: list[int] = []
        self._occ_ends: list[int] = []
        self._occ_monotone_ends = True
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
        """Occurrence ids whose [start, end) span overlaps [start, end).

        Occurrences are non-overlapping and sorted by both start and end
        (image markers cannot nest), so the overlapping set is a contiguous
        index range found with two binary searches instead of an O(n) scan
        per call — this runs 4× per packed block during ingestion.
        """
        if not self._occ_monotone_ends:
            return [
                occurrence.occurrence_id
                for occurrence in self.occurrences
                if occurrence.start < end and occurrence.end > start
            ]
        # start < end  → index < bisect_left(starts, end)
        hi = bisect_left(self._occ_starts, end)
        # end > start  → index >= bisect_right(ends, start)
        lo = bisect_right(self._occ_ends, start)
        if lo >= hi:
            return []
        return [self.occurrences[i].occurrence_id for i in range(lo, hi)]

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

        self._index_occurrence_spans()
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

    def _index_occurrence_spans(self) -> None:
        """Cache parallel start/end arrays for binary-search interval queries.

        ``_MD_IMAGE_RE.finditer`` yields non-overlapping matches in document
        order, so ``occurrences`` is already sorted by both ``start`` and
        ``end``.  We verify monotone ends and fall back to a linear scan if
        that assumption is ever violated (defensive; keeps correctness).
        """
        self._occ_starts = [occ.start for occ in self.occurrences]
        self._occ_ends = [occ.end for occ in self.occurrences]
        monotone = all(
            self._occ_starts[i] <= self._occ_starts[i + 1]
            and self._occ_ends[i] <= self._occ_ends[i + 1]
            for i in range(len(self.occurrences) - 1)
        )
        self._occ_monotone_ends = monotone

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
        usage_tracker,
        concurrency: int = 1,
    ) -> None:
        from .providers import vision

        self._image_client = vision
        self._usage_tracker = usage_tracker
        self._concurrency = max(1, int(concurrency))

    @staticmethod
    def image_cache_dir(kb_path: str) -> str:
        return os.path.join(kb_path, ".kb_index", "image_cache")

    @classmethod
    def _cache_base(cls, kb: dict) -> str:
        return str(kb.get("image_cache_path") or kb.get("path") or "")

    def build_document_index(self, body: str) -> DocumentImageIndex:
        return DocumentImageIndex.build(self, body)

    def scan_image_refs(self, body: str):
        image_matches = list(_MD_IMAGE_RE.finditer(body or ""))
        out = []
        image_spans = [(match.start(), match.end()) for match in image_matches]
        for match in image_matches:
            raw = html.unescape(unquote((match.group(2) or "").strip()))
            if not raw or re.match(r"^[a-z][a-z0-9+.-]*:", raw, re.I):
                continue
            out.append({"alt": (match.group(1) or "").strip(), "path": raw, "start": match.start(), "end": match.end()})
        seen = {(row["start"], row["end"], row["path"]) for row in out}
        for match in _MD_LINK_RE.finditer(body or ""):
            if any(
                match.start() < end and match.end() > start
                for start, end in image_spans
            ):
                continue
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

    @staticmethod
    def analysis_context_key(title: str = "", near_text: str = "", ref_candidates=None) -> str:
        """Digest the prompt-shaping context of one VLM analysis (M1).

        The cached analysis is a function of the image bytes *and* the
        contextual text we feed the model (title / near_text /
        ref_candidates).  Keying the cache on ``image_sha`` alone means
        editing the surrounding Markdown returns a stale analysis that was
        produced from the old context.  Folding a digest of the context
        into the filename makes such an edit miss the cache and re-analyse,
        while an unchanged context still hits.
        """
        refs = ref_candidates or []
        if isinstance(refs, str):
            refs = [refs]
        payload = "\x1f".join(
            (
                str(title or ""),
                str(near_text or ""),
                "\x1e".join(str(item) for item in refs),
            )
        )
        return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:12]

    def analysis_cache_path(
        self, kb_path: str, image_sha: str, analysis_meta, focus: str = "general", context_key: str = ""
    ) -> str:
        version = analysis_meta.get("prompt_version", 1)
        preprocess_version = analysis_meta.get("preprocess_version", 1)
        protocol = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(analysis_meta.get("protocol") or "openai"))
        model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(analysis_meta.get("model") or "image"))
        focus_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(focus or "general"))
        ctx = re.sub(r"[^A-Za-z0-9]+", "", str(context_key or ""))[:12]
        ctx_part = f".c{ctx}" if ctx else ""
        return os.path.join(
            self.image_cache_dir(kb_path),
            f"{image_sha}.v{version}.p{preprocess_version}.{protocol}.{model}.{focus_part}{ctx_part}.json",
        )

    def load_cached_analysis(self, kb_path: str, image_sha: str, analysis_meta, focus: str = "general", context_key: str = ""):
        try:
            with open(self.analysis_cache_path(kb_path, image_sha, analysis_meta, focus, context_key), encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def save_cached_analysis(self, kb_path: str, image_sha: str, analysis_meta, payload, focus: str = "general", context_key: str = "") -> None:
        os.makedirs(self.image_cache_dir(kb_path), exist_ok=True)
        path = self.analysis_cache_path(kb_path, image_sha, analysis_meta, focus, context_key)
        temp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp, path)

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
        # title 常常是 HTML alt 属性，无图注时会是占位符 "image"——不能当标签用
        title = str(title or "").strip()
        if title and title.lower() != "image":
            return title
        return "图片"

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
            client = self._image_client
        except Exception as exc:
            client = None
            log(f"  [warn] 图片模块不可用，仅建立基础图片资产：{exc}")
        assets = []
        missing = []
        kb_root = os.path.realpath(kb["path"])
        analysis_meta = self._image_client.analysis_meta()
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
                    origins=[{"key": rel, "name": title}],
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
                    if not any(origin.get("key") == rel for origin in existing.origins):
                        existing.origins.append({"key": rel, "name": title})
            ref_sig = f"occ{occurrence.occurrence_id:06d}"
            # Occurrence-level data_id layered on the document-level id (see
            # build.py). The "::image::" marker lets callers tell the two
            # layers apart on the shared zvec primary-key column; the plain
            # document-level id is kept as source_data_id below.
            image_data_id = f"{data_id}::image::{image_sha[:16]}::{ref_sig}"
            ref_key = occurrence.ref_key
            # Sanitize the caption once (drop the literal "image" alt-text) and
            # reuse it for display_label — otherwise, when VLM analysis is
            # disabled, apply_image_analysis never re-derives display_label and
            # the raw "image" alt-text leaks through as the label.
            caption = occurrence.title if occurrence.title.lower() != "image" else ""
            display_label = self._display_label(ref_key, caption, title)
            asset = {
                "kind": "image",
                "image_id": image_sha,
                "data_id": image_data_id,
                "chunk_index": 0,
                "source_data_id": data_id,
                "source_chunk_index": int(occurrence.chunk_index),
                "title": occurrence.title,
                "caption": caption,
                "display_label": display_label,
                "source_file_name": title,
                "file_name": rel,
                "image_path": image_rel,
                "image_abspath": image_abs,
                # ``section`` is not a zvec column; it survives only long enough
                # to feed asset_body() → body → the embedding, below.
                "section": occurrence.section,
                "ref_key": ref_key,
                "near_text": occurrence.near_text,
                "related_text": occurrence.related_text,
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

    @staticmethod
    def _usage_tokens(usage: Any) -> tuple[int, int]:
        """Read (prompt, completion) tokens from a provider ``usage`` block.

        VLM providers report either ``prompt_tokens``/``completion_tokens``
        (OpenAI style) or ``input_tokens``/``output_tokens``; be tolerant.
        """
        if not isinstance(usage, dict):
            return 0, 0
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return prompt, completion

    def analyze_image_job(
        self,
        kb_path: str,
        job: ImageContent,
        *,
        cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ):
        delta = {"calls": 0, "cached": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
        image_sha = job.image_sha
        analysis_meta = job.analysis_meta
        focus = str(job.focus or "general")
        # M1: the cache key must include the prompt-shaping context so that
        # editing the surrounding Markdown re-analyses instead of returning
        # an analysis produced from the stale context.
        context_key = self.analysis_context_key(
            title=job.title or "",
            near_text=job.near_text or "",
            ref_candidates=job.ref_candidates or [],
        )
        cached = self.load_cached_analysis(kb_path, image_sha, analysis_meta, focus, context_key)
        if cached:
            delta["cached"] += 1
            if callable(on_progress):
                on_progress({"event": "cache_hit"})
            result = cached.get("result", cached)
            return result, delta
        if os.environ.get("GA_KB_IMAGE_ANALYSIS_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
            return {"error": "image analysis cache missing", "uncertain": ["image analysis cache missing"]}, delta
        try:
            client = self._image_client
            image_abs = job.image_abspath
            delta["calls"] += 1
            analysis_kwargs = {
                "focus": focus,
                "title": job.title or "",
                "near_text": job.near_text or "",
                "ref_candidates": job.ref_candidates or [],
            }
            if callable(cancelled):
                analysis_kwargs["cancelled"] = cancelled
            if callable(on_progress):
                on_progress({"event": "request_started"})
                analysis_kwargs["on_progress"] = on_progress
            analysis = client.analyze_image(image_abs, **analysis_kwargs)
            # Do not persist a response that completed after the user asked to
            # stop.  The surrounding pipeline will remove staging, but this
            # guard also keeps cancellation from populating the image cache.
            check_cancelled(cancelled)
            usage = analysis.pop("_usage", None)
            request_id = analysis.pop("_request_id", None)
            prompt_tokens, completion_tokens = self._usage_tokens(usage)
            delta["prompt_tokens"] += prompt_tokens
            delta["completion_tokens"] += completion_tokens
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
            }, focus, context_key)
            return analysis, delta
        except KnowledgeBaseCancelled:
            raise
        except Exception as exc:
            delta["failed"] += 1
            return {"error": str(exc), "uncertain": [str(exc)]}, delta

    def apply_image_analysis(self, asset: Dict[str, Any], analysis):
        analysis = analysis or {}
        asset["description"] = analysis.get("description", "")
        asset["table_markdown"] = analysis.get("table_markdown", "")
        asset["uncertain"] = analysis.get("uncertain", [])
        # A model that explicitly rejects image input is a configuration
        # warning, not a failed image.  Keep the canonical image record with
        # its caption/context/source fields so text-only models still provide
        # useful image references.  Ordinary per-image errors remain failures
        # and are handled by RecordBuilder as before.
        asset["analysis_warning"] = analysis.get("analysis_warning", "")
        asset["analysis_error"] = "" if analysis.get("vision_skipped") else analysis.get("error", "")
        if asset["analysis_error"] and analysis.get("finish_reason"):
            asset["analysis_error"] += (
                f" (finish_reason={analysis.get('finish_reason')})"
            )
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

    def analyze_image_jobs(
        self,
        kb,
        image_jobs: dict[str, ImageContent],
        log,
        progress: Callable[[dict], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        check_cancelled(cancelled)
        if not image_jobs:
            return {}
        jobs = list(image_jobs.values())
        try:
            client = self._image_client
            enabled = getattr(client, "enabled", lambda: False)()
        except Exception:
            enabled = False
        if not enabled:
            return {}
        workers = max(1, int(os.environ.get("GA_KB_IMAGE_CONCURRENCY", str(self._concurrency))))
        workers = min(workers, len(jobs))
        log(f"  图片分析任务 {len(jobs)} 个，并发 {workers}...")
        activity_lock = threading.Lock()
        job_activity: dict[str, dict[str, Any]] = {}
        done = 0
        cached_count = 0
        document_progress: dict[str, dict[str, Any]] = {}
        for job in jobs:
            for origin in job.origins:
                key = str(origin.get("key") or "")
                if not key:
                    continue
                item = document_progress.setdefault(key, {
                    "key": key,
                    "name": str(origin.get("name") or os.path.basename(key)),
                    "completed": 0,
                    "total": 0,
                })
                item["total"] += 1

        def progress_snapshot() -> list[dict[str, Any]]:
            with activity_lock:
                return [dict(item) for item in document_progress.values()]

        def mark_document_progress(job: ImageContent) -> None:
            with activity_lock:
                for origin in job.origins:
                    item = document_progress.get(str(origin.get("key") or ""))
                    if item is not None:
                        item["completed"] += 1

        def job_name(job: ImageContent) -> str:
            for origin in job.origins:
                name = str(origin.get("name") or "").strip()
                if name:
                    return name
            return os.path.basename(str(job.image_path or ""))

        def mark_job(job: ImageContent, state: str, payload: dict | None = None) -> None:
            now = time.monotonic()
            with activity_lock:
                item = job_activity.setdefault(job.image_sha, {
                    "name": job_name(job),
                    "state": "queued",
                    "started_at": now,
                    "attempt": 0,
                    "attempts": 0,
                    "reason": "",
                })
                item["state"] = state
                if payload:
                    for key in ("attempt", "attempts", "reason", "delay", "timeout"):
                        if key in payload:
                            item[key] = payload[key]
                if state in {"cached", "completed", "failed", "skipped"}:
                    item["finished_at"] = now

        def activity_snapshot(completed: int) -> dict[str, Any]:
            now = time.monotonic()
            with activity_lock:
                active = []
                retrying = 0
                waiting = 0
                for item in job_activity.values():
                    state = str(item.get("state") or "")
                    if state == "retrying":
                        retrying += 1
                    if state == "rate_limited":
                        waiting += 1
                    if state not in {"running", "retrying", "rate_limited"}:
                        continue
                    active_item = {
                        "name": str(item.get("name") or ""),
                        "state": state,
                        "attempt": max(0, int(item.get("attempt") or 0)),
                        "attempts": max(0, int(item.get("attempts") or 0)),
                        "elapsed": max(0, int(now - float(item.get("started_at") or now))),
                    }
                    if item.get("reason"):
                        active_item["reason"] = str(item["reason"])
                    if item.get("delay") is not None:
                        active_item["delay"] = max(0, float(item["delay"]))
                    active.append(active_item)
                active.sort(key=lambda value: (-int(value.get("elapsed") or 0), value.get("name") or ""))
                return {
                    "completed": max(0, int(completed)),
                    "total": len(jobs),
                    "cached": max(0, int(cached_count)),
                    "active": len(active),
                    "retrying": retrying,
                    "waiting": waiting,
                    "current": active[0].get("name", "") if active else "",
                    "items": active[:3],
                }

        def emit_progress(job: ImageContent | None, completed: int) -> None:
            if not callable(progress):
                return
            names = [
                str(origin.get("name") or "")
                for origin in (job.origins if job is not None else [])
                if origin.get("name")
            ]
            activity = activity_snapshot(completed)
            progress({
                "phase": "image_analysis",
                "current": names[0] if names else activity.get("current", ""),
                "analysis_completed": completed,
                "analysis_total": len(jobs),
                "image_documents": progress_snapshot(),
                "image_activity": activity,
            })

        def handle_job_progress(job: ImageContent, payload: dict) -> None:
            if not isinstance(payload, dict):
                return
            event = str(payload.get("event") or "")
            state = {
                "cache_hit": "cached",
                "request_started": "running",
                "attempt_started": "running",
                "retry_scheduled": "retrying",
                "rate_limit_wait": "rate_limited",
                "deadline_exceeded": "failed",
            }.get(event)
            if state:
                mark_job(job, state, payload)
            if callable(progress):
                emit_progress(job, done)

        def run_image_job(job: ImageContent):
            cache_base = self._cache_base(kb)
            mark_job(job, "running")
            on_progress = lambda payload: handle_job_progress(job, payload)
            if callable(cancelled):
                return self.analyze_image_job(
                    cache_base, job, cancelled=cancelled, on_progress=on_progress
                )
            return self.analyze_image_job(cache_base, job, on_progress=on_progress)

        def finish_job(job: ImageContent, analysis: dict, delta: dict) -> None:
            nonlocal cached_count
            analysis = analysis or {}
            delta = delta or {}
            with activity_lock:
                cached_count += max(0, int(delta.get("cached") or 0))
            mark_job(job, "failed" if analysis.get("error") else (
                "cached" if delta.get("cached") else "completed"
            ))

        if callable(progress):
            emit_progress(None, 0)
        results = {}
        heartbeat_stop = threading.Event()
        heartbeat_thread = None

        def heartbeat() -> None:
            while not heartbeat_stop.wait(2.0):
                if callable(cancelled) and cancelled():
                    return
                emit_progress(None, done)

        if callable(progress):
            heartbeat_thread = threading.Thread(
                target=heartbeat,
                name="ga-kb-image-progress",
                daemon=True,
            )
            heartbeat_thread.start()

        def stop_heartbeat() -> None:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.5)

        # Use the first image as a capability probe before opening the worker
        # pool.  A text-only model must not receive dozens of doomed image
        # requests when one explicit rejection is enough to establish the
        # capability.  This is intentionally based on the provider's error
        # classification, never on a model-name heuristic.
        first_job = jobs[0]
        try:
            check_cancelled(cancelled)
        except KnowledgeBaseCancelled:
            stop_heartbeat()
            raise
        try:
            first_analysis, first_delta = run_image_job(first_job)
        except KnowledgeBaseCancelled:
            stop_heartbeat()
            raise
        try:
            check_cancelled(cancelled)
        except KnowledgeBaseCancelled:
            stop_heartbeat()
            raise
        self._usage_tracker.merge_image_analysis(first_delta)
        first_analysis = first_analysis or {}
        finish_job(first_job, first_analysis, first_delta)
        try:
            from .providers import vision as vision_provider
            unsupported = vision_provider.is_vision_unsupported_error(first_analysis)
        except Exception:
            unsupported = False
        if unsupported:
            warning = (
                "当前模型不支持图片输入，已跳过图片 VLM 分析；"
                "图片的图号、图注和来源仍已保留。"
            )
            fallback = {
                "vision_skipped": True,
                "analysis_warning": warning,
                "uncertain": [warning],
            }
            log(
                "  [warn] 当前模型不支持图片输入，已跳过剩余 "
                f"{len(jobs)} 个图片分析请求，保留基础图片引用。"
            )
            for job in jobs:
                try:
                    check_cancelled(cancelled)
                except KnowledgeBaseCancelled:
                    stop_heartbeat()
                    raise
                results[job.image_sha] = dict(fallback)
                if job.image_sha != first_job.image_sha:
                    mark_job(job, "skipped")
                done += 1
                mark_document_progress(job)
                emit_progress(job, done)
            stop_heartbeat()
            return results

        results[first_job.image_sha] = first_analysis
        done = 1
        mark_document_progress(first_job)
        emit_progress(first_job, done)
        remaining_jobs = jobs[1:]
        if not remaining_jobs:
            stop_heartbeat()
            return results

        if workers <= 1:
            for job in remaining_jobs:
                try:
                    check_cancelled(cancelled)
                except KnowledgeBaseCancelled:
                    stop_heartbeat()
                    raise
                try:
                    analysis, delta = run_image_job(job)
                except KnowledgeBaseCancelled:
                    stop_heartbeat()
                    raise
                try:
                    check_cancelled(cancelled)
                except KnowledgeBaseCancelled:
                    stop_heartbeat()
                    raise
                self._usage_tracker.merge_image_analysis(delta)
                analysis = analysis or {}
                finish_job(job, analysis, delta)
                results[job.image_sha] = analysis
                done += 1
                mark_document_progress(job)
                emit_progress(job, done)
                if done % 50 == 0 or done == len(jobs):
                    log(f"  图片分析进度 {done}/{len(jobs)}")
            stop_heartbeat()
            return results
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {}
        try:
            futures = {
                executor.submit(run_image_job, job): job
                for job in remaining_jobs
            }
            for future in as_completed(futures):
                check_cancelled(cancelled)
                job = futures[future]
                try:
                    analysis, delta = future.result()
                except KnowledgeBaseCancelled:
                    raise
                except Exception as exc:
                    analysis, delta = {"error": str(exc), "uncertain": [str(exc)]}, {"failed": 1}
                check_cancelled(cancelled)
                self._usage_tracker.merge_image_analysis(delta)
                analysis = analysis or {}
                finish_job(job, analysis, delta)
                results[job.image_sha] = analysis
                done += 1
                mark_document_progress(job)
                emit_progress(job, done)
                if analysis.get("error"):
                    log(f"  [warn] 图片分析失败 {job.image_path}: {analysis.get('error')}")
                if done % 50 == 0 or done == len(jobs):
                    log(f"  图片分析进度 {done}/{len(jobs)}")
        except KnowledgeBaseCancelled:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            stop_heartbeat()
            raise
        else:
            executor.shutdown(wait=True)
            stop_heartbeat()
        return results
