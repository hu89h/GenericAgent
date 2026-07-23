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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict
from urllib.parse import unquote


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
_REF_CANDIDATE_RE = re.compile(
    r"(?:图|表)\s*[0-9０-９]{1,3}(?:\s*[-－–—.．·]\s*[0-9０-９]{1,3}){0,3}"
)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


class ImageAssetProcessor:
    def __init__(
        self,
        *,
        image_client_fn: Callable[[], Any],
        image_meta_fn: Callable[[], Dict[str, Any]],
        image_cache_dir_fn: Callable[[str], str],
        image_assets_path_fn: Callable[[str], str],
        index_dir_fn: Callable[[str], str],
        read_text_fn: Callable[[str], str],
        merge_usage_fn: Callable[[Dict[str, Any]], None],
        model_usage_delta_fn: Callable[[str, Dict[str, Any] | None, int], Dict[str, Any]],
        concurrency: int = 1,
    ) -> None:
        self._image_client_fn = image_client_fn
        self._image_meta_fn = image_meta_fn
        self._image_cache_dir_fn = image_cache_dir_fn
        self._image_assets_path_fn = image_assets_path_fn
        self._index_dir_fn = index_dir_fn
        self._read_text_fn = read_text_fn
        self._merge_usage_fn = merge_usage_fn
        self._model_usage_delta_fn = model_usage_delta_fn
        self._concurrency = max(1, int(concurrency))

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

    def context_heading(self, text: str, pos: int) -> str:
        before = (text or "")[:max(0, pos)]
        for line in reversed(before.splitlines()[-80:]):
            value = line.strip()
            if value.startswith("#"):
                return value.lstrip("#").strip()
        return ""

    def image_title(self, alt: str, body: str, pos: int) -> str:
        alt = (alt or "").strip()
        if alt and alt.lower() != "image":
            return alt
        after = self.image_after_title_line(body, pos)
        if after and re.search(r"(图|表)\s*\d|图[一二三四五六七八九十]|表[一二三四五六七八九十]", after):
            return after
        return alt or "image"

    def near_text(self, body: str, pos: int, window: int = 300) -> str:
        text = body or ""
        start = max(0, pos - window)
        end = min(len(text), pos + window)
        near = _MD_IMAGE_RE.sub(lambda match: f"[图片:{(match.group(1) or 'image').strip()}]", text[start:end])
        return re.sub(r"\s+", " ", near).strip()

    @staticmethod
    def image_line_index(body: str, pos: int):
        lines = (body or "").splitlines()
        offset = 0
        for index, line in enumerate(lines):
            next_offset = offset + len(line) + 1
            if offset <= pos < next_offset:
                return index, lines
            offset = next_offset
        return 0, lines

    def image_after_title_line(self, body: str, pos: int, max_scan: int = 6) -> str:
        image_line, lines = self.image_line_index(body, pos)
        scanned = 0
        for line in lines[image_line + 1:]:
            scanned += 1
            if scanned > max_scan:
                break
            value = line.strip()
            if not value:
                continue
            if _MD_IMAGE_RE.search(value):
                return ""
            return _MD_IMAGE_RE.sub("", value).strip().strip("*").strip()
        return ""

    def image_post_title_context(self, body: str, pos: int) -> str:
        return self.image_after_title_line(body, pos)

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

    def build_related_index(self, text: str):
        candidates = self.extract_ref_candidates(text)
        for ref in self.scan_image_refs(text):
            title = self.image_title(ref.get("alt", ""), text, ref["start"])
            post_title = self.image_post_title_context(text, ref["start"])
            for value in (title, post_title):
                value = str(value or "").strip()
                if value and value not in candidates:
                    candidates.append(value)
        if not candidates:
            return {}
        mapping = {candidate: self.local_ref_key(candidate) for candidate in candidates}
        index = {"__mapping": mapping}
        for paragraph_index, (offset, paragraph) in enumerate(self.paragraphs_with_offsets(text)):
            if _MD_IMAGE_RE.search(paragraph):
                continue
            keys = []
            for candidate in self.extract_ref_candidates(paragraph):
                key = mapping.get(candidate) or self.local_ref_key(candidate)
                if key and key not in keys:
                    keys.append(key)
            for key in keys:
                if self.is_caption_like(paragraph, key):
                    continue
                index.setdefault(key, []).append({
                    "paragraph_index": paragraph_index,
                    "offset": offset,
                    "text": re.sub(r"\s+", " ", paragraph).strip(),
                })
        return index

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
            ("图片路径", "image_path"),
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

    def write_assets(self, kb_path: str, assets) -> None:
        os.makedirs(self._index_dir_fn(kb_path), exist_ok=True)
        payload = {
            "schema_version": 5,
            "built_at": int(time.time()),
            "analysis": self._image_meta_fn(),
            "n_assets": len(assets),
            "assets": assets,
        }
        with open(self._image_assets_path_fn(kb_path), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_assets(self, kb_path: str):
        try:
            with open(self._image_assets_path_fn(kb_path), encoding="utf-8") as handle:
                payload = json.load(handle)
            assets = payload.get("assets") if isinstance(payload, dict) else None
            return assets if isinstance(assets, list) else []
        except Exception:
            return []

    def image_source_fingerprint(self, kb_path: str, scanned):
        fingerprint = {}
        for rel, ap, _mt, _size in scanned:
            if os.path.splitext(rel)[1].lower() not in (".md", ".markdown"):
                continue
            text = self._read_text_fn(ap)
            for ref in self.scan_image_refs(text):
                image_rel = os.path.normpath(os.path.join(os.path.dirname(rel), ref["path"])).replace(os.sep, "/")
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

    def image_records_for_chunk(
        self, kb, rel, ap, data_id, chunk_index, body, title, log,
        image_jobs=None, related_index=None,
    ):
        if os.path.splitext(rel)[1].lower() not in (".md", ".markdown"):
            return []
        refs = self.scan_image_refs(body)
        if not refs:
            return []
        try:
            client = self._image_client_fn()
        except Exception as exc:
            client = None
            log(f"  [warn] 图片模块不可用，仅建立基础图片资产：{exc}")
        assets = []
        kb_root = os.path.realpath(kb["path"])
        analysis_meta = self._image_meta_fn()
        for seq, ref in enumerate(refs):
            image_rel = os.path.normpath(os.path.join(os.path.dirname(rel), ref["path"])).replace(os.sep, "/")
            image_abs = os.path.realpath(os.path.join(kb["path"], image_rel))
            if not (image_abs == kb_root or image_abs.startswith(kb_root + os.sep)) or not os.path.isfile(image_abs):
                continue
            try:
                image_sha = self._sha256_file(image_abs)
            except Exception:
                image_sha = hashlib.sha1(f"{rel}:{chunk_index}:{ref['path']}:{seq}".encode("utf-8")).hexdigest()
            image_title = self.image_title(ref.get("alt", ""), body, ref["start"])
            post_title_context = self.image_post_title_context(body, ref["start"])
            near = self.near_text(body, ref["start"])
            section = self.context_heading(body, ref["start"])
            ref_candidates = self.collect_ref_candidates(image_title, post_title_context, near)
            focus = client.understanding_focus(image_title, near, ref_candidates) if client is not None else "general"
            ref_key = next((self.local_ref_key(candidate) for candidate in ref_candidates if self.local_ref_key(candidate)), "")
            if ref_key:
                related_text, related_refs = self.related_text_for_ref_key(ref_key, related_index or {})
            else:
                related_text, related_refs = self.related_text_for_image(image_title, related_index or {})
            analysis = {}
            if client is not None and getattr(client, "enabled", lambda: False)():
                job = {
                    "image_sha": image_sha,
                    "image_path": image_rel,
                    "image_abspath": image_abs,
                    "focus": focus,
                    "title": image_title,
                    "near_text": near,
                    "ref_candidates": ref_candidates,
                    "analysis_meta": analysis_meta,
                }
                if image_jobs is not None:
                    image_jobs.setdefault(image_sha, job)
                else:
                    analysis, usage_delta = self.analyze_image_job(kb["path"], job)
                    self._merge_usage_fn(usage_delta)
            ref_sig = hashlib.sha1(f"{image_rel}:{seq}".encode("utf-8")).hexdigest()[:8]
            asset = {
                "kind": "image",
                "image_id": image_sha,
                "data_id": f"{data_id}::image::{image_sha[:16]}::{chunk_index}::{seq}-{ref_sig}",
                "chunk_index": 0,
                "parent_data_id": data_id,
                "parent_chunk_index": chunk_index,
                "title": image_title,
                "file_name": rel,
                "image_path": image_rel,
                "image_abspath": image_abs,
                "alt_text": ref.get("alt", ""),
                "section": section,
                "understanding_focus": focus,
                "ref_key": ref_key,
                "near_text": near,
                "related_text": related_text,
                "related_text_refs": related_refs,
                "_related_index": related_index or {},
                "description": analysis.get("description", ""),
                "table_markdown": analysis.get("table_markdown", ""),
                "uncertain": analysis.get("uncertain", []),
                "analysis_error": analysis.get("error", ""),
            }
            asset["body"] = self.asset_body(asset)
            assets.append(asset)
        return assets

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

    def analyze_image_job(self, kb_path: str, job):
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
        image_sha = job["image_sha"]
        analysis_meta = job["analysis_meta"]
        focus = str(job.get("focus") or "general")
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
            image_abs = job["image_abspath"]
            image_size = os.path.getsize(image_abs)
            delta["calls"] += 1
            delta["input_images"] += 1
            delta["input_image_bytes"] += image_size
            delta["input_text_chars"] += len(job.get("title") or "") + len(job.get("near_text") or "")
            analysis = client.analyze_image(
                image_abs,
                focus=focus,
                title=job.get("title") or "",
                near_text=job.get("near_text") or "",
                ref_candidates=job.get("ref_candidates") or [],
            )
            usage = analysis.pop("_usage", None)
            request_id = analysis.pop("_request_id", None)
            output_chars = self.analysis_output_chars(analysis)
            delta["output_chars"] += output_chars
            delta["models"] = self._model_usage_delta_fn(str(analysis.get("model") or ""), usage, output_chars)
            self.save_cached_analysis(kb_path, image_sha, analysis_meta, {
                "image_sha256": image_sha,
                "image_path": job.get("image_path") or "",
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
        asset["ref_key"] = self.local_ref_key(analysis.get("ref_key") or "") or asset.get("ref_key", "")
        related_text, related_refs = self.related_text_for_ref_key(asset["ref_key"], asset.get("_related_index") or {})
        asset["related_text"] = related_text
        asset["related_text_refs"] = related_refs
        asset["body"] = self.asset_body(asset)
        return asset

    def analyze_image_jobs(self, kb, image_jobs, log):
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
                results[job["image_sha"]] = analysis
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
                results[job["image_sha"]] = analysis
                done += 1
                if analysis.get("error"):
                    log(f"  [warn] 图片分析失败 {job.get('image_path')}: {analysis.get('error')}")
                if done % 50 == 0 or done == len(jobs):
                    log(f"  图片分析进度 {done}/{len(jobs)}")
        return results
