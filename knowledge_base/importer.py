"""Unified knowledge-base import pipeline.

The importer keeps the processing contract small: prepare either one source
directory or an explicitly selected set of files. Markdown is copied directly,
while supported non-Markdown files go through MinerU. PDFs larger than the
service limit are rejected before any remote job is submitted; the importer
deliberately does not split or merge PDFs.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

try:
    from .cancellation import KnowledgeBaseCancelled, check_cancelled
    from .documents import read_textfile
    from .providers import mineru
except ImportError:  # pragma: no cover - supports direct CLI execution
    from cancellation import KnowledgeBaseCancelled, check_cancelled
    from documents import read_textfile
    from providers import mineru


MAX_PDF_PAGES = 200
MARKDOWN_EXTS = {".md", ".markdown"}
IMAGE_EXTS = mineru.IMAGE_EXTS
SUPPORTED_EXTS = mineru.SUPPORTED_EXTS
_IMAGE_SUFFIXES = tuple(sorted(IMAGE_EXTS | {".tif", ".tiff"}))
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK_RE = re.compile(
    r"(?<!!)\[([^\]\r\n]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)"
)
_IMAGE_SUFFIX_RE = re.compile(
    r"(?i)(?:"
    + "|".join(
        re.escape(suffix)
        for suffix in sorted(_IMAGE_SUFFIXES, key=len, reverse=True)
    )
    + r")"
)


def _path_is_within(root: Path, path: Path) -> bool:
    try:
        root_value = os.path.realpath(str(root))
        path_value = os.path.realpath(str(path))
        return path_value == root_value or os.path.commonpath((root_value, path_value)) == root_value
    except (OSError, ValueError):
        return False


def _scan_source(source_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(source_root, followlinks=False):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(source_root).as_posix()
            files.append((path, relative))
    files.sort(key=lambda row: row[1].casefold())
    return files


def _safe_component(value: str, limit: int = 48) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", str(value or "")).strip(" ._")
    value = re.sub(r"\s+", " ", value)
    return value[:limit] or "document"


def _compact_asset_name(path_part: str, source_image: Path) -> str:
    suffix = source_image.suffix.lower() or Path(path_part).suffix.lower() or ".bin"
    stem = _safe_component(Path(path_part).stem, 36)
    digest = hashlib.sha256(str(source_image).encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{digest}{suffix}"


def _image_refs(body: str) -> list[tuple[re.Match[str], str]]:
    image_matches = list(_MD_IMAGE_RE.finditer(body or ""))
    matches: list[re.Match[str]] = list(image_matches)
    # Image links without the ! prefix are accepted because MinerU sometimes
    # emits them for a page asset instead of Markdown image syntax.
    image_spans = [(match.start(), match.end()) for match in image_matches]
    for match in _MD_LINK_RE.finditer(body or ""):
        if any(
            match.start() < end and match.end() > start
            for start, end in image_spans
        ):
            continue
        matches.append(match)
    matches.sort(key=lambda match: (match.start(), match.end()))
    return [(match, html.unescape(unquote(match.group(2).strip()))) for match in matches]


def _local_image_path(markdown_path: Path, raw: str) -> tuple[str, Path] | None:
    value = raw.strip().strip("<>")
    if not value or re.match(r"^[a-z][a-z0-9+.-]*:", value, re.I):
        return None
    path_part = value.split("?", 1)[0].split("#", 1)[0]
    if not path_part.lower().endswith(_IMAGE_SUFFIXES):
        return None
    return path_part, (markdown_path.parent / path_part).resolve()


def _canonical_image_reference(markdown_path: Path, image: Path) -> str:
    return quote(
        os.path.relpath(image, markdown_path.parent).replace(os.sep, "/"),
        safe="/-._~",
    )


def _resolve_unique_image_basename(source_root: Path, path_part: str) -> Path | None:
    """Resolve a flattened MinerU asset only when its basename is unique."""
    name = Path(path_part).name.casefold()
    if not name:
        return None
    candidates = sorted(
        {
            path.resolve()
            for path in source_root.rglob("*")
            if path.is_file() and path.name.casefold() == name
        },
        key=lambda path: path.as_posix().casefold(),
    )
    return candidates[0] if len(candidates) == 1 else None


def _split_concatenated_image_target(
    markdown_path: Path,
    source_root: Path,
    raw: str,
) -> list[str] | None:
    """Return local image targets when MinerU joined consecutive links.

    A small number of MinerU Markdown results have two image links joined at
    the image suffix, for example ``first.jpgsecond.assets/second.jpg``.
    Splitting on a suffix alone would corrupt legitimate URLs, so every
    candidate segment must resolve to a real image below the extracted source
    directory. MinerU can also flatten an asset directory while leaving its
    old prefix in Markdown; in that case a unique basename is accepted and
    rewritten to the real relative path. ``None`` means the target is either
    valid as-is or cannot be unambiguously repaired.
    """
    value = raw.strip().strip("<>")
    if not value or "?" in value or "#" in value:
        return None
    if re.match(r"^[a-z][a-z0-9+.-]*:", value, re.I):
        return None

    # Do not reinterpret a legitimate local filename that happens to contain
    # another image suffix in its name.
    decoded_value = html.unescape(unquote(value))
    whole_local = _local_image_path(markdown_path, decoded_value)
    if whole_local is not None:
        _path_part, whole_image = whole_local
        if _path_is_within(source_root, whole_image) and whole_image.is_file():
            return None

    boundaries = sorted({match.end() for match in _IMAGE_SUFFIX_RE.finditer(value)})
    if len(boundaries) < 2:
        return None

    memo: dict[int, list[str] | None] = {}

    def resolve(start: int) -> list[str] | None:
        if start in memo:
            return memo[start]
        for end in boundaries:
            if end <= start:
                continue
            part = value[start:end]
            decoded = html.unescape(unquote(part))
            local = _local_image_path(markdown_path, decoded)
            if local is None:
                continue
            path_part, image = local
            if not _path_is_within(source_root, image) or not image.is_file():
                image = _resolve_unique_image_basename(source_root, path_part)
                if image is None or not _path_is_within(source_root, image):
                    continue
            canonical = _canonical_image_reference(markdown_path, image)
            if end == len(value):
                result = [canonical]
            else:
                rest = resolve(end)
                result = [canonical, *rest] if rest else None
            if result:
                memo[start] = result
                return result
        memo[start] = None
        return None

    parts = resolve(0)
    return parts if parts and len(parts) > 1 else None


def _normalize_mineru_image_links(
    markdown_path: Path,
    source_root: Path,
    body: str,
) -> str:
    """Repair MinerU links that contain multiple local image paths."""
    replacements: list[tuple[int, int, str]] = []
    for match, raw in _image_refs(body):
        parts = _split_concatenated_image_target(markdown_path, source_root, raw)
        if not parts:
            continue
        prefix = body[match.start():match.start(2)]
        suffix = body[match.end(2):match.end()]
        links = [
            prefix + part + (")" if index < len(parts) - 1 else suffix)
            for index, part in enumerate(parts)
        ]
        replacements.append((match.start(), match.end(), "\n".join(links)))

    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return body


def _prepare_markdown(
    markdown_path: Path,
    source_root: Path,
    target_markdown: Path,
    target_root: Path,
    asset_namespace: Path | None = None,
    extra_images: list[Path] | None = None,
) -> tuple[str, int]:
    """Copy referenced images and rewrite their paths for a final Markdown."""
    body = read_textfile(str(markdown_path))
    body = _normalize_mineru_image_links(markdown_path, source_root, body)
    replacements: list[tuple[int, int, str]] = []
    copied: set[Path] = set()
    namespace = asset_namespace or Path(".")
    for match, raw in _image_refs(body):
        local = _local_image_path(markdown_path, raw)
        if local is None:
            continue
        path_part, source_image = local
        if not _path_is_within(source_root, source_image) or not source_image.is_file():
            continue
        target_image = (
            target_markdown.parent / namespace / _compact_asset_name(path_part, source_image)
        ).resolve()
        if not _path_is_within(target_root, target_image):
            continue
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
        copied.add(target_image)
        tail = raw[len(path_part):]
        relative = quote(
            os.path.relpath(target_image, target_markdown.parent).replace(os.sep, "/"),
            safe="/-._~",
        )
        replacements.append((match.start(2), match.end(2), relative + tail))

    for image in extra_images or []:
        image = Path(image).resolve()
        if not image.is_file():
            continue
        target_image = (target_markdown.parent / namespace / f"source{image.suffix.lower()}").resolve()
        if not _path_is_within(target_root, target_image):
            continue
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        copied.add(target_image)
        relative = quote(
            os.path.relpath(target_image, target_markdown.parent).replace(os.sep, "/"),
            safe="/-._~",
        )
        if not re.search(rf"\]\({re.escape(relative)}(?:[?#\s)]|$)", body):
            body = body.rstrip() + f"\n\n![原始图片]({relative})\n"

    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return body, len(copied)


def _write_markdown(
    source_markdown: Path,
    source_root: Path,
    target_markdown: Path,
    target_root: Path,
    *,
    asset_namespace: Path | None = None,
    extra_images: list[Path] | None = None,
) -> tuple[str, int]:
    body, image_count = _prepare_markdown(
        source_markdown,
        source_root,
        target_markdown,
        target_root,
        asset_namespace=asset_namespace,
        extra_images=extra_images,
    )
    target_markdown.parent.mkdir(parents=True, exist_ok=True)
    target_markdown.write_text(body, encoding="utf-8")
    return body, image_count


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            destination = (target_dir / member.filename).resolve()
            if not _path_is_within(target_dir, destination):
                raise ValueError(f"MinerU ZIP 包含非法路径：{member.filename}")
        archive.extractall(target_dir)


def _find_markdown(directory: Path) -> Path | None:
    candidates = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.name.lower() == "full.md"),
        key=lambda path: path.as_posix().casefold(),
    )
    if not candidates:
        candidates = sorted(
            (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".md"),
            key=lambda path: path.as_posix().casefold(),
        )
    return candidates[0] if candidates else None


def _pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF 页数检查需要 pypdf，请先安装项目依赖") from error
    try:
        with path.open("rb") as handle:
            reader = PdfReader(handle, strict=False)
            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                except Exception as error:
                    raise RuntimeError(f"PDF 已加密，无法读取页数：{path.name}") from error
                if not decrypted:
                    raise RuntimeError(f"PDF 已加密，无法读取页数：{path.name}")
            page_count = len(reader.pages)
            if page_count <= 0:
                raise RuntimeError(f"PDF 没有可处理的页面：{path.name}")
            if page_count > MAX_PDF_PAGES:
                raise ValueError(
                    f"PDF 超过 {MAX_PDF_PAGES} 页，暂不支持，请拆分后重新导入：{path.name}"
                )
            return page_count
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"读取 PDF 页数失败：{path.name}：{error}") from error


def _emit(progress: Callable[[dict], None] | None, phase: str, counts: dict[str, int], **extra: Any) -> None:
    if callable(progress):
        progress({
            "phase": phase,
            **counts,
            "document_progress": {
                "completed": max(0, int(counts.get("completed") or 0)),
                "total": max(0, int(counts.get("total") or 0)),
                "failed": max(0, int(counts.get("failed") or 0)),
                "ready": max(0, int(counts.get("ready") or 0)),
            },
            **extra,
        })


class DocumentProcessor:
    """Prepare source documents inside a caller-owned staging directory."""

    @staticmethod
    def _mark_failure(entry: dict, error: Exception | str, stage: str) -> None:
        entry.update(
            status="failed",
            error=str(error),
            stage=stage,
            error_type=type(error).__name__ if isinstance(error, Exception) else "ProcessingError",
        )

    def prepare(
        self,
        source_dir: str,
        *,
        stage_root: str,
        kb_id: str,
        name: str = "",
        progress: Callable[[dict], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        check_cancelled(cancelled)
        source_root = Path(os.path.realpath(os.path.expanduser(str(source_dir or ""))))
        if not source_root.is_dir():
            raise ValueError(f"sourceDir is not a directory: {source_root}")

        stage = Path(stage_root).resolve()
        processed_root = stage / "processed"
        downloads_root = stage / ".mineru_downloads"
        extract_root = stage / ".mineru_extract"
        stage.mkdir(parents=True, exist_ok=True)

        files = _scan_source(source_root)
        entries: list[dict[str, Any]] = []
        entry_by_source: dict[str, dict[str, Any]] = {}
        output_rels: set[str] = set()
        counts = {
            "scanned": len(files), "total": 0, "completed": 0,
            "markdown": 0, "converting": 0, "ready": 0,
            "succeeded": 0, "failed": 0, "ignored": 0, "skipped": 0,
            "assets": 0,
        }

        def refresh_document_counts() -> None:
            documents = [item for item in entries if item["kind"] == "document"]
            counts.update({
                "total": len(documents),
                "ready": sum(item["status"] == "ready" for item in documents),
                "failed": sum(item["status"] == "failed" for item in documents),
                "assets": sum(item["kind"] == "asset" for item in entries),
                "ignored": sum(item["kind"] == "ignored" for item in entries),
            })
            counts["skipped"] = counts["ignored"]
            counts["succeeded"] = counts["ready"]
            counts["completed"] = counts["ready"] + counts["failed"]

        def output_rel_for(source_rel: str) -> str:
            digest = hashlib.sha256(source_rel.encode("utf-8")).hexdigest()[:12]
            stem = _safe_component(Path(source_rel).stem, 48)
            candidate = f"documents/{digest}-{stem}.md"
            if candidate in output_rels:
                candidate = (
                    f"documents/{digest}-"
                    f"{hashlib.sha256(source_rel.encode('utf-8')).hexdigest()[12:20]}.md"
                )
            output_rels.add(candidate)
            return candidate

        try:
            _emit(progress, "scanning", counts)
            source_by_rel = {os.path.normcase(rel): rel for _path, rel in files}
            has_markdown = any(path.suffix.lower() in MARKDOWN_EXTS for path, _rel in files)
            referenced_images: set[str] = set()
            for path, rel in files:
                check_cancelled(cancelled)
                if path.suffix.lower() not in MARKDOWN_EXTS:
                    continue
                try:
                    body = read_textfile(str(path))
                except Exception:
                    continue
                body = _normalize_mineru_image_links(path, source_root, body)
                for _match, raw in _image_refs(body):
                    local = _local_image_path(path, raw)
                    if local is None:
                        continue
                    _path_part, image = local
                    if _path_is_within(source_root, image) and image.is_file():
                        matched = source_by_rel.get(
                            os.path.normcase(image.relative_to(source_root).as_posix())
                        )
                        if matched:
                            referenced_images.add(matched)

            conversion_specs: list[dict[str, Any]] = []
            for path, rel in files:
                check_cancelled(cancelled)
                ext = path.suffix.lower()
                entry = {
                    "source": rel,
                    "name": path.name,
                    "kind": "ignored",
                    "status": "ignored",
                    "processed": [],
                    "error": "",
                    "stage": "",
                    "error_type": "",
                }
                entries.append(entry)
                entry_by_source[rel] = entry
                if ext in MARKDOWN_EXTS:
                    entry["kind"] = "document"
                    try:
                        target_rel = output_rel_for(rel)
                        target = processed_root / target_rel
                        namespace = Path(
                            f"{target.stem}.assets-"
                            f"{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:8]}"
                        )
                        _write_markdown(
                            path, source_root, target, processed_root,
                            asset_namespace=namespace,
                        )
                        entry.update(status="ready", processed=[target_rel])
                    except Exception as error:
                        self._mark_failure(entry, error, "markdown")
                    counts["markdown"] += 1
                    continue
                if ext in IMAGE_EXTS and rel in referenced_images:
                    entry.update(kind="asset", status="asset")
                    continue
                if ext in IMAGE_EXTS and has_markdown:
                    continue
                if ext not in SUPPORTED_EXTS:
                    continue
                entry.update(kind="document", status="queued")
                conversion_specs.append({"source": rel, "path": path, "entry": entry})
                counts["converting"] += 1

            document_entries = [item for item in entries if item["kind"] == "document"]
            refresh_document_counts()
            _emit(progress, "scanned", counts)

            upload_specs: list[dict[str, Any]] = []
            for spec in conversion_specs:
                check_cancelled(cancelled)
                source = spec["source"]
                path = spec["path"]
                entry = spec["entry"]
                if path.suffix.lower() == ".pdf":
                    try:
                        _pdf_page_count(path)
                    except Exception as error:
                        self._mark_failure(entry, error, "pdf_validation")
                        refresh_document_counts()
                        _emit(
                            progress, "processing", counts,
                            source=source, name=path.name, current=source,
                            file_status="failed", error=str(error),
                        )
                        continue
                upload_specs.append({"source": source, "path": path, "relative_path": source})
            if upload_specs:
                _emit(progress, "processing", counts, current="开始提交 MinerU")

            jobs_by_relative: dict[str, dict[str, Any]] = {}
            if upload_specs:
                def on_mineru_update(job: mineru.MinerUFile) -> None:
                    spec = jobs_by_relative.get(job.relative_path)
                    if spec is None:
                        return
                    entry = entry_by_source[spec["source"]]
                    entry.update(status=job.state, error=job.error)
                    refresh_document_counts()
                    _emit(
                        progress, "processing", counts,
                        source=spec["source"], name=Path(spec["source"]).name,
                        current=spec["source"], file_status=entry["status"],
                        error=job.error,
                    )

                jobs_by_relative.update({
                    spec["relative_path"]: spec for spec in upload_specs
                })
                try:
                    jobs = mineru.process_batches(
                        [(spec["path"], spec["relative_path"]) for spec in upload_specs],
                        downloads_root,
                        on_update=on_mineru_update,
                        cancelled=cancelled,
                    )
                except KnowledgeBaseCancelled:
                    raise
                except Exception as error:
                    jobs = []
                    for spec in upload_specs:
                        self._mark_failure(
                            entry_by_source[spec["source"]], error, "mineru"
                        )
                for job in jobs:
                    if job.relative_path in jobs_by_relative:
                        jobs_by_relative[job.relative_path]["job"] = job

                for spec in upload_specs:
                    check_cancelled(cancelled)
                    source = spec["source"]
                    entry = entry_by_source[source]
                    job = jobs_by_relative.get(spec["relative_path"], {}).get("job")
                    if not job or job.state != "downloaded":
                        error = getattr(job, "error", "") or entry.get("error") or "MinerU 处理失败"
                        self._mark_failure(entry, error, "mineru")
                        refresh_document_counts()
                        _emit(
                            progress, "processing", counts,
                            source=source, name=Path(source).name, current=source,
                            file_status="failed", error=error,
                        )
                        continue

                    target_rel = output_rel_for(source)
                    target = processed_root / target_rel
                    output_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
                    asset_root = Path(f"{target.stem}.assets-{output_key}")
                    try:
                        extract_dir = extract_root / output_key
                        _safe_extract_zip(downloads_root / f"{job.data_id}.zip", extract_dir)
                        markdown = _find_markdown(extract_dir)
                        if markdown is None:
                            raise ValueError("MinerU 结果中未找到 Markdown 文档")
                        extra_images = (
                            [Path(spec["path"])]
                            if Path(source).suffix.lower() in IMAGE_EXTS else []
                        )
                        body, _ = _prepare_markdown(
                            markdown, extract_dir, target, processed_root,
                            asset_namespace=asset_root, extra_images=extra_images,
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(body.strip() + "\n", encoding="utf-8")
                        entry.update(
                            status="ready", error="", processed=[target_rel],
                            stage="", error_type="",
                        )
                        refresh_document_counts()
                        _emit(
                            progress, "processing", counts,
                            source=source, name=Path(source).name, current=source,
                            file_status="ready",
                        )
                    except KnowledgeBaseCancelled:
                        raise
                    except Exception as error:
                        self._mark_failure(entry, error, "mineru_result")
                        shutil.rmtree(target.parent / asset_root, ignore_errors=True)
                        refresh_document_counts()
                        _emit(
                            progress, "processing", counts,
                            source=source, name=Path(source).name, current=source,
                            file_status="failed", error=str(error),
                        )

            refresh_document_counts()
            check_cancelled(cancelled)
            if not counts["ready"]:
                raise RuntimeError("没有成功处理的知识库文档")

            source_fingerprint = [
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path, rel in files
            ]
            manifest = {
                "schema_version": 1,
                "kb_id": kb_id,
                "name": str(name or source_root.name),
                "source_path": str(source_root),
                "imported_at": int(time.time()),
                "source_fingerprint": source_fingerprint,
                "files": entries,
                "summary": dict(counts),
                "failures": [
                    {
                        "source": item["source"],
                        "stage": item.get("stage") or "document",
                        "error_type": item.get("error_type") or "ProcessingError",
                        "error": item.get("error") or "",
                    }
                    for item in document_entries if item["status"] == "failed"
                ],
            }
            processed_root.mkdir(parents=True, exist_ok=True)
            (stage / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _emit(progress, "prepared", counts, current="文档处理完成", files=entries)
            return {
                "source_path": str(source_root),
                "name": manifest["name"],
                "stage_path": str(stage),
                "processed_path": str(processed_root),
                "manifest": manifest,
                "summary": dict(counts),
                "files": document_entries,
                "failures": list(manifest["failures"]),
            }
        finally:
            for path in (downloads_root, extract_root):
                shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _selection_label(path: Path) -> str:
        digest = hashlib.sha256(
            os.path.normcase(str(path)).encode("utf-8", "replace")
        ).hexdigest()[:12]
        return f"files/{digest}-{_safe_component(path.name, 64)}"

    def _prepare_selection_workspace(
        self,
        source_files: list[str],
        workspace: Path,
    ) -> dict[str, Path]:
        """Create a small, private source tree for selected external files.

        MinerU still receives a directory, but the directory is assembled only
        from the files the user selected. Markdown image references are copied
        into a per-document assets folder and rewritten, so selecting one file
        does not accidentally ingest its whole parent directory.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, Path] = {}
        seen: set[str] = set()
        for raw in source_files:
            path = Path(str(raw or "")).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"source file not found: {path}")
            identity = os.path.normcase(str(path))
            if identity in seen:
                continue
            seen.add(identity)
            key = hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()[:16]
            doc_dir = workspace / key
            doc_dir.mkdir(parents=True, exist_ok=True)
            target = doc_dir / _safe_component(path.name, 96)
            if path.suffix.lower() in MARKDOWN_EXTS:
                body = read_textfile(str(path))
                body = _normalize_mineru_image_links(path, path.parent, body)
                replacements: list[tuple[int, int, str]] = []
                for match, raw_ref in _image_refs(body):
                    local = _local_image_path(path, raw_ref)
                    if local is None:
                        continue
                    _path_part, image = local
                    if not image.is_file():
                        continue
                    image_name = (
                        hashlib.sha256(str(image).encode("utf-8", "replace"))
                        .hexdigest()[:16]
                        + (image.suffix.lower() or ".bin")
                    )
                    image_target = doc_dir / "assets" / image_name
                    image_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image, image_target)
                    relative = quote(
                        os.path.relpath(image_target, target.parent)
                        .replace(os.sep, "/"),
                        safe="/-._~",
                    )
                    replacements.append((match.start(2), match.end(2), relative))
                for start, end, replacement in reversed(replacements):
                    body = body[:start] + replacement + body[end:]
                target.write_text(body, encoding="utf-8")
            else:
                shutil.copy2(path, target)
            mapping[f"{key}/{target.name}"] = path
        if not mapping:
            raise ValueError("no source files selected")
        return mapping

    def prepare_files(
        self,
        source_files: list[str],
        *,
        stage_root: str,
        kb_id: str,
        name: str = "",
        progress: Callable[[dict], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Prepare a user-selected set of files without importing a parent folder."""
        selected_paths: list[Path] = []
        seen: set[str] = set()
        for raw in source_files or []:
            path = Path(str(raw or "")).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"source file not found: {path}")
            identity = os.path.normcase(str(path))
            if identity in seen:
                continue
            seen.add(identity)
            selected_paths.append(path)
        if not selected_paths:
            raise ValueError("no source files selected")

        fingerprint_paths = list(selected_paths)
        fingerprint_seen = {os.path.normcase(str(path)) for path in fingerprint_paths}
        for path in selected_paths:
            if path.suffix.lower() not in MARKDOWN_EXTS:
                continue
            try:
                body = _normalize_mineru_image_links(
                    path, path.parent, read_textfile(str(path))
                )
            except Exception:
                continue
            for _match, raw_ref in _image_refs(body):
                local = _local_image_path(path, raw_ref)
                if local is None:
                    continue
                _path_part, image = local
                if not image.is_file():
                    continue
                identity = os.path.normcase(str(image))
                if identity not in fingerprint_seen:
                    fingerprint_seen.add(identity)
                    fingerprint_paths.append(image)

        stage = Path(stage_root).resolve()
        workspace = stage / ".selected_sources"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            mapping = self._prepare_selection_workspace(
                [str(path) for path in selected_paths], workspace
            )
            result = self.prepare(
                str(workspace),
                stage_root=str(stage),
                kb_id=kb_id,
                name=name,
                progress=progress,
                cancelled=cancelled,
            )
            manifest = dict(result.get("manifest") or {})
            source_by_temp = mapping
            for entry in manifest.get("files") or []:
                temp_source = str(entry.get("source") or "").replace("\\", "/")
                original = source_by_temp.get(temp_source)
                if original is None:
                    continue
                entry["source"] = self._selection_label(original)
                entry["source_path"] = str(original)
                entry["name"] = original.name
            for failure in manifest.get("failures") or []:
                temp_source = str(failure.get("source") or "").replace("\\", "/")
                for temp_rel, original in source_by_temp.items():
                    if temp_source == temp_rel or temp_source.startswith(f"{temp_rel}:"):
                        failure["source"] = self._selection_label(original)
                        failure["source_path"] = str(original)
                        break
            manifest["source_path"] = ""
            manifest["source_fingerprint"] = sorted(
                [
                    {
                        "source": self._selection_label(path),
                        "path": str(path),
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                    for path in fingerprint_paths
                ],
                key=lambda item: item["path"].casefold(),
            )
            result["manifest"] = manifest
            result["source_path"] = ""
            result["files"] = [
                entry
                for entry in manifest.get("files") or []
                if entry.get("kind") == "document"
            ]
            result["failures"] = list(manifest.get("failures") or [])
            return result
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


__all__ = ["DocumentProcessor", "MAX_PDF_PAGES"]
