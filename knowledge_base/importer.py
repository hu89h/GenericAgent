"""Unified knowledge-base import pipeline.

The importer keeps the public operation intentionally small: import one source
directory.  Markdown is copied directly, while supported non-Markdown files
go through MinerU.  PDFs larger than the service limit are rejected before any
remote job is submitted; the importer deliberately does not split or merge
PDFs.
"""
from __future__ import annotations

import hashlib
import gc
import html
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

try:
    from .config import DATA_ROOT, ROOT, kb_by_id, kb_id_for_source, upsert_kb
    from .documents import read_textfile
    from .providers import mineru
except ImportError:  # pragma: no cover - supports direct CLI execution
    from config import DATA_ROOT, ROOT, kb_by_id, kb_id_for_source, upsert_kb
    from documents import read_textfile
    from providers import mineru


MAX_PDF_PAGES = 200
MARKDOWN_EXTS = {".md", ".markdown"}
IMAGE_EXTS = mineru.IMAGE_EXTS
SUPPORTED_EXTS = mineru.SUPPORTED_EXTS
_IMAGE_SUFFIXES = tuple(sorted(IMAGE_EXTS | {".tif", ".tiff"}))
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


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
    matches: list[re.Match[str]] = []
    matches.extend(_MD_IMAGE_RE.finditer(body or ""))
    # Image links without the ! prefix are accepted because MinerU sometimes
    # emits them for a page asset instead of Markdown image syntax.
    matches.extend(_MD_LINK_RE.finditer(body or ""))
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
        progress({"phase": phase, **counts, **extra})


def _rename_with_retry(source: Path, destination: Path) -> None:
    """Rename a completed import, tolerating short Windows scanner locks."""
    last_error: PermissionError | None = None
    for delay in (0, 0.2, 0.5, 1, 2, 4):
        if delay:
            time.sleep(delay)
        try:
            source.rename(destination)
            return
        except PermissionError as error:
            last_error = error
    if last_error is not None:
        raise last_error


def import_knowledge_base(
    source_dir: str,
    *,
    kb_id: str = "",
    name: str = "",
    overwrite: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Import one directory and persist only generated Markdown/assets."""
    source_root = Path(os.path.realpath(os.path.expanduser(str(source_dir or ""))))
    if not source_root.is_dir():
        raise ValueError(f"sourceDir is not a directory: {source_root}")
    kid = kb_id_for_source(str(source_root))
    destination = (Path(DATA_ROOT) / kid).resolve()
    if destination.exists() and not overwrite:
        raise ValueError(f"knowledge base already exists: {kid}")

    # Keep the temporary root shallow.  Windows installations may not have
    # long-path support enabled, and source-relative paper titles can already
    # be long before the generated asset directory is appended.
    stage_parent = Path(DATA_ROOT)
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".import-", dir=str(stage_parent))).resolve()
    processed_root = stage / "processed"
    downloads_root = stage / ".mineru_downloads"
    extract_root = stage / ".mineru_extract"

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

    def output_rel_for(source_rel: str) -> str:
        digest = hashlib.sha256(source_rel.encode("utf-8")).hexdigest()[:12]
        stem = _safe_component(Path(source_rel).stem, 48)
        candidate = f"documents/{digest}-{stem}.md"
        if candidate in output_rels:
            candidate = f"documents/{digest}-{hashlib.sha256(source_rel.encode('utf-8')).hexdigest()[12:20]}.md"
        output_rels.add(candidate)
        return candidate

    try:
        _emit(progress, "scanning", counts)
        source_by_rel = {os.path.normcase(rel): rel for _path, rel in files}
        has_markdown = any(path.suffix.lower() in MARKDOWN_EXTS for path, _rel in files)
        referenced_images: set[str] = set()
        for path, rel in files:
            if path.suffix.lower() not in MARKDOWN_EXTS:
                continue
            for _match, raw in _image_refs(read_textfile(str(path))):
                local = _local_image_path(path, raw)
                if local is None:
                    continue
                _path_part, image = local
                if _path_is_within(source_root, image) and image.is_file():
                    image_rel = image.relative_to(source_root).as_posix()
                    matched_rel = source_by_rel.get(os.path.normcase(image_rel))
                    if matched_rel:
                        referenced_images.add(matched_rel)

        conversion_specs: list[dict[str, Any]] = []
        for path, rel in files:
            ext = path.suffix.lower()
            entry = {
                "source": rel,
                "name": path.name,
                "kind": "ignored",
                "status": "ignored",
                "processed": [],
                "error": "",
            }
            entries.append(entry)
            entry_by_source[rel] = entry
            if ext in MARKDOWN_EXTS:
                target_rel = output_rel_for(rel)
                target = processed_root / target_rel
                namespace = Path(f"{target.stem}.assets-{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:8]}")
                _write_markdown(path, source_root, target, processed_root, asset_namespace=namespace)
                entry.update(kind="document", status="ready", processed=[target_rel])
                counts["markdown"] += 1
                continue
            if ext in IMAGE_EXTS and rel in referenced_images:
                entry.update(kind="asset", status="asset")
                counts["assets"] += 1
                continue
            # A document folder commonly contains stale extraction images that
            # are no longer referenced by its Markdown.  Treating those files
            # as standalone MinerU documents creates dozens of bogus entries.
            # Image-only folders remain supported as intentional image imports.
            if ext in IMAGE_EXTS and has_markdown:
                counts["ignored"] += 1
                counts["skipped"] += 1
                continue
            if ext not in SUPPORTED_EXTS:
                counts["ignored"] += 1
                counts["skipped"] += 1
                continue
            entry.update(kind="document", status="queued")
            conversion_specs.append({"source": rel, "path": path, "entry": entry})
            counts["converting"] += 1
        counts["total"] = counts["markdown"] + counts["converting"]
        counts["ready"] = counts["markdown"]
        counts["completed"] = counts["markdown"]
        counts["succeeded"] = counts["markdown"]
        _emit(progress, "scanned", counts)

        upload_specs: list[dict[str, Any]] = []
        for spec in conversion_specs:
            source = spec["source"]
            path = spec["path"]
            entry = spec["entry"]
            if path.suffix.lower() == ".pdf":
                try:
                    _pdf_page_count(path)
                except Exception as error:
                    entry.update(status="failed", error=str(error))
                    _emit(
                        progress,
                        "processing",
                        counts,
                        source=source,
                        name=path.name,
                        current=source,
                        file_status="failed",
                        error=str(error),
                    )
                    continue
            upload_specs.append({"source": source, "path": path, "relative_path": source})
        _emit(progress, "processing", counts, current="开始提交 MinerU")

        jobs_by_relative: dict[str, dict[str, Any]] = {}
        if upload_specs:
            def on_mineru_update(job: mineru.MinerUFile) -> None:
                spec = jobs_by_relative.get(job.relative_path)
                if spec is None:
                    return
                entry = entry_by_source[spec["source"]]
                entry.update(status=job.state, error=job.error)
                _emit(
                    progress,
                    "processing",
                    counts,
                    source=spec["source"],
                    name=Path(spec["source"]).name,
                    current=spec["source"],
                    file_status=entry["status"],
                    error=job.error,
                )

            for spec in upload_specs:
                jobs_by_relative[spec["relative_path"]] = spec
            try:
                jobs = mineru.process_batches(
                    [(spec["path"], spec["relative_path"]) for spec in upload_specs],
                    downloads_root,
                    on_update=on_mineru_update,
                )
            except Exception as error:
                jobs = []
                for spec in upload_specs:
                    entry = entry_by_source[spec["source"]]
                    entry.update(status="failed", error=str(error))
                _emit(progress, "processing", counts, current="MinerU 提交失败", error=str(error))
            jobs_by_relative.update({job.relative_path: {**jobs_by_relative[job.relative_path], "job": job} for job in jobs})

            for spec in upload_specs:
                source = spec["source"]
                entry = entry_by_source[source]
                job = jobs_by_relative.get(spec["relative_path"], {}).get("job")
                if not job or job.state != "downloaded":
                    error = getattr(job, "error", "") or entry.get("error") or "MinerU 处理失败"
                    entry.update(status="failed", error=error)
                    counts["failed"] += 1
                    counts["completed"] += 1
                    _emit(
                        progress,
                        "processing",
                        counts,
                        source=source,
                        name=Path(source).name,
                        current=source,
                        file_status="failed",
                        error=error,
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
                    extra_images = [Path(spec["path"])] if Path(source).suffix.lower() in IMAGE_EXTS else []
                    body, _ = _prepare_markdown(
                        markdown,
                        extract_dir,
                        target,
                        processed_root,
                        asset_namespace=asset_root,
                        extra_images=extra_images,
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(body.strip() + "\n", encoding="utf-8")
                    entry.update(status="ready", error="", processed=[target_rel])
                    counts["ready"] += 1
                    counts["succeeded"] += 1
                    counts["completed"] += 1
                    _emit(
                        progress, "processing", counts,
                        source=source, name=Path(source).name, current=source,
                        file_status="ready",
                    )
                except Exception as error:
                    entry.update(status="failed", error=str(error))
                    shutil.rmtree(target.parent / asset_root, ignore_errors=True)
                    counts["failed"] += 1
                    counts["completed"] += 1
                    _emit(
                        progress, "processing", counts,
                        source=source, name=Path(source).name, current=source,
                        file_status="failed", error=str(error),
                    )

        # Recalculate terminal counts after provider failures and conversion.
        document_entries = [item for item in entries if item["kind"] == "document"]
        counts["ready"] = sum(item["status"] == "ready" for item in document_entries)
        counts["assets"] = sum(item["kind"] == "asset" for item in entries)
        counts["ignored"] = sum(item["kind"] == "ignored" for item in entries)
        counts["skipped"] = counts["ignored"]
        counts["failed"] = sum(item["status"] == "failed" for item in document_entries)
        counts["total"] = len(document_entries)
        counts["succeeded"] = counts["ready"]
        counts["completed"] = counts["ready"] + counts["failed"]
        manifest = {
            "schema_version": 5,
            "imported_at": int(time.time()),
            "source_dir_name": source_root.name,
            "files": entries,
        }
        processed_root.mkdir(parents=True, exist_ok=True)
        (stage / "import_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Release response/ZIP/PDF objects before moving the staging tree.
        # Windows also allows antivirus/indexer handles to outlive the close;
        # _rename_with_retry gives those short-lived readers a bounded window.
        gc.collect()
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.replaced-{os.getpid()}-{int(time.time())}")
            _rename_with_retry(destination, backup)
        else:
            backup = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _rename_with_retry(stage, destination)
        except Exception:
            if backup is not None and not destination.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)

        processed = destination / "processed"
        upsert_kb(
            kid,
            path=os.path.relpath(processed, ROOT).replace(os.sep, "/"),
            preload=True,
            name=(name or source_root.name),
            source_path=str(source_root),
        )
        result = {
            "ok": True,
            "kb": kb_by_id(kid),
            "copiedTo": str(destination),
            "manifest": str(destination / "import_manifest.json"),
            "summary": dict(counts),
            "succeeded": [
                item["source"] for item in document_entries if item["status"] == "ready"
            ],
            "failed": [
                item for item in document_entries if item["status"] == "failed"
            ],
            "files": document_entries,
        }
        _emit(progress, "completed", counts, current="导入完成", files=entries, result=result)
        return result
    finally:
        for path in (downloads_root, extract_root):
            shutil.rmtree(path, ignore_errors=True)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
