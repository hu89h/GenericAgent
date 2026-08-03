"""Record generation and complete-index construction."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable

from . import documents
from .assets import ImageAssetProcessor
from .cancellation import KnowledgeBaseCancelled, check_cancelled
from .providers import provider_settings, vision
from .schema import normalize_record
from .usage import UsageTracker


@dataclass
class RecordBuildResult:
    records: list[dict]
    failures: list[dict]
    sources: dict
    stats: dict


class RecordBuilder:
    """Turn processed Markdown into one homogeneous text/image record stream."""

    def __init__(self, *, assets: ImageAssetProcessor) -> None:
        self.assets = assets

    @staticmethod
    def _title_map(manifest: dict) -> dict[str, str]:
        titles: dict[str, str] = {}
        for entry in manifest.get("files") or []:
            if not isinstance(entry, dict) or entry.get("kind") != "document":
                continue
            title = str(entry.get("name") or "") or os.path.basename(str(entry.get("source") or ""))
            for rel in entry.get("processed") or []:
                titles[str(rel).replace("\\", "/").lstrip("/")] = title
        return titles

    def build(
        self,
        kb: dict,
        manifest: dict,
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        include_files: set[str] | None = None,
    ) -> RecordBuildResult:
        """Build records for all staged documents or a selected subset.

        Incremental document imports pass ``include_files`` and merge the
        resulting records with the already published record stream in the
        pipeline.  Reindex and full imports leave it unset and process every
        staged document.
        """
        log = logfn or (lambda _message: None)
        all_scanned = documents.scan_documents(kb["path"])
        selected_files = {
            str(value or "").replace("\\", "/").lstrip("/")
            for value in (include_files or set())
            if str(value or "").strip()
        }
        scanned = (
            [item for item in all_scanned if item[0].replace("\\", "/") in selected_files]
            if include_files is not None
            else all_scanned
        )
        titles = self._title_map(manifest)
        records: list[dict] = []
        failures: list[dict] = []
        image_records: list[dict] = []
        image_jobs = {}
        image_indexes = {}
        image_capability_warnings: set[str] = set()
        docs_with_chunks = 0

        for position, (rel, absolute_path, _mtime, _size) in enumerate(scanned, 1):
            check_cancelled(cancelled)
            title = titles.get(rel, os.path.basename(rel))
            if callable(progress):
                progress({
                    "phase": "chunking",
                    "current": title,
                    "document": rel,
                    "processed": position - 1,
                    "total": len(scanned),
                })
            ext = os.path.splitext(rel)[1].lower().lstrip(".")
            data_id = f"{kb['id']}::{rel}"
            try:
                text = documents.extract_text(absolute_path)
                image_index = (
                    self.assets.build_document_index(text)
                    if ext in ("md", "markdown") else None
                )
                if image_index is not None:
                    image_indexes[rel] = image_index
                chunks = documents.chunk_document_records(
                    text,
                    ext=ext,
                    file_name=rel,
                    image_index=image_index,
                )
                if image_index is not None:
                    image_index.assign_chunks(chunks)
                if not chunks:
                    raise ValueError("文档没有可索引正文")
                docs_with_chunks += 1
                for chunk_index, chunk in enumerate(chunks):
                    records.append({
                        "data_id": data_id,
                        "chunk_index": chunk_index,
                        "title": title,
                        "file_name": rel,
                        "kind": "text",
                        "source_chunk_index": -1,
                        "header_path": chunk.get("header_path", ""),
                        "body": chunk.get("body", ""),
                    })
                if image_index is not None:
                    image_result = self.assets.image_records_for_document(
                        kb,
                        rel,
                        data_id,
                        text,
                        title,
                        log,
                        image_jobs=image_jobs,
                        image_index=image_index,
                    )
                    image_records.extend(image_result.get("assets") or [])
                    for missing in image_result.get("missing") or []:
                        failures.append({
                            "source": f"{rel}:{missing.get('path') or ''}",
                            "document": rel,
                            "stage": "image_resolve",
                            "error_type": "ImageNotFound",
                            "error": "Markdown 图片引用无法解析",
                        })
                check_cancelled(cancelled)
            except KnowledgeBaseCancelled:
                raise
            except Exception as error:
                failures.append({
                    "source": rel,
                    "document": rel,
                    "stage": "chunking",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                log(f"  [warn] 跳过文档 {rel}: {error}")

        image_results = self.assets.analyze_image_jobs(
            kb,
            image_jobs,
            log,
            progress=progress,
            cancelled=cancelled,
        )
        check_cancelled(cancelled)
        for record in image_records:
            check_cancelled(cancelled)
            analysis = image_results.get(record.get("image_id"))
            self.assets.apply_image_analysis(record, analysis)
            capability_warning = str(record.get("analysis_warning") or "").strip()
            if capability_warning:
                document = str(record.get("file_name") or "")
                if document and document not in image_capability_warnings:
                    image_capability_warnings.add(document)
                    failures.append({
                        "source": document,
                        "document": document,
                        "stage": "image_capability",
                        "error_type": "VisionUnsupported",
                        "error": capability_warning,
                    })
            if record.get("analysis_error"):
                failures.append({
                    "source": (
                        f"{record.get('file_name') or ''}:"
                        f"{record.get('image_path') or ''}"
                    ).strip(":"),
                    "document": record.get("file_name") or "",
                    "stage": "image_analysis",
                    "error_type": "ImageAnalysisError",
                    "error": record.get("analysis_error") or "图片分析失败",
                })
                continue
            if record.get("body"):
                records.append(record)

        if not records and include_files is None:
            raise RuntimeError("没有可索引的图文记录")

        sources = {
            "documents": documents.fingerprint(scanned),
            "chunking": documents.chunking_meta(),
            "images": self.assets.image_source_fingerprint(
                kb["path"], scanned, image_indexes=image_indexes
            ),
            "image_analysis": vision.build_analysis_meta(),
        }
        stats = {
            "n_docs": docs_with_chunks,
            "n_chunks": len(records),
            "text_chunks": sum(record.get("kind") != "image" for record in records),
            "image_chunks": sum(record.get("kind") == "image" for record in records),
            "image_assets": sum(record.get("kind") == "image" for record in records),
        }
        if callable(progress):
            progress({
                "phase": "records_ready",
                "processed": len(scanned),
                "total": len(scanned),
                **stats,
            })
        return RecordBuildResult(records=records, failures=failures, sources=sources, stats=stats)

    @staticmethod
    def write_records(path: str, records: list[dict], *, kb_id: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".records-", suffix=".jsonl", dir=os.path.dirname(path)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            normalize_record(record, kb_id=kb_id),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def read_records(path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                value = line.strip()
                if not value:
                    continue
                try:
                    record = json.loads(value)
                except json.JSONDecodeError as error:
                    raise ValueError(f"records.jsonl 第 {line_number} 行损坏") from error
                if not isinstance(record, dict):
                    raise ValueError(f"records.jsonl 第 {line_number} 行不是对象")
                records.append(record)
        if not records:
            raise ValueError("records.jsonl 为空")
        return records

    @staticmethod
    def records_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class IndexBuilder:
    """Build and verify one complete index from persisted unified records."""

    def __init__(self, *, index, usage_tracker) -> None:
        self.index = index
        self.usage = usage_tracker

    def begin_build(self) -> None:
        usage = self.usage.empty()
        # Snapshot model names at the start of the mutation.  These names are
        # display metadata only; credentials and endpoints never enter the
        # persisted usage report.
        try:
            usage["models"] = {
                "image": str(provider_settings.vision_config().get("model") or ""),
                "embedding": str(provider_settings.embedding_config().get("model") or ""),
            }
        except Exception:
            # Usage reporting must not make an otherwise valid build fail.
            pass
        self.usage.set_current(usage)

    @staticmethod
    def usage_summary(usage: dict) -> dict:
        return UsageTracker.summary(usage)

    def build(
        self,
        kb: dict,
        *,
        records: list[dict],
        sources: dict,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        check_cancelled(cancelled)
        if callable(progress):
            progress({"phase": "indexing", "processed": 0, "total": len(records)})
        stats = self.index.build(
            kb,
            records,
            sources,
            logfn=logfn,
            cancelled=cancelled,
            progress=progress,
        )
        check_cancelled(cancelled)
        usage = self.usage.current()
        usage["stats"] = dict(stats)
        self.usage.write(kb["path"], usage)
        probe = self.index.probe(kb["path"])
        check_cancelled(cancelled)
        if not (
            probe["present"]
            and probe["openable"]
            and probe["schema_valid"]
            and probe["embedding_matches"]
        ):
            raise RuntimeError(
                "索引校验失败: "
                + (probe.get("error") or json.dumps({
                    key: probe[key]
                    for key in ("present", "openable", "schema_valid", "embedding_matches")
                }, ensure_ascii=False))
            )
        if callable(progress):
            progress({
                "phase": "validated",
                "processed": len(records),
                "total": len(records),
                "usage": self.usage_summary(usage),
                **stats,
            })
        return {**stats, "usage": self.usage_summary(usage)}


__all__ = ["IndexBuilder", "RecordBuildResult", "RecordBuilder"]
