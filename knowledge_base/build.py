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
from .providers import vision
from .schema import normalize_record


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
            title = os.path.basename(str(entry.get("source") or "")) or str(entry.get("name") or "")
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
    ) -> RecordBuildResult:
        log = logfn or (lambda _message: None)
        scanned = documents.scan_documents(kb["path"])
        titles = self._title_map(manifest)
        records: list[dict] = []
        failures: list[dict] = []
        image_records: list[dict] = []
        image_jobs = {}
        image_indexes = {}
        docs_with_chunks = 0

        for position, (rel, absolute_path, _mtime, _size) in enumerate(scanned, 1):
            if callable(progress):
                progress({
                    "phase": "chunking",
                    "current": rel,
                    "processed": position - 1,
                    "total": len(scanned),
                })
            ext = os.path.splitext(rel)[1].lower().lstrip(".")
            title = titles.get(rel, os.path.basename(rel))
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
                            "stage": "image_resolve",
                            "error_type": "ImageNotFound",
                            "error": "Markdown 图片引用无法解析",
                        })
            except Exception as error:
                failures.append({
                    "source": rel,
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
        )
        for record in image_records:
            analysis = image_results.get(record.get("image_id"))
            self.assets.apply_image_analysis(record, analysis)
            if record.get("analysis_error"):
                failures.append({
                    "source": record.get("image_path") or record.get("file_name") or "",
                    "stage": "image_analysis",
                    "error_type": "ImageAnalysisError",
                    "error": record.get("analysis_error") or "图片分析失败",
                })
                continue
            if record.get("body"):
                records.append(record)

        if not records:
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
        self.usage.set_current(self.usage.empty())

    @staticmethod
    def usage_summary(usage: dict) -> dict:
        image = usage.get("image_analysis") or {}
        dense = usage.get("embedding") or {}
        sparse = usage.get("sparse_embedding") or {}
        return {
            "image_calls": image.get("calls", 0),
            "image_cached": image.get("cached", 0),
            "image_failed": image.get("failed", 0),
            "image_prompt_tokens": image.get("prompt_tokens", 0),
            "image_completion_tokens": image.get("completion_tokens", 0),
            "embedding_calls": dense.get("calls", 0),
            "embedding_texts": dense.get("texts", 0),
            "embedding_api_tokens": dense.get("api_tokens", 0),
            "sparse_embedding_calls": sparse.get("calls", 0),
            "sparse_embedding_texts": sparse.get("texts", 0),
            "sparse_embedding_api_tokens": sparse.get("api_tokens", 0),
        }

    def build(
        self,
        kb: dict,
        *,
        records: list[dict],
        sources: dict,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
    ) -> dict:
        if callable(progress):
            progress({"phase": "indexing", "processed": 0, "total": len(records)})
        stats = self.index.build(kb, records, sources, logfn=logfn)
        usage = self.usage.current()
        usage["stats"] = dict(stats)
        self.usage.write(kb["path"], usage)
        probe = self.index.probe(kb["path"])
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
                **stats,
            })
        return {**stats, "usage": self.usage_summary(usage)}


__all__ = ["IndexBuilder", "RecordBuildResult", "RecordBuilder"]
