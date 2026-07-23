"""Knowledge-base build orchestration.

This module owns the build lifecycle and record preparation.  It deliberately
receives storage/provider operations from ``backend`` so the public backend
facade can keep its existing API without making this module import it back.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import threading
import time
import traceback
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class DocumentServices:
    scan_documents: Callable[..., list]
    extract_text: Callable[..., str]
    chunk_document_records: Callable[..., list]
    build_sources: Callable[..., Dict[str, Any]]
    write_parent_chunks: Callable[..., None]


@dataclass(frozen=True)
class ImageServices:
    build_image_related_index: Callable[..., dict]
    image_records_for_chunk: Callable[..., list]
    analyze_image_jobs: Callable[..., dict]
    apply_image_analysis: Callable[..., None]
    write_image_assets: Callable[..., None]
    load_image_assets: Callable[..., list]


@dataclass(frozen=True)
class IndexServices:
    zvec_path: Callable[..., str]
    require_zvec: Callable[..., Any]
    zvec_meta: Callable[..., Dict[str, Any]]
    zvec_is_quickly_fresh: Callable[..., bool]
    build_zvec_index: Callable[..., tuple]
    append_zvec_image_index: Callable[..., tuple]


@dataclass(frozen=True)
class UsageServices:
    set_usage: Callable[..., None]
    empty_usage: Callable[..., Dict[str, Any]]
    usage: Callable[..., Dict[str, Any]]
    write_build_usage: Callable[..., None]
    build_usage_summary: Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class BuildServices:
    documents: DocumentServices
    images: ImageServices
    index: IndexServices
    usage: UsageServices
    update_build_state: Callable[..., None]
    load_config: Callable[..., list]


class BuildCoordinator:
    """Coordinate one-at-a-time builds and prepare index records."""

    def __init__(self, services: BuildServices, lock_port: int, build_lock=None) -> None:
        self._services = services
        self._lock_port = int(lock_port)
        self._build_lock = build_lock if build_lock is not None else threading.Lock()

    def records(
        self,
        kb,
        scanned,
        log,
        include_text=True,
        include_images=True,
        progressfn=None,
    ):
        """Generate the records consumed by the Zvec index builder."""
        services = self._services
        documents = services.documents
        images = services.images
        kb_id = kb["id"]
        n_files = len(scanned)
        n_ok = n_empty = 0
        parent_chunks = []
        image_records = []
        image_jobs = {}
        image_count = 0

        for i, (rel, ap, _mt, _sz) in enumerate(scanned, 1):
            ext = os.path.splitext(rel)[1].lower().lstrip(".")
            data_id = f"{kb_id}::{rel}"
            title = os.path.basename(rel)
            try:
                text = documents.extract_text(ap)
            except Exception as exc:
                log(f"  [warn] 抽取失败 {rel}: {exc}")
                text = ""

            related_index = (
                images.build_image_related_index(text) if include_images else {}
            )
            try:
                chunk_records = documents.chunk_document_records(
                    text, ext=ext, file_name=rel
                )
            except Exception as exc:
                log(f"  [warn] Markdown 分块失败 {rel}: {exc}")
                chunk_records = []

            text_chunk_records = [
                chunk for chunk in chunk_records
                if chunk.get("chunk_role") != "parent"
            ]
            for chunk in chunk_records:
                if chunk.get("chunk_role") == "parent":
                    parent_chunks.append({
                        "data_id": data_id,
                        "parent_chunk_index": int(chunk.get("parent_chunk_index", -1)),
                        "title": title,
                        "file_name": rel,
                        "header_path": chunk.get("header_path", ""),
                        "body": chunk.get("body", ""),
                    })

            if not text_chunk_records:
                n_empty += 1
            else:
                n_ok += 1

            for chunk_index, chunk in enumerate(text_chunk_records):
                body = chunk.get("body", "")
                record = {
                    "data_id": data_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "file_name": rel,
                    "kind": "text",
                    "image_path": "",
                    "parent_data_id": "",
                    "parent_chunk_index": int(chunk.get("parent_chunk_index", -1)),
                    "header_path": chunk.get("header_path", ""),
                    "chunk_role": chunk.get("chunk_role", "leaf"),
                    "body": body,
                }
                if include_text:
                    # Text records can be embedded while documents are scanned;
                    # image records remain buffered until VLM analysis finishes.
                    yield record
                if include_images:
                    chunk_images = images.image_records_for_chunk(
                        kb,
                        rel,
                        ap,
                        data_id,
                        chunk_index,
                        body,
                        title,
                        log,
                        image_jobs=image_jobs,
                        related_index=related_index,
                    )
                    for image_record in chunk_images:
                        image_count += 1
                        image_records.append(image_record)

            if i % 100 == 0 or i == n_files:
                if callable(progressfn):
                    progressfn(i, n_files)
                log(f"  进度 {i}/{n_files} 文件（有效 {n_ok}，空 {n_empty}）...")

        image_results = (
            images.analyze_image_jobs(kb, image_jobs, log)
            if include_images
            else {}
        )
        if image_results:
            for record in image_records:
                if record.get("kind") == "image":
                    images.apply_image_analysis(
                        record, image_results.get(record.get("image_id"))
                    )

        stored_assets = [
            {
                key: value
                for key, value in record.items()
                if not key.startswith("_") and key != "body"
            }
            for record in image_records
            if record.get("kind") == "image"
        ]
        images.write_image_assets(kb["path"], stored_assets)
        documents.write_parent_chunks(kb["path"], parent_chunks)
        log(
            f"  抽取完成：{n_files} 文件，有效 {n_ok}，无正文 {n_empty}，图片资产 {image_count}"
        )
        yield from image_records

    def _finalize_build_result(self, kb, scanned, z_status, z_stats, log):
        """Persist the common build report and publish the final build state."""
        services = self._services
        images = services.images
        usage_service = services.usage
        z_stats = dict(z_stats or {})
        n_images = z_stats.get("image_assets")
        if n_images is None:
            n_images = len(images.load_image_assets(kb["path"]))

        stats = dict(z_stats)
        stats["zvec"] = dict(z_stats)
        stats["index_backend"] = "zvec"
        stats["zvec_status"] = z_status
        stats["image_assets"] = n_images

        usage = usage_service.usage()
        usage["stats"] = {
            "n_docs": stats.get("n_docs", 0),
            "n_chunks": stats.get("n_chunks", 0),
            "image_assets": n_images,
        }
        usage_service.write_build_usage(kb["path"], usage)
        stats["usage"] = usage_service.build_usage_summary(usage)

        done = "已最新" if z_status == "up-to-date" else "完成"
        log(
            f"索引{done}：{stats.get('n_docs', 0)} 文档 / "
            f"{stats.get('n_chunks', 0)} chunk / 图片 {n_images} / zvec={z_status}"
        )
        if z_status == "unavailable":
            log(f"  [error] zvec 不可用：{z_stats.get('error')}")

        succeeded = z_status in ("built", "up-to-date")
        services.update_build_state(
            phase="completed" if succeeded else "failed",
            message=f"索引{done}",
            processed=len(scanned),
            total=len(scanned),
        )
        return (z_status if succeeded else "unavailable", stats)

    def build_kb(self, kb, force=False, verbose=True, logfn=None, mode="full"):
        """Build one knowledge base and return ``(status, stats)``."""
        services = self._services
        documents = services.documents
        index = services.index
        usage = services.usage

        def log(message):
            if logfn:
                logfn(message)
            elif verbose:
                print(f"[kb:{kb['id']}] {message}", flush=True)

        if not kb.get("exists"):
            log(f"路径不存在，跳过：{kb['path']}")
            return ("missing", {"error": f"path not found: {kb['path']}"})

        mode = mode if mode in ("full", "text", "images") else "full"
        services.update_build_state(
            kb=kb["id"],
            phase="scanning",
            message="扫描知识库文档",
            processed=0,
            total=0,
        )
        scanned = documents.scan_documents(kb["path"])
        if not scanned:
            log("未发现可索引文档")
            services.update_build_state(
                phase="completed",
                message="未发现可索引文档",
                processed=0,
                total=0,
            )
            return ("empty", {"n_docs": 0, "n_chunks": 0})

        services.update_build_state(
            phase="checking",
            message="检查已有索引是否最新",
            processed=0,
            total=len(scanned),
        )
        usage.set_usage(usage.empty_usage(kb["id"], kb["path"]))
        mode_label = {"full": "完整", "text": "文本", "images": "图片资产"}[mode]
        try:
            index.require_zvec()
        except Exception as exc:
            log(f"Zvec 不可用，无法建立索引：{exc}")
            services.update_build_state(phase="failed", message=f"Zvec 不可用：{exc}")
            return ("unavailable", {"error": str(exc), "index_backend": "zvec"})

        meta = index.zvec_meta(kb["path"])
        if not force and index.zvec_is_quickly_fresh(
            kb["path"], scanned, meta=meta, mode=mode
        ):
            return self._finalize_build_result(
                kb, scanned, "up-to-date", meta.get("stats") or {}, log
            )

        sources = documents.build_sources(
            kb["path"], scanned, mode="text" if mode == "text" else "full"
        )
        services.update_build_state(
            phase="extracting",
            message=f"抽取{mode_label}索引记录",
            processed=0,
            total=len(scanned),
        )
        log(
            f"发现 {len(scanned)} 个文档，开始建立 {mode_label} Zvec 索引 → "
            f"{index.zvec_path(kb['path'])}"
        )
        records = self.records(
            kb,
            scanned,
            log,
            include_text=mode in ("full", "text"),
            include_images=mode in ("full", "images"),
            progressfn=lambda processed, total: services.update_build_state(
                phase="extracting", processed=processed, total=total
            ),
        )
        services.update_build_state(
            phase="indexing",
            message="写入 Zvec 索引",
            processed=0,
            total=len(scanned),
        )
        if mode == "images":
            z_status, z_stats = index.append_zvec_image_index(
                kb, records, sources, force=force, logfn=log
            )
        else:
            z_status, z_stats = index.build_zvec_index(
                kb, records, sources, force=force, logfn=log
            )
        return self._finalize_build_result(kb, scanned, z_status, z_stats, log)

    @staticmethod
    def build_summary(results):
        summary = {"total": len(results), "succeeded": 0, "failed": 0, "skipped": 0}
        for status, _stats in results.values():
            if status in ("built", "up-to-date"):
                summary["succeeded"] += 1
            elif status in ("empty", "locked"):
                summary["skipped"] += 1
            else:
                summary["failed"] += 1
        return summary

    def build_all(self, force=False, verbose=True, logfn=None, kb_id=None, mode="full"):
        """Build configured knowledge bases with process-local and TCP locks."""
        services = self._services
        if not self._build_lock.acquire(blocking=False):
            return {"_": ("locked", {})}

        proc_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            proc_lock.bind(("127.0.0.1", self._lock_port))
        except OSError:
            proc_lock.close()
            self._build_lock.release()
            if logfn:
                logfn("[build] 已有其它进程在构建索引，跳过本次。")
            return {"_": ("locked", {})}

        results = {}
        services.update_build_state(
            running=True,
            result=None,
            last_build_summary=None,
            started_at=int(time.time()),
            finished_at=None,
            kb=None,
            phase="preparing",
            message="准备构建知识库",
            processed=0,
            total=0,
        )
        try:
            configured = services.load_config()
            selected = [kb for kb in configured if not kb_id or kb["id"] == kb_id]
            for kb in selected:
                services.update_build_state(
                    kb=kb["id"], phase="preparing", message="准备当前知识库"
                )
                try:
                    kb_log = None
                    if callable(logfn):
                        kb_log = lambda message, current_kb=kb["id"]: logfn(
                            f"[kb:{current_kb}] {message}"
                        )
                    results[kb["id"]] = self.build_kb(
                        kb,
                        force=force,
                        verbose=verbose,
                        logfn=kb_log,
                        mode=mode,
                    )
                except Exception as exc:
                    results[kb["id"]] = ("error", {"error": str(exc)})
                    lines = [
                        f"[kb:{kb['id']}] 构建异常: {type(exc).__name__}: {exc}",
                        traceback.format_exc(),
                    ]
                    if logfn:
                        for line in lines:
                            logfn(line)
                    elif verbose:
                        for line in lines:
                            print(line, flush=True)

            if kb_id and not results:
                results[kb_id] = ("missing", {"error": f"unknown kb_id: {kb_id}"})
            services.update_build_state(
                result={key: value[0] for key, value in results.items()},
                last_build_summary=self.build_summary(results),
            )
        finally:
            services.update_build_state(
                running=False,
                kb=None,
                phase="idle",
                message="构建结束",
                finished_at=int(time.time()),
            )
            try:
                proc_lock.close()
            finally:
                self._build_lock.release()
        return results
