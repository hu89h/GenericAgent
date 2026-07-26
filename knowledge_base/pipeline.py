"""Transactional knowledge-base ingest, publish, reindex, and delete flows."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
from typing import Callable

from . import config
from .build import IndexBuilder, RecordBuilder
from .importer import DocumentProcessor
from .locking import mutation_lock


def _write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temp_path)


class Publisher:
    """Swap a verified staging package into the deterministic active path."""

    @staticmethod
    def _rename_with_retry(source: str, destination: str) -> None:
        last_error = None
        for delay in (0, 0.2, 0.5, 1, 2, 4):
            if delay:
                time.sleep(delay)
            try:
                os.rename(source, destination)
                return
            except PermissionError as error:
                last_error = error
        if last_error is not None:
            raise last_error

    def publish(self, *, kb_id: str, name: str, source_path: str) -> dict:
        root = config.kb_root(kb_id)
        stage = config.staging_root(kb_id)
        active = config.active_root(kb_id)
        rollback = os.path.join(root, "rollback")
        if not os.path.isdir(stage):
            raise RuntimeError("staging 知识库不存在")

        shutil.rmtree(rollback, ignore_errors=True)

        had_active = os.path.isdir(active)
        if had_active:
            self._rename_with_retry(active, rollback)
        try:
            self._rename_with_retry(stage, active)
            config.upsert_kb(
                kb_id,
                name=name,
                source_path=source_path,
            )
        except Exception:
            shutil.rmtree(active, ignore_errors=True)
            if had_active and os.path.isdir(rollback):
                self._rename_with_retry(rollback, active)
            raise

        shutil.rmtree(rollback, ignore_errors=True)
        return config.kb_by_id(kb_id) or {}


class IngestPipeline:
    def __init__(
        self,
        *,
        document_processor: DocumentProcessor,
        record_builder: RecordBuilder,
        index_builder: IndexBuilder,
        publisher: Publisher,
        index,
    ) -> None:
        self.documents = document_processor
        self.records = record_builder
        self.index_builder = index_builder
        self.publisher = publisher
        self.index = index

    @staticmethod
    def _emit(progress: Callable[[dict], None] | None, **event) -> None:
        if callable(progress):
            progress(event)

    @staticmethod
    def _copy_image_cache(kb_id: str, stage_processed: str) -> None:
        active_cache = os.path.join(
            config.processed_path(kb_id), ".kb_index", "image_cache"
        )
        stage_cache = os.path.join(stage_processed, ".kb_index", "image_cache")
        if os.path.isdir(active_cache):
            shutil.copytree(active_cache, stage_cache, dirs_exist_ok=True)

    def cleanup_orphans(self) -> None:
        """Remove crash leftovers only while no KB mutation is running."""
        with mutation_lock:
            data_root = os.path.realpath(config.DATA_ROOT)
            if not os.path.isdir(data_root):
                return
            for name in os.listdir(data_root):
                unresolved = os.path.join(data_root, name)
                if os.path.islink(unresolved):
                    continue
                candidate = os.path.realpath(unresolved)
                if (
                    candidate == data_root
                    or os.path.commonpath((data_root, candidate)) != data_root
                ):
                    continue
                deleting_prefix = name.split(".deleting-", 1)[0]
                if ".deleting-" in name and config.valid_kb_id(deleting_prefix):
                    shutil.rmtree(candidate, ignore_errors=True)
                    continue
                if not config.valid_kb_id(name) or not os.path.isdir(candidate):
                    continue
                for transient in ("staging", "rollback"):
                    shutil.rmtree(
                        os.path.join(candidate, transient),
                        ignore_errors=True,
                    )
                index_root = os.path.join(
                    candidate, "active", "processed", ".kb_index"
                )
                if not os.path.isdir(index_root):
                    continue
                for child in os.listdir(index_root):
                    if child.startswith(("zvec.tmp.", "zvec.rollback.")):
                        shutil.rmtree(
                            os.path.join(index_root, child),
                            ignore_errors=True,
                        )

    def import_kb(
        self,
        source_dir: str,
        *,
        name: str = "",
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
    ) -> dict:
        source_path = config.canonical_source_path(source_dir)
        kb_id = config.kb_id_for_source(source_path)
        root = config.kb_root(kb_id)
        stage = config.staging_root(kb_id)
        with mutation_lock:
            os.makedirs(root, exist_ok=True)
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(os.path.join(root, "rollback"), ignore_errors=True)
            try:
                self.index_builder.begin_build()
                prepared = self.documents.prepare(
                    source_path,
                    stage_root=stage,
                    kb_id=kb_id,
                    name=name,
                    progress=progress,
                )
                stage_processed = prepared["processed_path"]
                self._copy_image_cache(kb_id, stage_processed)
                stage_kb = {
                    "id": kb_id,
                    "name": prepared["name"],
                    "source_path": source_path,
                    "path": stage_processed,
                    "exists": True,
                }
                built = self.records.build(
                    stage_kb,
                    prepared["manifest"],
                    progress=progress,
                    logfn=logfn,
                )
                records_file = os.path.join(stage, "records.jsonl")
                self.records.write_records(records_file, built.records, kb_id=kb_id)
                sources = {
                    **built.sources,
                    "records_sha256": self.records.records_sha256(records_file),
                }
                index_stats = self.index_builder.build(
                    stage_kb,
                    records=built.records,
                    sources=sources,
                    progress=progress,
                    logfn=logfn,
                )

                failures = list(prepared["failures"]) + list(built.failures)
                manifest = dict(prepared["manifest"])
                summary = {
                    **prepared["summary"],
                    **{
                        key: int(index_stats.get(key) or 0)
                        for key in (
                            "n_docs", "n_chunks", "text_chunks",
                            "image_chunks", "image_assets",
                        )
                    },
                    "failed": len(failures),
                }
                manifest.update({
                    "state": "ready_with_warnings" if failures else "ready",
                    "published_at": int(time.time()),
                    "failures": failures,
                    "summary": summary,
                    "index_sources": sources,
                    "index_stats": index_stats,
                })
                _write_json_atomic(os.path.join(stage, "manifest.json"), manifest)
                probe = self.index.probe(stage_processed)
                if not all(
                    probe[key]
                    for key in ("present", "openable", "schema_valid", "embedding_matches")
                ):
                    raise RuntimeError("staging 索引发布前校验失败")

                self._emit(
                    progress,
                    phase="publishing",
                    current=prepared["name"],
                    **summary,
                )
                kb = self.publisher.publish(
                    kb_id=kb_id,
                    name=prepared["name"],
                    source_path=source_path,
                )
                result = {
                    "ok": True,
                    "state": manifest["state"],
                    "kb": kb,
                    "summary": summary,
                    "failures": failures,
                    "files": prepared["files"],
                }
                self._emit(
                    progress,
                    phase="completed_with_failures" if failures else "completed",
                    current="",
                    files=prepared["files"],
                    result=result,
                    **summary,
                )
                return result
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise

    def reindex(
        self,
        kb_id: str,
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
    ) -> dict:
        with mutation_lock:
            kb = config.kb_by_id(kb_id)
            if kb is None or not kb.get("exists"):
                raise KeyError("knowledge_base_not_found")
            self.index_builder.begin_build()
            with open(kb["manifest_path"], encoding="utf-8") as handle:
                manifest = json.load(handle)
            records = self.records.read_records(kb["records_path"])
            sources = dict(manifest.get("index_sources") or {})
            sources["records_sha256"] = self.records.records_sha256(kb["records_path"])
            stats = self.index_builder.build(
                kb,
                records=records,
                sources=sources,
                progress=progress,
                logfn=logfn,
            )
            manifest["index_sources"] = sources
            manifest["index_stats"] = stats
            manifest["reindexed_at"] = int(time.time())
            _write_json_atomic(kb["manifest_path"], manifest)
            return {"ok": True, "kb_id": kb_id, "stats": stats}

    def delete(self, kb_id: str) -> dict:
        with mutation_lock:
            kb = config.kb_by_id(kb_id)
            root = config.kb_root(kb_id)
            if kb is None and not os.path.lexists(root):
                return {"removed": False, "kb_id": kb_id, "data_deleted": False}
            deleting = f"{root}.deleting-{os.getpid()}-{time.time_ns()}"
            moved = False
            if os.path.lexists(root):
                if os.path.islink(root):
                    raise RuntimeError("拒绝删除符号链接知识库目录")
                Publisher._rename_with_retry(root, deleting)
                moved = True
            try:
                removed = config.remove_kb(kb_id)
            except Exception:
                if moved and os.path.exists(deleting):
                    Publisher._rename_with_retry(deleting, root)
                raise
            if moved:
                shutil.rmtree(deleting)
            return {
                "removed": bool(removed or moved),
                "kb_id": kb_id,
                "data_deleted": moved,
            }


__all__ = ["IngestPipeline", "Publisher"]
