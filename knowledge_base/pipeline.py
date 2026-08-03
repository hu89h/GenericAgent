"""Transactional knowledge-base ingest, publish, reindex, and delete flows."""
from __future__ import annotations

import contextlib
import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import config
from .build import IndexBuilder, RecordBuilder
from .assets import cleanup_image_cache
from .cancellation import KnowledgeBaseCancelled, check_cancelled
from .fs import remove_tree
from .importer import DocumentProcessor
from .locking import mutation_lock
from .providers import mineru


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

        remove_tree(rollback)

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
            remove_tree(active)
            if had_active and os.path.isdir(rollback):
                self._rename_with_retry(rollback, active)
            raise

        remove_tree(rollback)
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
    def _checkpoint_manifest(stage: str) -> dict | None:
        path = os.path.join(stage, "manifest.json")
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            return None
        if not isinstance(manifest, dict) or manifest.get("state") != "checkpoint":
            return None
        checkpoint = manifest.get("checkpoint")
        return manifest if isinstance(checkpoint, dict) else None

    @classmethod
    def _write_checkpoint_marker(
        cls,
        *,
        stage: str,
        kb_id: str,
        name: str,
        source_path: str,
        mode: str,
        source_files: list[str] | None = None,
    ) -> dict:
        path = os.path.join(stage, "manifest.json")
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.update({
            "schema_version": int(manifest.get("schema_version") or 1),
            "kb_id": kb_id,
            "name": str(name or manifest.get("name") or kb_id),
            "source_path": str(source_path or manifest.get("source_path") or ""),
            "state": "checkpoint",
        })
        manifest["checkpoint"] = {
            "created_at": int(time.time()),
            "mode": str(mode or "import"),
            "source_files": [str(item) for item in (source_files or []) if item],
            "ready_documents": sum(
                item.get("status") == "ready"
                for item in manifest.get("files") or []
                if isinstance(item, dict) and item.get("kind") == "document"
            ),
            "total_documents": sum(
                item.get("kind") == "document"
                for item in manifest.get("files") or []
                if isinstance(item, dict)
            ),
        }
        _write_json_atomic(path, manifest)
        return manifest

    def checkpoint_status(self, kb_id: str) -> dict:
        stage = config.staging_root(str(kb_id or "").strip())
        manifest = self._checkpoint_manifest(stage)
        if not manifest:
            return {"available": False}
        checkpoint = manifest.get("checkpoint") or {}
        return {
            "available": True,
            "created_at": checkpoint.get("created_at"),
            "checkpoint_at": checkpoint.get("created_at"),
            "mode": str(checkpoint.get("mode") or "import"),
            "ready_documents": int(checkpoint.get("ready_documents") or 0),
            "total_documents": int(checkpoint.get("total_documents") or 0),
            "completed_documents": int(checkpoint.get("ready_documents") or 0),
            "pending_documents": max(
                0,
                int(checkpoint.get("total_documents") or 0)
                - int(checkpoint.get("ready_documents") or 0),
            ),
        }

    def checkpoint_inputs(self, kb_id: str) -> dict:
        """Return private resume inputs for the import endpoint only."""
        stage = config.staging_root(str(kb_id or "").strip())
        manifest = self._checkpoint_manifest(stage)
        if not manifest:
            return {"available": False}
        checkpoint = manifest.get("checkpoint") or {}
        return {
            "available": True,
            "source_path": str(manifest.get("source_path") or ""),
            "source_files": [
                str(item)
                for item in (checkpoint.get("source_files") or [])
                if str(item).strip()
            ],
        }

    def discard_checkpoint(self, kb_id: str) -> dict:
        """Drop only an interrupted staging checkpoint, keeping the KB active."""
        value = str(kb_id or "").strip()
        with mutation_lock:
            stage = config.staging_root(value)
            available = self._checkpoint_manifest(stage) is not None
            if available:
                remove_tree(stage)
            return {"ok": True, "kb_id": value, "discarded": available}

    @staticmethod
    def _emit(progress: Callable[[dict], None] | None, **event) -> None:
        if callable(progress):
            progress(event)

    @staticmethod
    def _ensure_image_cache(kb_id: str) -> str:
        """Keep completed VLM results outside disposable staging."""
        durable = config.image_cache_root(kb_id)
        os.makedirs(durable, exist_ok=True)
        active_cache = os.path.join(
            config.processed_path(kb_id), ".kb_index", "image_cache"
        )
        if os.path.isdir(active_cache):
            shutil.copytree(active_cache, durable, dirs_exist_ok=True)
        return durable

    @staticmethod
    def _relative_key(value) -> str:
        return str(value or "").replace("\\", "/").lstrip("/")

    @classmethod
    def _manifest_document_target(
        cls,
        manifest: dict,
        *,
        kb_id: str,
        data_id: str = "",
        file_name: str = "",
        ref: str = "",
    ) -> tuple[dict | None, str]:
        value = str(data_id or "").strip()
        if "::" in value:
            value = value.split("::", 1)[1].split("::image::", 1)[0]
        if not value:
            value = str(file_name or "").strip()
        if not value:
            ref_value = cls._relative_key(ref)
            prefix = f"{kb_id}/"
            value = ref_value[len(prefix):] if ref_value.startswith(prefix) else ref_value
        value = cls._relative_key(value)
        if value.startswith("processed/"):
            value = value[len("processed/"):]
        if not value:
            return None, ""
        for entry in manifest.get("files") or []:
            if not isinstance(entry, dict) or entry.get("kind") != "document":
                continue
            processed = [cls._relative_key(item) for item in entry.get("processed") or []]
            if value in processed or f"processed/{value}" in processed:
                return entry, value
        return None, value

    @classmethod
    def _document_results(
        cls,
        manifest: dict,
        records: list[dict],
        failures: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Resolve preprocessing/build outcomes back to original source documents."""
        entries = [
            item for item in (manifest.get("files") or [])
            if isinstance(item, dict) and item.get("kind") == "document"
        ]
        by_processed: dict[str, dict] = {}
        by_source: dict[str, dict] = {}
        for entry in entries:
            source = cls._relative_key(entry.get("source"))
            if source:
                by_source[source] = entry
            for processed in entry.get("processed") or []:
                key = cls._relative_key(processed)
                if key:
                    by_processed[key] = entry

        decorated_failures: list[dict] = []
        for raw in failures:
            failure = dict(raw or {})
            failure_source = cls._relative_key(failure.get("source"))
            document_key = cls._relative_key(failure.get("document"))
            owner = by_processed.get(document_key) or by_source.get(document_key)
            if owner is None:
                owner = by_source.get(failure_source)
            if owner is None:
                for processed, entry in by_processed.items():
                    if failure_source == processed or failure_source.startswith(f"{processed}:"):
                        owner = entry
                        break
            if owner is not None:
                failure["source_document"] = str(
                    (
                        owner.get("name")
                        if owner.get("source_path")
                        else cls._relative_key(owner.get("source"))
                    )
                    or cls._relative_key(owner.get("source"))
                )
            decorated_failures.append(failure)

        document_results: list[dict] = []
        for entry in entries:
            source = cls._relative_key(entry.get("source"))
            processed = {
                cls._relative_key(value)
                for value in (entry.get("processed") or [])
                if cls._relative_key(value)
            }
            owned_failures = [
                failure for failure in decorated_failures
                if cls._relative_key(failure.get("source_document")) in {
                    source,
                    cls._relative_key(entry.get("name")),
                }
            ]
            owned_records = [
                record for record in records
                if cls._relative_key(record.get("file_name")) in processed
            ]
            text_chunks = sum(record.get("kind") != "image" for record in owned_records)
            images_indexed = sum(record.get("kind") == "image" for record in owned_records)
            image_failures = sum(
                failure.get("stage") in {"image_resolve", "image_analysis"}
                for failure in owned_failures
            )
            if text_chunks:
                status = "succeeded_with_warnings" if owned_failures else "succeeded"
            else:
                status = "failed"
            document_results.append({
                "source": source,
                "name": str(entry.get("name") or "") or os.path.basename(source),
                "status": status,
                "text_chunks": text_chunks,
                "images_indexed": images_indexed,
                "images_total": images_indexed + image_failures,
                "failures": owned_failures,
            })
        return document_results, decorated_failures

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
                    remove_tree(candidate)
                    continue
                if not config.valid_kb_id(name) or not os.path.isdir(candidate):
                    continue
                stage = os.path.join(candidate, "staging")
                if self._checkpoint_manifest(stage):
                    for transient in (".mineru_downloads", ".mineru_extract", ".selected_sources"):
                        remove_tree(os.path.join(stage, transient), ignore_errors=True)
                else:
                    remove_tree(stage, ignore_errors=True)
                remove_tree(os.path.join(candidate, "rollback"), ignore_errors=True)
                mineru.cleanup_cache(config.mineru_cache_root(name))
                cleanup_image_cache(config.image_cache_root(name))
                index_root = os.path.join(
                    candidate, "active", "processed", ".kb_index"
                )
                if not os.path.isdir(index_root):
                    continue
                for child in os.listdir(index_root):
                    if child.startswith(("zvec.tmp.", "zvec.rollback.")):
                        remove_tree(
                            os.path.join(index_root, child),
                            ignore_errors=True,
                        )

    def _build_and_publish(
        self,
        *,
        kb_id: str,
        name: str,
        source_path: str,
        stage: str,
        manifest: dict,
        prepared_summary: dict,
        prepared_failures: list[dict],
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        existing_records: list[dict] | None = None,
        include_files: set[str] | None = None,
        existing_sources: dict | None = None,
    ) -> dict:
        """Build a complete index from the staged processed documents and publish it."""
        check_cancelled(cancelled)
        stage_processed = os.path.join(stage, "processed")
        stage_kb = {
            "id": kb_id,
            "name": name,
            "source_path": source_path,
            "path": stage_processed,
            "image_cache_path": self._ensure_image_cache(kb_id),
            "exists": True,
        }
        build_kwargs = {
            "progress": progress,
            "logfn": logfn,
            "cancelled": cancelled,
        }
        if include_files is not None:
            build_kwargs["include_files"] = include_files
        built = self.records.build(stage_kb, manifest, **build_kwargs)
        check_cancelled(cancelled)
        records = list(existing_records or [])
        if include_files is not None:
            normalized_files = {
                str(value or "").replace("\\", "/").lstrip("/")
                for value in include_files
            }
            records = [
                record for record in records
                if str(record.get("file_name") or "").replace("\\", "/").lstrip("/")
                not in normalized_files
            ]
        records.extend(built.records)
        records_file = os.path.join(stage, "records.jsonl")
        self.records.write_records(records_file, records, kb_id=kb_id)
        check_cancelled(cancelled)
        sources = dict(existing_sources or {})
        for key, value in built.sources.items():
            if key == "documents" and isinstance(value, list):
                previous = {
                    str(item.get("path") or "").replace("\\", "/"): item
                    for item in sources.get(key) or []
                    if isinstance(item, dict) and item.get("path")
                }
                previous.update({
                    str(item.get("path") or "").replace("\\", "/"): item
                    for item in value
                    if isinstance(item, dict) and item.get("path")
                })
                sources[key] = sorted(previous.values(), key=lambda item: str(item.get("path") or ""))
            elif key == "images" and isinstance(value, dict):
                merged_images = dict(sources.get(key) or {})
                merged_images.update(value)
                sources[key] = merged_images
            else:
                sources[key] = value
        sources["records_sha256"] = self.records.records_sha256(records_file)
        index_stats = self.index_builder.build(
            stage_kb,
            records=records,
            sources=sources,
            progress=progress,
            logfn=logfn,
            cancelled=cancelled,
        )
        check_cancelled(cancelled)

        document_results, failures = self._document_results(
            manifest,
            records,
            list(prepared_failures) + list(built.failures),
        )
        documents_succeeded = sum(
            item["status"] in {"succeeded", "succeeded_with_warnings"}
            for item in document_results
        )
        documents_with_warnings = sum(
            item["status"] == "succeeded_with_warnings"
            for item in document_results
        )
        documents_failed = sum(item["status"] == "failed" for item in document_results)
        summary = {
            **dict(prepared_summary or {}),
            **{
                key: int(index_stats.get(key) or 0)
                for key in (
                    "n_docs", "n_chunks", "text_chunks",
                    "image_chunks", "image_assets",
                    "embedding_cache_hits", "embedding_cache_misses",
                )
            },
            "total": len(document_results),
            "completed": len(document_results),
            "succeeded": documents_succeeded,
            "failed": documents_failed,
            "documents_total": len(document_results),
            "documents_succeeded": documents_succeeded,
            "documents_with_warnings": documents_with_warnings,
            "documents_failed": documents_failed,
            "failure_items": len(failures),
        }
        manifest = dict(manifest)
        manifest["processing_fingerprint"] = {
            "mineru_cache_version": int(getattr(mineru, "MINERU_CACHE_VERSION", 1)),
            "chunking": dict(built.sources.get("chunking") or {}),
            "image_analysis": dict(built.sources.get("image_analysis") or {}),
            "embedding": dict(index_stats.get("embedding") or {}),
            "sparse_embedding": dict(index_stats.get("sparse_embedding") or {}),
        }
        manifest.update({
            "schema_version": int(manifest.get("schema_version") or 1),
            "kb_id": kb_id,
            "name": name,
            "source_path": source_path,
            "state": "ready_with_warnings" if failures else "ready",
            "published_at": int(time.time()),
            "failures": failures,
            "summary": summary,
            "document_results": document_results,
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

        self._emit(progress, phase="publishing", current=name, **summary)
        check_cancelled(cancelled)
        kb = self.publisher.publish(
            kb_id=kb_id,
            name=name,
            source_path=source_path,
        )
        result = {
            "ok": True,
            "state": manifest["state"],
            "kb": kb,
            "summary": summary,
            "usage": dict(index_stats.get("usage") or {}),
            "failures": failures,
            "documents": document_results,
        }
        if existing_records and int(index_stats.get("embedding_cache_misses") or 0) and not int(
            index_stats.get("embedding_cache_hits") or 0
        ):
            result["notice"] = "embedding_cache_rebuilt"
        self._emit(
            progress,
            phase="completed_with_failures" if failures else "completed",
            current="",
            documents=document_results,
            result=result,
            usage=result["usage"],
            **summary,
        )
        return result

    def import_kb(
        self,
        source_dir: str,
        *,
        name: str = "",
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        retain_partial: Callable[[], bool] | None = None,
    ) -> dict:
        source_path = config.canonical_source_path(source_dir)
        kb_id = config.kb_id_for_source(source_path)
        root = config.kb_root(kb_id)
        stage = config.staging_root(kb_id)
        with mutation_lock:
            check_cancelled(cancelled)
            os.makedirs(root, exist_ok=True)
            checkpoint = self._checkpoint_manifest(stage)
            resume_manifest = (
                checkpoint
                if checkpoint
                and os.path.normcase(os.path.realpath(str(checkpoint.get("source_path") or "")))
                == os.path.normcase(source_path)
                else None
            )
            if resume_manifest:
                self._emit(
                    progress,
                    phase="resuming",
                    current="继续使用已保留的文档处理结果",
                    completed=(resume_manifest.get("checkpoint") or {}).get("ready_documents", 0),
                    total=(resume_manifest.get("checkpoint") or {}).get("total_documents", 0),
                )
            else:
                remove_tree(stage)
            remove_tree(os.path.join(root, "rollback"))
            try:
                self.index_builder.begin_build()
                check_cancelled(cancelled)
                prepare_kwargs = {
                    "stage_root": stage,
                    "kb_id": kb_id,
                    "name": name,
                    "progress": progress,
                    "cancelled": cancelled,
                }
                if resume_manifest is not None:
                    prepare_kwargs["resume_manifest"] = resume_manifest
                if retain_partial is not None:
                    prepare_kwargs["retain_on_cancel"] = retain_partial
                prepared = self.documents.prepare(source_path, **prepare_kwargs)
                check_cancelled(cancelled)
                self._ensure_image_cache(kb_id)
                check_cancelled(cancelled)
                return self._build_and_publish(
                    kb_id=kb_id,
                    name=prepared["name"],
                    source_path=source_path,
                    stage=stage,
                    manifest=prepared["manifest"],
                    prepared_summary=prepared["summary"],
                    prepared_failures=prepared["failures"],
                    progress=progress,
                    logfn=logfn,
                    cancelled=cancelled,
                )
            except KnowledgeBaseCancelled:
                if callable(retain_partial) and retain_partial():
                    self._write_checkpoint_marker(
                        stage=stage,
                        kb_id=kb_id,
                        name=name,
                        source_path=source_path,
                        mode="import",
                    )
                else:
                    remove_tree(stage)
                raise
            except Exception:
                remove_tree(stage)
                raise

    def add_documents(
        self,
        kb_id: str,
        source_files: list[str],
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        retain_partial: Callable[[], bool] | None = None,
        replace_existing: bool = False,
    ) -> dict:
        """Append selected external files and rebuild one complete staged index."""
        value = str(kb_id or "").strip()
        if not value:
            raise ValueError("知识库 ID 不能为空")
        with mutation_lock:
            check_cancelled(cancelled)
            kb = config.kb_by_id(value)
            if kb is None:
                raise KeyError("knowledge_base_not_found")
            self.index_builder.begin_build()
            stage = config.staging_root(value)
            checkpoint = self._checkpoint_manifest(stage)
            checkpoint_meta = (checkpoint or {}).get("checkpoint") or {}
            requested_keys = {
                os.path.normcase(os.path.realpath(str(raw)))
                for raw in source_files or []
                if raw
            }
            checkpoint_keys = {
                os.path.normcase(os.path.realpath(str(raw)))
                for raw in checkpoint_meta.get("source_files") or []
                if raw
            }
            resume_checkpoint = bool(
                checkpoint
                and checkpoint_meta.get("mode") == "add_documents"
                and requested_keys
                and requested_keys == checkpoint_keys
            )
            if not resume_checkpoint:
                remove_tree(stage)
            try:
                active = config.active_root(value)
                if not resume_checkpoint and os.path.isdir(active):
                    shutil.copytree(active, stage, dirs_exist_ok=True)
                os.makedirs(os.path.join(stage, "processed"), exist_ok=True)
                old_manifest = {}
                manifest_path = os.path.join(stage, "manifest.json")
                if os.path.isfile(manifest_path):
                    with open(manifest_path, encoding="utf-8") as handle:
                        old_manifest = json.load(handle)
                old_manifest = old_manifest if isinstance(old_manifest, dict) else {}
                source_root = str(kb.get("source_path") or "").strip()
                existing_source_paths = set()
                existing_entries_by_path: dict[str, dict] = {}
                for entry in old_manifest.get("files") or []:
                    if not isinstance(entry, dict) or entry.get("kind") != "document":
                        continue
                    source_value = str(entry.get("source_path") or "").strip()
                    if not source_value and source_root:
                        relative = str(entry.get("source") or "").replace("/", os.sep)
                        candidate = os.path.realpath(os.path.join(source_root, relative))
                        if os.path.isfile(candidate):
                            source_value = candidate
                    if source_value and not resume_checkpoint:
                        identity = os.path.normcase(os.path.realpath(source_value))
                        existing_source_paths.add(identity)
                        existing_entries_by_path[identity] = entry
                selected = []
                selected_keys = set()
                for raw in source_files or []:
                    path = os.path.realpath(os.path.expanduser(str(raw or "")))
                    if not os.path.isfile(path):
                        raise ValueError(f"source file not found: {path}")
                    identity = os.path.normcase(path)
                    if (
                        resume_checkpoint
                        or replace_existing
                        or identity not in existing_source_paths
                    ) and identity not in selected_keys:
                        selected.append(path)
                        selected_keys.add(identity)
                if not selected:
                    raise ValueError("所选文件均已在知识库中")
                prepare_kwargs = {
                    "stage_root": stage,
                    "kb_id": value,
                    "name": kb.get("name") or value,
                    "progress": progress,
                    "cancelled": cancelled,
                }
                if resume_checkpoint:
                    prepare_kwargs["resume_manifest"] = old_manifest
                if retain_partial is not None:
                    prepare_kwargs["retain_on_cancel"] = retain_partial
                prepared = self.documents.prepare_files(selected, **prepare_kwargs)
                merged = dict(old_manifest)
                prepared_files = list(prepared["manifest"].get("files") or [])
                new_processed_files = {
                    str(processed).replace("\\", "/").lstrip("/")
                    for entry in prepared_files
                    if isinstance(entry, dict)
                    for processed in entry.get("processed") or []
                    if str(processed or "").strip()
                }
                existing_records = None
                existing_index_sources = None
                records_file = os.path.join(stage, "records.jsonl")
                if os.path.isfile(records_file):
                    existing_records = self.records.read_records(records_file)
                    existing_index_sources = old_manifest.get("index_sources") or {}
                selected_labels = {
                    self.documents._selection_label(Path(path))
                    for path in selected
                }
                replaced_processed_files = set()
                if replace_existing and not resume_checkpoint:
                    for path in selected:
                        old_entry = existing_entries_by_path.get(
                            os.path.normcase(os.path.realpath(path))
                        )
                        if old_entry:
                            replaced_processed_files.update(
                                str(item).replace("\\", "/").lstrip("/")
                                for item in (old_entry.get("processed") or [])
                                if str(item).strip()
                            )
                selected_identities = {
                    os.path.normcase(os.path.realpath(path)) for path in selected
                }

                def keep_existing_entry(item: dict) -> bool:
                    if resume_checkpoint:
                        return item.get("source") not in selected_labels
                    if not replace_existing:
                        return True
                    source_value = str(item.get("source_path") or "").strip()
                    if not source_value and source_root:
                        relative = str(item.get("source") or "").replace("/", os.sep)
                        source_value = os.path.join(source_root, relative)
                    if not source_value:
                        return True
                    return os.path.normcase(os.path.realpath(source_value)) not in selected_identities

                existing_files = [
                    item for item in (old_manifest.get("files") or [])
                    if isinstance(item, dict)
                    and keep_existing_entry(item)
                ]
                if replace_existing and replaced_processed_files:
                    processed_root = os.path.realpath(os.path.join(stage, "processed"))
                    for relative in replaced_processed_files:
                        target = os.path.realpath(os.path.join(processed_root, relative))
                        if os.path.commonpath((processed_root, target)) != processed_root:
                            raise ValueError("非法处理后文档路径")
                        if os.path.isfile(target):
                            os.remove(target)
                        parent = os.path.dirname(target)
                        stem = os.path.splitext(os.path.basename(target))[0]
                        for asset_dir in glob.glob(os.path.join(parent, f"{stem}.assets-*")):
                            remove_tree(asset_dir)
                merged["files"] = existing_files + prepared_files
                merged["source_path"] = kb.get("source_path") or ""
                fingerprints = {
                    str(item.get("path") or "").casefold(): item
                    for item in old_manifest.get("source_fingerprint") or []
                    if isinstance(item, dict) and item.get("path")
                }
                fingerprints.update({
                    str(item.get("path") or "").casefold(): item
                    for item in prepared["manifest"].get("source_fingerprint") or []
                    if isinstance(item, dict) and item.get("path")
                })
                merged["source_fingerprint"] = sorted(
                    fingerprints.values(), key=lambda item: str(item.get("path") or "").casefold()
                )
                old_failures = list(old_manifest.get("failures") or [])
                if resume_checkpoint:
                    old_failures = [
                        item for item in old_failures
                        if not isinstance(item, dict) or item.get("source") not in selected_labels
                    ]
                failures = [
                    *old_failures,
                    *list(prepared.get("failures") or []),
                ]
                merged["failures"] = failures
                merged["name"] = kb.get("name") or value
                return self._build_and_publish(
                    kb_id=value,
                    name=kb.get("name") or value,
                    source_path=kb.get("source_path") or "",
                    stage=stage,
                    manifest=merged,
                    prepared_summary=prepared.get("summary") or {},
                    prepared_failures=failures,
                    progress=progress,
                    logfn=logfn,
                    cancelled=cancelled,
                    existing_records=existing_records,
                    include_files=(
                        new_processed_files | replaced_processed_files
                        if existing_records
                        else None
                    ),
                    existing_sources=(existing_index_sources if existing_records else None),
                )
            except KnowledgeBaseCancelled:
                if callable(retain_partial) and retain_partial():
                    self._write_checkpoint_marker(
                        stage=stage,
                        kb_id=value,
                        name=kb.get("name") or value,
                        source_path=kb.get("source_path") or "",
                        mode="add_documents",
                        source_files=source_files,
                    )
                else:
                    remove_tree(stage)
                raise
            except Exception:
                remove_tree(stage)
                raise

    def reindex(
        self,
        kb_id: str,
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        """Rebuild only the search index from the published records.

        This operation deliberately does not read Markdown again or invoke
        image understanding.  It is the safe maintenance action for an index
        or embedding configuration problem.
        """
        with mutation_lock:
            kb = config.kb_by_id(kb_id)
            if kb is None or not kb.get("exists"):
                raise KeyError("knowledge_base_not_found")
            check_cancelled(cancelled)
            return self._reindex_records(
                kb,
                progress=progress,
                logfn=logfn,
                cancelled=cancelled,
            )

    def retry_image_analysis(
        self,
        kb_id: str,
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        """Retry image understanding against the published processed content."""
        with mutation_lock:
            kb = config.kb_by_id(kb_id)
            if kb is None or not kb.get("exists"):
                raise KeyError("knowledge_base_not_found")
            check_cancelled(cancelled)
            try:
                from .providers import vision
                available = bool(vision.enabled())
            except Exception:
                available = False
            if not available:
                raise RuntimeError("知识库图片理解未启用或没有可用的多模态配置")
            return self._repair_processed_content(
                kb,
                progress=progress,
                logfn=logfn,
                cancelled=cancelled,
            )

    def _reindex_records(
        self,
        kb: dict,
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        """Rebuild Zvec from published records in staging, then publish atomically."""
        check_cancelled(cancelled)
        kb_id = str(kb.get("id") or "")
        if not kb_id:
            raise KeyError("knowledge_base_not_found")
        active = config.active_root(kb_id)
        if not os.path.isdir(active):
            raise KeyError("knowledge_base_not_found")
        stage = config.staging_root(kb_id)
        remove_tree(stage)
        try:
            shutil.copytree(active, stage, dirs_exist_ok=True)
            manifest_path = os.path.join(stage, "manifest.json")
            records_path = os.path.join(stage, "records.jsonl")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            records = self.records.read_records(records_path)
            if not records:
                raise RuntimeError("知识库没有可重建的图文记录")
            stage_processed = os.path.join(stage, "processed")
            stage_kb = {
                **kb,
                "path": stage_processed,
                "manifest_path": manifest_path,
                "records_path": records_path,
                "exists": True,
            }
            sources = dict(manifest.get("index_sources") or {})
            sources["records_sha256"] = self.records.records_sha256(records_path)
            self.index_builder.begin_build()
            stats = self.index_builder.build(
                stage_kb,
                records=records,
                sources=sources,
                progress=progress,
                logfn=logfn,
                cancelled=cancelled,
            )
            check_cancelled(cancelled)
            manifest = dict(manifest)
            manifest["index_sources"] = sources
            manifest["index_stats"] = stats
            manifest["processing_fingerprint"] = {
                "mineru_cache_version": int(getattr(mineru, "MINERU_CACHE_VERSION", 1)),
                "chunking": dict((manifest.get("index_sources") or {}).get("chunking") or {}),
                "image_analysis": dict((manifest.get("index_sources") or {}).get("image_analysis") or {}),
                "embedding": dict(stats.get("embedding") or {}),
                "sparse_embedding": dict(stats.get("sparse_embedding") or {}),
            }
            manifest["reindexed_at"] = int(time.time())
            manifest["state"] = (
                "ready_with_warnings" if manifest.get("failures") else "ready"
            )
            _write_json_atomic(manifest_path, manifest)
            probe = self.index.probe(stage_processed)
            if not all(
                probe.get(key)
                for key in ("present", "openable", "schema_valid", "embedding_matches")
            ):
                raise RuntimeError("重建索引发布前校验失败")
            self._emit(
                progress,
                phase="publishing",
                current=str(manifest.get("name") or kb.get("name") or kb_id),
            )
            check_cancelled(cancelled)
            published = self.publisher.publish(
                kb_id=kb_id,
                name=str(manifest.get("name") or kb.get("name") or kb_id),
                source_path=str(manifest.get("source_path") or kb.get("source_path") or ""),
            )
            return {
                "ok": True,
                "kb_id": kb_id,
                "stats": stats,
                "usage": dict(stats.get("usage") or {}),
                "kb": published,
            }
        except Exception:
            remove_tree(stage)
            raise

    def _repair_processed_content(
        self,
        kb: dict,
        *,
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        """Retry processed image analyses and publish a rebuilt complete index."""
        kb_id = str(kb.get("id") or "")
        active = config.active_root(kb_id)
        if not os.path.isdir(active):
            raise KeyError("knowledge_base_not_found")
        with open(kb["manifest_path"], encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("知识库清单格式无效")

        # Start a fresh usage window before retrying image analysis.  The
        # subsequent index build must retain those image counters and the
        # model snapshot instead of resetting them after analysis completes.
        self.index_builder.begin_build()
        stage = config.staging_root(kb_id)
        remove_tree(stage)
        try:
            check_cancelled(cancelled)
            shutil.copytree(active, stage, dirs_exist_ok=True)
            name = str(manifest.get("name") or kb.get("name") or kb_id)
            source_path = str(manifest.get("source_path") or kb.get("source_path") or "")
            total_documents = sum(
                1
                for item in (manifest.get("files") or [])
                if isinstance(item, dict) and item.get("kind") == "document"
            )
            self._emit(
                progress,
                phase="preparing",
                current=name,
                processed=0,
                total=total_documents,
            )
            # RecordBuilder recomputes image failures from the staged Markdown.
            # Keep unrelated source/parse failures, but do not carry stale VLM
            # warnings into the new publication.
            preserved_failures = [
                item
                for item in (manifest.get("failures") or [])
                if isinstance(item, dict)
                and str(item.get("stage") or "") not in {"image_analysis", "image_capability"}
            ]
            result = self._build_and_publish(
                kb_id=kb_id,
                name=name,
                source_path=source_path,
                stage=stage,
                manifest=manifest,
                prepared_summary=dict(manifest.get("summary") or {}),
                prepared_failures=preserved_failures,
                progress=progress,
                logfn=logfn,
                cancelled=cancelled,
            )
            return result
        except Exception:
            remove_tree(stage)
            raise

    def delete_document(
        self,
        kb_id: str,
        *,
        data_id: str = "",
        file_name: str = "",
        ref: str = "",
        progress: Callable[[dict], None] | None = None,
        logfn: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        """Remove one processed document and publish a rebuilt index atomically."""
        value = str(kb_id or "").strip()
        with mutation_lock:
            check_cancelled(cancelled)
            kb = config.kb_by_id(value)
            if kb is None or not kb.get("exists"):
                raise KeyError("knowledge_base_not_found")
            with open(kb["manifest_path"], encoding="utf-8") as handle:
                manifest = json.load(handle)
            entry, target_rel = self._manifest_document_target(
                manifest,
                kb_id=value,
                data_id=data_id,
                file_name=file_name,
                ref=ref,
            )
            if entry is None:
                raise KeyError("document_not_found")

            removed_name = str(entry.get("name") or "")
            removed_source = self._relative_key(entry.get("source"))
            removed_processed = {
                self._relative_key(item)
                for item in (entry.get("processed") or [])
                if self._relative_key(item)
            }
            remaining_files = [
                item for item in (manifest.get("files") or [])
                if item is not entry
            ]
            remaining_documents = [
                item for item in remaining_files
                if isinstance(item, dict)
                and item.get("kind") == "document"
                and item.get("processed")
            ]
            root = config.kb_root(value)
            if not remaining_documents:
                stage = config.staging_root(value)
                remove_tree(stage)
                try:
                    os.makedirs(os.path.join(stage, "processed"), exist_ok=True)
                    empty_manifest = dict(manifest)
                    empty_manifest.update({
                        "state": "empty",
                        "files": [],
                        "failures": [],
                        "document_results": [],
                        "index_sources": {},
                        "index_stats": {},
                        "summary": {
                            "n_docs": 0,
                            "n_chunks": 0,
                            "text_chunks": 0,
                            "image_chunks": 0,
                            "image_assets": 0,
                            "documents_total": 0,
                            "documents_succeeded": 0,
                            "documents_failed": 0,
                            "documents_with_warnings": 0,
                        },
                        "published_at": int(time.time()),
                    })
                    _write_json_atomic(
                        os.path.join(stage, "manifest.json"),
                        empty_manifest,
                    )
                    published = self.publisher.publish(
                        kb_id=value,
                        name=kb.get("name") or value,
                        source_path=kb.get("source_path") or "",
                    )
                except Exception:
                    remove_tree(stage)
                    raise
                self._emit(progress, phase="completed", processed=1, total=1)
                return {
                    "ok": True,
                    "kb_id": value,
                    "data_id": data_id or f"{value}::{target_rel}",
                    "document_name": removed_name or target_rel,
                    "empty": True,
                    "kb": published,
                }

            stage = config.staging_root(value)
            remove_tree(stage)
            try:
                active = config.active_root(value)
                if not os.path.isdir(active):
                    raise KeyError("knowledge_base_not_found")
                shutil.copytree(active, stage, dirs_exist_ok=True)
                stage_processed = os.path.join(stage, "processed")
                for relative in removed_processed:
                    target = os.path.realpath(os.path.join(stage_processed, relative))
                    processed_root = os.path.realpath(stage_processed)
                    if os.path.commonpath((processed_root, target)) != processed_root:
                        raise ValueError("非法处理后文档路径")
                    if os.path.isfile(target):
                        os.remove(target)
                    parent = os.path.dirname(target)
                    stem = os.path.splitext(os.path.basename(target))[0]
                    for asset_dir in glob.glob(os.path.join(parent, f"{stem}.assets-*")):
                        if os.path.isdir(asset_dir):
                            remove_tree(asset_dir)

                records = self.records.read_records(kb["records_path"])
                kept_records = []
                for record in records:
                    check_cancelled(cancelled)
                    record_rel = self._relative_key(record.get("file_name"))
                    if record_rel.startswith("processed/"):
                        record_rel = record_rel[len("processed/"):]
                    if record_rel not in removed_processed:
                        kept_records.append(record)
                if not kept_records:
                    raise RuntimeError("删除后没有可检索的文档记录")
                records_path = os.path.join(stage, "records.jsonl")
                self.records.write_records(records_path, kept_records, kb_id=value)

                failures = []
                for failure in manifest.get("failures") or []:
                    values = (
                        self._relative_key(failure.get("source")),
                        self._relative_key(failure.get("document")),
                        self._relative_key(failure.get("source_document")),
                    )
                    owned = any(
                        item and (
                            item == removed_source
                            or (removed_source and item.startswith(f"{removed_source}:"))
                            or item == removed_name
                        )
                        for item in values
                    )
                    if not owned:
                        failures.append(dict(failure))

                sources = dict(manifest.get("index_sources") or {})
                sources["records_sha256"] = self.records.records_sha256(records_path)
                stage_kb = {
                    **kb,
                    "path": stage_processed,
                    "exists": True,
                }
                self.index_builder.begin_build()
                self._emit(progress, phase="indexing", processed=0, total=len(kept_records))
                stats = self.index_builder.build(
                    stage_kb,
                    records=kept_records,
                    sources=sources,
                    progress=progress,
                    logfn=logfn,
                    cancelled=cancelled,
                )
                check_cancelled(cancelled)
                next_manifest = dict(manifest)
                next_manifest["files"] = remaining_files
                next_manifest["failures"] = failures
                next_manifest["index_sources"] = sources
                next_manifest["index_stats"] = stats
                next_manifest["state"] = "ready_with_warnings" if failures else "ready"
                next_manifest["published_at"] = int(time.time())
                document_results, decorated_failures = self._document_results(
                    next_manifest, kept_records, failures
                )
                next_manifest["failures"] = decorated_failures
                next_manifest["document_results"] = document_results
                next_manifest["summary"] = {
                    **dict(manifest.get("summary") or {}),
                    "n_docs": int(stats.get("n_docs") or 0),
                    "n_chunks": int(stats.get("n_chunks") or 0),
                    "text_chunks": int(stats.get("text_chunks") or 0),
                    "image_chunks": int(stats.get("image_chunks") or 0),
                    "image_assets": int(stats.get("image_assets") or 0),
                    "documents_total": len(document_results),
                    "documents_succeeded": sum(
                        item["status"] in {"succeeded", "succeeded_with_warnings"}
                        for item in document_results
                    ),
                    "documents_with_warnings": sum(
                        item["status"] == "succeeded_with_warnings"
                        for item in document_results
                    ),
                    "documents_failed": sum(
                        item["status"] == "failed" for item in document_results
                    ),
                }
                _write_json_atomic(os.path.join(stage, "manifest.json"), next_manifest)
                probe = self.index.probe(stage_processed)
                if not all(
                    probe[key]
                    for key in ("present", "openable", "schema_valid", "embedding_matches")
                ):
                    raise RuntimeError("删除文档后的索引校验失败")
                self._emit(progress, phase="publishing", current=removed_name)
                check_cancelled(cancelled)
                published = self.publisher.publish(
                    kb_id=value,
                    name=kb.get("name") or value,
                    source_path=kb.get("source_path") or "",
                )
                self._emit(progress, phase="completed", processed=1, total=1)
                return {
                    "ok": True,
                    "kb_id": value,
                    "data_id": data_id or f"{value}::{target_rel}",
                    "document_name": removed_name or target_rel,
                    "empty": False,
                    "summary": next_manifest["summary"],
                    "stats": stats,
                    "kb": published,
                }
            except Exception:
                remove_tree(stage)
                raise

    def delete(
        self,
        kb_id: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        with mutation_lock:
            check_cancelled(cancelled)
            kb = config.kb_by_id(kb_id)
            root = config.kb_root(kb_id)
            if kb is None and not os.path.lexists(root):
                return {"removed": False, "kb_id": kb_id, "data_deleted": False}
            deleting = f"{root}.deleting-{os.getpid()}-{time.time_ns()}"
            moved = False
            if os.path.lexists(root):
                check_cancelled(cancelled)
                if os.path.islink(root):
                    raise RuntimeError("拒绝删除符号链接知识库目录")
                Publisher._rename_with_retry(root, deleting)
                moved = True
            try:
                check_cancelled(cancelled)
                removed = config.remove_kb(kb_id)
            except Exception:
                if moved and os.path.exists(deleting):
                    Publisher._rename_with_retry(deleting, root)
                raise
            if moved:
                remove_tree(deleting, ignore_errors=False)
            return {
                "removed": bool(removed or moved),
                "kb_id": kb_id,
                "data_deleted": moved,
            }


__all__ = ["IngestPipeline", "Publisher"]
