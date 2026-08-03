#!/usr/bin/env python3
"""Public facade for the GenericAgent knowledge-base runtime."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass

from . import config
from .assets import ImageAssetProcessor
from .build import IndexBuilder, RecordBuilder
from .importer import DocumentProcessor
from .fs import remove_tree
from .locking import KnowledgeBaseLockedError, mutation_lock
from .pipeline import IngestPipeline, Publisher
from .providers import provider_settings
from .retrieval import KnowledgeBaseRetriever
from .usage import UsageTracker
from .zvec import ZvecIndex


DEFAULT_KB_ID = "default_kb"
_SNIPPET = 220
ZVEC_BATCH = int(os.environ.get("GA_KB_ZVEC_BATCH", "256"))
ZVEC_QUERY_FACTOR = max(1, int(os.environ.get("GA_KB_ZVEC_QUERY_FACTOR", "4")))
ZVEC_VECTOR_WEIGHT = float(os.environ.get("GA_KB_VECTOR_WEIGHT", "1.2"))
ZVEC_SPARSE_WEIGHT = float(os.environ.get("GA_KB_SPARSE_WEIGHT", "1.0"))
IMAGE_ANALYSIS_CONCURRENCY = max(
    1, int(os.environ.get("GA_KB_IMAGE_CONCURRENCY", "64"))
)

CONFIG_PATH = config.CONFIG_PATH
DATA_ROOT = config.DATA_ROOT
ROOT = config.ROOT
load_config = config.load_config
kb_id_for_source = config.kb_id_for_source


@dataclass
class _Runtime:
    dimension: int
    usage: UsageTracker
    assets: ImageAssetProcessor
    index: ZvecIndex
    records: RecordBuilder
    index_builder: IndexBuilder
    publisher: Publisher
    pipeline: IngestPipeline
    retrieval: KnowledgeBaseRetriever


_runtime_lock = threading.RLock()
_runtime_instance: _Runtime | None = None
_processing_lock = threading.Lock()
_processing_kbs: set[str] = set()


def _embedding_dimension() -> int:
    value = provider_settings.embedding_config().get("dimension")
    return max(1, int(value or 1024))


def _runtime() -> _Runtime:
    global _runtime_instance
    dimension = _embedding_dimension()
    with _runtime_lock:
        if _runtime_instance is not None and _runtime_instance.dimension == dimension:
            return _runtime_instance
        usage = UsageTracker()
        assets = ImageAssetProcessor(
            usage_tracker=usage,
            concurrency=IMAGE_ANALYSIS_CONCURRENCY,
        )
        index = ZvecIndex(
            dimension=dimension,
            batch_size=ZVEC_BATCH,
            usage_tracker=usage,
        )
        records = RecordBuilder(assets=assets)
        index_builder = IndexBuilder(index=index, usage_tracker=usage)
        publisher = Publisher()
        pipeline = IngestPipeline(
            document_processor=DocumentProcessor(),
            record_builder=records,
            index_builder=index_builder,
            publisher=publisher,
            index=index,
        )
        try:
            pipeline.cleanup_orphans()
        except KnowledgeBaseLockedError:
            # Another process is actively mutating a KB.  Its temporary paths
            # are live, not orphans; the next clean startup can sweep them.
            pass
        retrieval = KnowledgeBaseRetriever(
            registry=config,
            index=index,
            assets=assets,
            query_factor=ZVEC_QUERY_FACTOR,
            vector_weight=ZVEC_VECTOR_WEIGHT,
            sparse_weight=ZVEC_SPARSE_WEIGHT,
            snippet_width=_SNIPPET,
        )
        _runtime_instance = _Runtime(
            dimension=dimension,
            usage=usage,
            assets=assets,
            index=index,
            records=records,
            index_builder=index_builder,
            publisher=publisher,
            pipeline=pipeline,
            retrieval=retrieval,
        )
        return _runtime_instance


def _mark_processing(kb_id: str, active: bool) -> None:
    with _processing_lock:
        if active:
            _processing_kbs.add(kb_id)
        else:
            _processing_kbs.discard(kb_id)


def _is_processing(kb_id: str) -> bool:
    with _processing_lock:
        return kb_id in _processing_kbs


def import_kb(
    source_dir: str,
    *,
    name: str = "",
    progress=None,
    cancelled=None,
    retain_partial=None,
    rescan_source: bool = False,
) -> dict:
    source = config.canonical_source_path(source_dir)
    kb_id = config.kb_id_for_source(source)
    _mark_processing(kb_id, True)
    try:
        return _runtime().pipeline.import_kb(
            source,
            name=name,
            progress=progress,
            cancelled=cancelled,
            retain_partial=retain_partial,
            rescan_source=rescan_source,
        )
    finally:
        _mark_processing(kb_id, False)


def create_kb(name: str) -> dict:
    with mutation_lock:
        return config.create_kb(name)


def add_documents(
    kb_id: str,
    source_files: list[str],
    *,
    progress=None,
    cancelled=None,
    retain_partial=None,
    duplicate_policy: str = "skip",
    rescan_source: bool = False,
) -> dict:
    value = str(kb_id or "").strip()
    _mark_processing(value, True)
    try:
        return _runtime().pipeline.add_documents(
            value,
            source_files,
            progress=progress,
            cancelled=cancelled,
            retain_partial=retain_partial,
            duplicate_policy=duplicate_policy,
            rescan_source=rescan_source,
        )
    finally:
        _mark_processing(value, False)


def delete_document(
    kb_id: str,
    *,
    data_id: str = "",
    file_name: str = "",
    ref: str = "",
    progress=None,
    logfn=None,
    cancelled=None,
) -> dict:
    value = str(kb_id or "").strip()
    _mark_processing(value, True)
    try:
        return _runtime().pipeline.delete_document(
            value,
            data_id=data_id,
            file_name=file_name,
            ref=ref,
            progress=progress,
            logfn=logfn,
            cancelled=cancelled,
        )
    finally:
        _mark_processing(value, False)


def reindex(kb_id: str, *, progress=None, logfn=None, cancelled=None) -> dict:
    value = str(kb_id or "").strip()
    _mark_processing(value, True)
    try:
        return _runtime().pipeline.reindex(
            value,
            progress=progress,
            logfn=logfn,
            cancelled=cancelled,
        )
    finally:
        _mark_processing(value, False)


def retry_image_analysis(
    kb_id: str, *, progress=None, logfn=None, cancelled=None, retain_partial=None
) -> dict:
    value = str(kb_id or "").strip()
    _mark_processing(value, True)
    try:
        return _runtime().pipeline.retry_image_analysis(
            value,
            progress=progress,
            logfn=logfn,
            cancelled=cancelled,
            retain_partial=retain_partial,
        )
    finally:
        _mark_processing(value, False)


def delete_kb(kb_id: str, *, cancelled=None) -> dict:
    value = str(kb_id or "").strip()
    _mark_processing(value, True)
    try:
        return _runtime().pipeline.delete(value, cancelled=cancelled)
    finally:
        _mark_processing(value, False)


def existing_document_files(kb_id: str, source_files: list[str]) -> list[str]:
    """Return selected source names already represented by a KB manifest."""
    kb = config.kb_by_id(str(kb_id or "").strip())
    if not kb:
        return []
    manifest = _load_manifest(kb)
    existing: set[str] = set()
    source_root = str(kb.get("source_path") or "").strip()
    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict) or entry.get("kind") != "document":
            continue
        value = str(entry.get("source_path") or "").strip()
        if not value and source_root:
            relative = str(entry.get("source") or "").replace("/", os.sep)
            value = os.path.join(source_root, relative)
        if value:
            existing.add(os.path.normcase(os.path.realpath(value)))
    return [
        os.path.basename(str(path))
        for path in source_files or []
        if os.path.normcase(os.path.realpath(str(path))) in existing
    ]


def source_inputs(kb_id: str) -> dict:
    """Resolve registered source files privately for a server-side rescan."""
    kb = config.kb_by_id(str(kb_id or "").strip())
    if not kb:
        return {"available": False}
    source_path = str(kb.get("source_path") or "").strip()
    if source_path:
        return {
            "available": os.path.isdir(source_path),
            "source_dir": source_path,
            "source_files": [],
        }
    manifest = _load_manifest(kb)
    files = []
    seen = set()
    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict) or entry.get("kind") != "document":
            continue
        path = str(entry.get("source_path") or "").strip()
        if not path or not os.path.isfile(path):
            continue
        identity = os.path.normcase(os.path.realpath(path))
        if identity in seen:
            continue
        seen.add(identity)
        files.append(path)
    return {"available": bool(files), "source_dir": "", "source_files": files}


def _load_manifest(kb: dict) -> dict:
    try:
        with open(kb["manifest_path"], encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _source_fingerprint(source_path: str) -> list[dict] | None:
    root = os.path.realpath(str(source_path or ""))
    if not os.path.isdir(root):
        return None
    rows = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = os.path.join(directory, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            stat = os.stat(path)
            rows.append({
                "path": os.path.relpath(path, root).replace(os.sep, "/"),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    rows.sort(key=lambda item: item["path"].casefold())
    return rows


def _selected_source_fingerprint(manifest: dict) -> list[dict] | None:
    rows = []
    expected = manifest.get("source_fingerprint") or []
    if not expected or not all(isinstance(item, dict) and item.get("path") for item in expected):
        return None
    for item in expected:
        path = os.path.realpath(str(item.get("path") or ""))
        if not os.path.isfile(path):
            return None
        stat = os.stat(path)
        rows.append({
            "source": str(item.get("source") or ""),
            "path": path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    rows.sort(key=lambda item: item["path"].casefold())
    return rows


def _fingerprint_signature(rows: list[dict] | None) -> list[dict] | None:
    """Compare fast source metadata without making every status poll rehash files."""
    if not isinstance(rows, list):
        return None
    signature = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        signature.append({
            key: value
            for key, value in item.items()
            if key != "sha256"
        })
    return sorted(
        signature,
        key=lambda item: str(item.get("path") or "").casefold(),
    )


def kb_status(kb: dict) -> dict:
    manifest = _load_manifest(kb)
    processing = _is_processing(kb["id"])
    probe = {
        "present": False,
        "openable": False,
        "schema_valid": False,
        "embedding_matches": False,
        "error": "",
        "meta": {},
    }
    if kb.get("exists"):
        probe = _runtime().index.probe(kb["path"])
    summary = manifest.get("summary") or {}
    failures = manifest.get("failures") or []
    manifest_state = str(manifest.get("state") or "").strip().lower()
    is_empty = manifest_state == "empty" or (
        not kb.get("exists")
        and not kb.get("source_path")
        and not manifest
    )
    healthy = all(
        probe.get(key)
        for key in ("present", "openable", "schema_valid", "embedding_matches")
    )
    if processing:
        state = "processing"
    elif is_empty:
        state = "empty"
    elif not kb.get("exists"):
        state = "missing"
    elif not healthy:
        state = "broken"
    elif failures:
        state = "ready_with_warnings"
    else:
        state = "ready"
    expected_source = manifest.get("source_fingerprint")
    if kb.get("source_path"):
        has_external_files = any(
            isinstance(item, dict) and os.path.isabs(str(item.get("path") or ""))
            for item in (expected_source or [])
        )
        current_source = None if has_external_files else _source_fingerprint(kb.get("source_path") or "")
    else:
        current_source = _selected_source_fingerprint(manifest)
    if not isinstance(expected_source, list) or not expected_source:
        source_changed = None
        source_change_reason = "not_tracked"
    elif current_source is None:
        source_changed = True
        source_change_reason = "source_missing"
    elif _fingerprint_signature(current_source) != _fingerprint_signature(expected_source):
        source_changed = True
        source_change_reason = "source_changed"
    else:
        source_changed = False
        source_change_reason = "unchanged"
    if state == "ready" and source_changed is True:
        state = "ready_with_warnings"
    index_meta = probe.get("meta") or {}
    documents = []
    if kb.get("exists"):
        try:
            documents = _runtime().retrieval.list_documents(kb_id=kb["id"])
        except Exception:
            documents = []
    success_times = [
        int(value)
        for value in (
            manifest.get("published_at"),
            manifest.get("reindexed_at"),
            index_meta.get("built_at"),
            manifest.get("imported_at"),
        )
        if value
    ]
    checkpoint = _runtime().pipeline.checkpoint_status(kb["id"])
    usage_raw = _runtime().usage.load(kb["path"]) if kb.get("exists") else {}
    usage = IndexBuilder.usage_summary(usage_raw) if usage_raw else {}
    return {
        "id": kb["id"],
        "name": kb["name"],
        "source_path": kb.get("source_path") or "",
        "state": state,
        "source_changed": source_changed,
        "source_change_reason": source_change_reason,
        "empty": is_empty,
        "index": {
            "present": bool(probe.get("present")),
            "openable": bool(probe.get("openable")),
            "schema_valid": bool(probe.get("schema_valid")),
            "embedding_matches": bool(probe.get("embedding_matches")),
            "error": probe.get("error") or "",
        },
        "counts": {
            "documents": int(summary.get("n_docs") or summary.get("ready") or 0),
            "text_chunks": int(summary.get("text_chunks") or 0),
            "images": int(summary.get("image_assets") or 0),
            "failures": len(failures),
        },
        "last_success_at": max(success_times) if success_times else None,
        "usage": usage,
        "checkpoint": checkpoint,
        "resume_available": bool(
            checkpoint.get("available")
            and str(checkpoint.get("mode") or "import") in {"import", "add_documents"}
        ),
        "failures": failures,
        "documents": documents,
    }


def status(kb_id: str | None = None) -> dict:
    rows = [
        kb_status(kb)
        for kb in config.load_config()
        if not kb_id or kb["id"] == kb_id
    ]
    return {
        "knowledge_bases": rows,
        "configured": bool(rows),
    }


def checkpoint_status(kb_id: str) -> dict:
    return _runtime().pipeline.checkpoint_status(kb_id)


def checkpoint_inputs(kb_id: str) -> dict:
    return _runtime().pipeline.checkpoint_inputs(kb_id)


def discard_checkpoint(kb_id: str) -> dict:
    return _runtime().pipeline.discard_checkpoint(kb_id)


def search(
    query: str,
    top_k: int = 6,
    kb_id: str | None = None,
    snippet_chars: int = _SNIPPET,
    file_name: str | None = None,
    title: str | None = None,
    mode: str = "rrf",
    scope_targets: list[dict] | None = None,
) -> dict:
    return _runtime().retrieval.search(
        query,
        top_k=top_k,
        kb_id=kb_id,
        snippet_chars=snippet_chars,
        file_name=file_name,
        title=title,
        mode=mode,
        scope_targets=scope_targets,
    )


def search_diagnostics() -> list[dict]:
    return _runtime().retrieval.search_diagnostics()


def document_exists(file_name=None, title=None, kb_id=None):
    return _runtime().retrieval.document_exists(
        file_name=file_name, title=title, kb_id=kb_id
    )


def read_chunk(data_id=None, chunk_index=0, kb_id=None, ref=None, max_chars=4000):
    return _runtime().retrieval.read_chunk(
        data_id=data_id,
        chunk_index=chunk_index,
        kb_id=kb_id,
        ref=ref,
        max_chars=max_chars,
    )


def reference_for_chunk(data_id=None, chunk_index=0, kb_id=None, ref=None):
    return _runtime().retrieval.reference_for_chunk(
        data_id=data_id,
        chunk_index=chunk_index,
        kb_id=kb_id,
        ref=ref,
    )


def list_chunks(data_id=None, kb_id=None, ref=None, preview_chars=80, limit=400):
    return _runtime().retrieval.list_chunks(
        data_id=data_id,
        kb_id=kb_id,
        ref=ref,
        preview_chars=preview_chars,
        limit=limit,
    )


def read_image(data_id=None, ref_key=None, kb_id=None, source_data_id=None):
    return _runtime().retrieval.read_image(
        data_id=data_id,
        ref_key=ref_key,
        kb_id=kb_id,
        source_data_id=source_data_id,
    )


def resolve_file(cited):
    return _runtime().retrieval.resolve_file(cited)


def list_documents(kb_id=None):
    return _runtime().retrieval.list_documents(kb_id=kb_id)


def read_document(
    kb_id=None,
    data_id=None,
    file_name=None,
    ref=None,
    max_chars=200000,
):
    return _runtime().retrieval.read_document(
        kb_id=kb_id,
        data_id=data_id,
        file_name=file_name,
        ref=ref,
        max_chars=max_chars,
    )


def resolve_source_document(kb_id=None, data_id=None, file_name=None, ref=None):
    return _runtime().retrieval.resolve_source_document(
        kb_id=kb_id,
        data_id=data_id,
        file_name=file_name,
        ref=ref,
    )


def resolve_processed_document(kb_id=None, data_id=None, file_name=None, ref=None):
    return _runtime().retrieval.resolve_processed_document(
        kb_id=kb_id,
        data_id=data_id,
        file_name=file_name,
        ref=ref,
    )


def resolve_source_asset(kb_id=None, data_id=None, ref=None, image_path=None):
    return _runtime().retrieval.resolve_source_asset(
        kb_id=kb_id,
        data_id=data_id,
        ref=ref,
        image_path=image_path,
    )


def resolve_open_target(
    *,
    kb_id: str,
    data_id: str = "",
    ref: str = "",
    ref_key: str = "",
) -> str | None:
    if "::image::" in str(data_id or "") or ref_key:
        image = read_image(data_id=data_id, ref_key=ref_key, kb_id=kb_id)
        path = str(image.get("image_abspath") or "")
        return path if path and os.path.isfile(path) else None
    source = resolve_source_document(kb_id=kb_id, data_id=data_id, ref=ref)
    path = str(source.get("path") or "")
    return path if path and os.path.isfile(path) else None


def reset_managed_data() -> dict:
    """Delete all application-managed KB data and registry entries.

    This intentionally has no source-directory cleanup and exists for the
    pre-production layout reset requested for this refactor.
    """
    runtime = _runtime()
    removed = []
    for kb in list(config.load_config()):
        removed.append(runtime.pipeline.delete(kb["id"]))
    if os.path.isdir(config.DATA_ROOT):
        for name in os.listdir(config.DATA_ROOT):
            root = os.path.realpath(config.DATA_ROOT)
            unresolved = os.path.join(root, name)
            if os.path.islink(unresolved):
                continue
            candidate = os.path.realpath(unresolved)
            if (
                candidate != root
                and os.path.commonpath((root, candidate)) == root
            ):
                remove_tree(candidate)
    config._dump_raw_config({"knowledge_base": {}}, config.CONFIG_PATH)
    return {"ok": True, "removed": removed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GenericAgent knowledge base")
    parser.add_argument("--import", dest="import_dir", metavar="DIR")
    parser.add_argument("--name")
    parser.add_argument("--reindex", metavar="KB_ID")
    parser.add_argument("--delete", metavar="KB_ID")
    parser.add_argument("--reset-managed-data", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--search", metavar="QUERY")
    parser.add_argument("--kb")
    parser.add_argument("--mode", choices=("rrf", "vector", "sparse"), default="rrf")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args(argv)

    if args.reset_managed_data:
        print(json.dumps(reset_managed_data(), ensure_ascii=False, indent=2))
        return 0
    if args.import_dir:
        print(json.dumps(import_kb(args.import_dir, name=args.name or ""), ensure_ascii=False, indent=2))
        return 0
    if args.reindex:
        print(json.dumps(reindex(args.reindex), ensure_ascii=False, indent=2))
        return 0
    if args.delete:
        print(json.dumps(delete_kb(args.delete), ensure_ascii=False, indent=2))
        return 0
    if args.status:
        print(json.dumps(status(args.kb), ensure_ascii=False, indent=2))
        return 0
    if args.search:
        print(json.dumps(
            search(args.search, top_k=args.top_k, kb_id=args.kb, mode=args.mode),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
