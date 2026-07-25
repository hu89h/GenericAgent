#!/usr/bin/env python3
"""GenericAgent unified Markdown/image KB engine with a Zvec hybrid index.

设计要点：
- 读取知识库文件夹下的 Markdown 图书；
- 抽取正文 → 分块 → 构建 Zvec dense 语义向量索引与 sparse 关键词向量索引；
- GA 运行时检索和读取依赖 Zvec，不依赖外部项目的 domain_sop/分词链；
- 每个知识库的 zvec 集合与 zvec_meta.json 放在该知识库路径下的 `.kb_index/`；
- 增量构建：源文件 mtime+size 指纹未变则跳过；
- search() 返回「命中定位 + 命中内容」，read_chunk() 精确取单个 chunk 原文用于补充核对；
- preload=true 的库可生成「知识库目录概览」注入上下文。

CLI：
    python -m knowledge_base.backend --build           # 增量构建所有配置库
    python -m knowledge_base.backend --rebuild         # 强制全量重建
    python -m knowledge_base.backend --status          # 查看各库状态
    python -m knowledge_base.backend --search "关键词"  # 检索测试
"""
from __future__ import annotations
import os
import re
import sys
import json
import shutil
import threading
from urllib.parse import unquote

try:
    from .documents import (
        chunk_document_records,
        chunking_meta as _chunking_meta,
        extract_text,
        fingerprint as _fingerprint,
        read_textfile as _read_textfile,
        scan_documents as _scan,
    )
except ImportError:  # pragma: no cover - supports direct CLI execution
    from documents import (
        chunk_document_records,
        chunking_meta as _chunking_meta,
        extract_text,
        fingerprint as _fingerprint,
        read_textfile as _read_textfile,
        scan_documents as _scan,
    )

try:
    from .config import (
        CONFIG_PATH,
        DATA_ROOT,
        ROOT,
        canonical_source_path as _configured_canonical_source_path,
        kb_id_for_source as _configured_kb_id_for_source,
        kb_by_id as _kb_by_id,
        load_config,
        remove_kb,
        upsert_kb,
    )

    from .usage import UsageTracker as _UsageTracker
    from .assets import ImageAssetProcessor as _ImageAssetProcessor
    from .build import (
        BuildCoordinator as _BuildCoordinator,
        BuildServices as _BuildServices,
        DocumentServices as _DocumentServices,
        ImageServices as _ImageServices,
        IndexServices as _IndexServices,
        UsageServices as _UsageServices,
    )
    from .retrieval import KnowledgeBaseRetriever as _KnowledgeBaseRetriever
    from .zvec import ZvecIndex as _ZvecIndex
except ImportError:  # pragma: no cover - supports direct CLI execution
    from config import (
        CONFIG_PATH,
        DATA_ROOT,
        ROOT,
        canonical_source_path as _configured_canonical_source_path,
        kb_id_for_source as _configured_kb_id_for_source,
        kb_by_id as _kb_by_id,
        load_config,
        remove_kb,
        upsert_kb,
    )

    from usage import UsageTracker as _UsageTracker
    from assets import ImageAssetProcessor as _ImageAssetProcessor
    from build import (
        BuildCoordinator as _BuildCoordinator,
        BuildServices as _BuildServices,
        DocumentServices as _DocumentServices,
        ImageServices as _ImageServices,
        IndexServices as _IndexServices,
        UsageServices as _UsageServices,
    )
    from retrieval import KnowledgeBaseRetriever as _KnowledgeBaseRetriever
    from zvec import ZvecIndex as _ZvecIndex

DEFAULT_KB_ID = "default_kb"
INDEX_SUBDIR = ".kb_index"          # 放在每个知识库路径下
ZVEC_SUBDIR = "zvec"                # 向量索引目录：.kb_index/zvec/
IMAGE_CACHE_SUBDIR = "image_cache"
IMAGE_ASSETS_FILE = "image_assets.json"
DOCUMENT_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tif", ".tiff",
})
BUILD_USAGE_FILE = "build_usage.json"
ZVEC_SCHEMA_VERSION = 9
_SNIPPET = 220
ZVEC_DIM = int(os.environ.get("GA_KB_ZVEC_DIM", os.environ.get("GA_KB_EMBED_DIM", "1024")))
ZVEC_BATCH = int(os.environ.get("GA_KB_ZVEC_BATCH", "256"))
ZVEC_QUERY_FACTOR = max(1, int(os.environ.get("GA_KB_ZVEC_QUERY_FACTOR", "4")))
ZVEC_VECTOR_WEIGHT = float(os.environ.get("GA_KB_VECTOR_WEIGHT", "1.2"))
ZVEC_SPARSE_WEIGHT = float(os.environ.get("GA_KB_SPARSE_WEIGHT", "1.0"))
IMAGE_ANALYSIS_CONCURRENCY = max(1, int(os.environ.get("GA_KB_IMAGE_CONCURRENCY", "1")))

_local = threading.local()
_build_lock = threading.Lock()
_build_state_lock = threading.Lock()
_LOCK_PORT = 45764                  # 跨进程单飞锁端口（与 indexer 45763 / scheduler 45762 错开）
# 构建态：供前端/服务端查询进度
build_state = {
    "running": False,
    "result": None,
    "last_build_summary": None,
    "started_at": None,
    "finished_at": None,
    "kb": None,
    "phase": "idle",
    "message": "",
    "processed": 0,
    "total": 0,
}


def _update_build_state(**changes):
    with _build_state_lock:
        build_state.update(changes)


def _build_state_snapshot():
    with _build_state_lock:
        return dict(build_state)


def clear_last_hits():
    _local.last_hits = []


def get_last_hits():
    return list(getattr(_local, "last_hits", []) or [])


# Knowledge-base registry functions are implemented in config.py.

def _managed_kb_root(kb):
    """Return the package-local import root for *kb*, or ``None`` for external paths."""
    if not isinstance(kb, dict):
        return None
    data_root = os.path.realpath(DATA_ROOT)
    candidate = os.path.realpath(os.path.join(DATA_ROOT, str(kb.get("id") or "")))
    kb_path = os.path.realpath(str(kb.get("path") or ""))
    if not candidate or candidate == data_root:
        return None
    try:
        if os.path.commonpath((data_root, candidate)) != data_root:
            return None
        if os.path.commonpath((candidate, kb_path)) != candidate:
            return None
    except ValueError:
        return None
    return candidate


def delete_kb(kb_id, delete_data=False, config_path=CONFIG_PATH):
    """Remove a KB registration and optionally its package-local imported copy.

    External source directories are never removed.  ``delete_data=True`` only
    removes ``DATA_ROOT/<kb_id>`` when the configured path is inside that exact
    package-local import root.
    """
    kb_id = str(kb_id or "").strip()
    if not kb_id:
        raise ValueError("知识库 ID 不能为空")
    if not _build_lock.acquire(blocking=False):
        raise RuntimeError("知识库正在构建索引，请稍后再删除")
    try:
        kb = next((item for item in load_config(config_path) if item["id"] == kb_id), None)
        if kb is None:
            return {"removed": False, "kb_id": kb_id, "data_deleted": False}

        managed_root = _managed_kb_root(kb)
        data_deleted = False
        # Search and Agent tool calls may have opened the same Zvec collection
        # from worker threads.  Release those native handles before removing
        # the package-local index, especially on Windows where IPC files cannot
        # be unlinked while a collection is still alive.
        _clear_zvec_cache(_zvec_path(kb["path"]))
        if delete_data and managed_root and os.path.lexists(managed_root):
            # Do not follow a symlink/junction supplied through a package-local path.
            if not os.path.islink(managed_root):
                shutil.rmtree(managed_root)
                data_deleted = True

        removed = remove_kb(kb_id, config_path=config_path)
        if _retrieval_instance is not None:
            _retrieval_instance.clear_asset_cache(kb.get("path"))
        return {
            "removed": bool(removed),
            "kb_id": kb_id,
            "data_deleted": data_deleted,
            "managed_data": bool(managed_root),
        }
    finally:
        _build_lock.release()


def _canonical_source_path(source_dir):
    """Return the normalized absolute source path used for KB identity."""
    return _configured_canonical_source_path(source_dir)


def kb_id_for_source(source_dir):
    """Return a stable, package-safe ID derived from a source directory path."""
    return _configured_kb_id_for_source(source_dir)


def import_kb(source_dir, kb_id="", name="", overwrite=False, progress=None):
    """Import one source directory through the unified MinerU pipeline.

    ``kb_id`` remains in the function signature for bridge compatibility, but
    the registry identity is deliberately derived from the canonical source
    path.  PDFs over the current service limit are rejected before MinerU
    submission; this importer does not split or merge PDF parts.
    """
    if not _build_lock.acquire(blocking=False):
        raise RuntimeError("知识库正在构建索引，请稍后再导入")
    try:
        try:
            from .importer import import_knowledge_base
        except ImportError:  # pragma: no cover - supports direct CLI execution
            from importer import import_knowledge_base
        return import_knowledge_base(
            source_dir,
            kb_id=kb_id,
            name=name,
            overwrite=overwrite,
            progress=progress,
        )
    finally:
        _build_lock.release()


def _index_dir(kb_path):
    return os.path.join(kb_path, INDEX_SUBDIR)


def _zvec_path(kb_path):
    return os.path.join(_index_dir(kb_path), ZVEC_SUBDIR)


def _zvec_meta_path(kb_path):
    return os.path.join(_index_dir(kb_path), "zvec_meta.json")


def _image_cache_dir(kb_path):
    return os.path.join(_index_dir(kb_path), IMAGE_CACHE_SUBDIR)


def _image_assets_path(kb_path):
    return os.path.join(_index_dir(kb_path), IMAGE_ASSETS_FILE)


def _build_usage_path(kb_path):
    return os.path.join(_index_dir(kb_path), BUILD_USAGE_FILE)


# ─────────────────────────── 文档扫描与抽取 ───────────────────────────

def _build_sources(kb_path, scanned, mode="full", image_indexes=None):
    sources = {
        "documents": _fingerprint(scanned),
        "chunking": _chunking_meta(),
    }
    if mode != "text":
        if image_indexes is None:
            raise ValueError("构建图片来源指纹需要预先生成的文档图片索引")
        sources.update({
            "images": _image_source_fingerprint(
                kb_path, scanned, image_indexes=image_indexes
            ),
            "image_analysis": _image_build_fingerprint_meta(),
        })
    return sources


_usage_tracker_instance = None


def _usage_tracker():
    global _usage_tracker_instance
    if _usage_tracker_instance is None:
        _usage_tracker_instance = _UsageTracker(
            image_meta_fn=_image_analysis_meta,
            embedding_meta_fn=_embedding_meta,
            sparse_embedding_meta_fn=_sparse_embedding_meta,
            embedding_provider_fn=_embedding_provider,
            index_dir_fn=_index_dir,
            usage_path_fn=_build_usage_path,
        )
    return _usage_tracker_instance


def _empty_usage(kb_id="", kb_path=""):
    return _usage_tracker().empty(kb_id, kb_path)


def _usage():
    return _usage_tracker().current()


def _set_usage(value):
    _usage_tracker().set_current(value)


def _add_model_usage(model, usage, output_chars=0):
    _usage_tracker().add_model_usage(model, usage, output_chars)


def _model_usage_delta(model, usage, output_chars=0):
    return _usage_tracker().model_usage_delta(model, usage, output_chars)


def _merge_image_analysis_usage(usage_delta):
    _usage_tracker().merge_image_analysis(usage_delta)


def _write_build_usage(kb_path, usage):
    _usage_tracker().write(kb_path, usage)


def _load_build_usage(kb_path):
    return _usage_tracker().load(kb_path)


def _calculate_build_cost(usage):
    return _usage_tracker().calculate_cost(usage)


def _build_usage_summary(usage):
    return {
        "image_calls": usage.get("image_analysis", {}).get("calls", 0),
        "image_cached": usage.get("image_analysis", {}).get("cached", 0),
        "image_models": usage.get("image_analysis", {}).get("models", {}),
        "image_cached_models": usage.get("image_analysis", {}).get("cached_models", {}),
        "embedding_calls": usage.get("embedding", {}).get("calls", 0),
        "embedding_estimated_input_tokens": usage.get("embedding", {}).get("estimated_input_tokens", 0),
        "sparse_embedding_calls": usage.get("sparse_embedding", {}).get("calls", 0),
        "sparse_embedding_estimated_input_tokens": usage.get("sparse_embedding", {}).get("estimated_input_tokens", 0),
        "cost": _calculate_build_cost(usage),
    }


# ───────────────────────────── 图片资产 ─────────────────────────────

def _truthy_env(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _load_image_client():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from .providers import vision
    except ImportError:  # pragma: no cover - supports direct CLI execution
        from providers import vision
    return vision


def _image_analysis_meta():
    """Full analysis meta (incl. runtime_image_qa) for asset-file display."""
    try:
        return _load_image_client().analysis_meta()
    except Exception as e:
        return {
            "enabled": _truthy_env("GA_KB_IMAGE_ANALYSIS"),
            "error": str(e),
            "prompt_version": int(os.environ.get("GA_KB_IMAGE_PROMPT_VERSION", "6")),
        }


def _image_build_fingerprint_meta():
    """Build/index fingerprint meta — excludes the query-time runtime_image_qa
    switch so toggling it no longer forces a full re-index (bug M2)."""
    try:
        return _load_image_client().build_analysis_meta()
    except Exception as e:
        return {
            "enabled": _truthy_env("GA_KB_IMAGE_ANALYSIS"),
            "error": str(e),
            "prompt_version": int(os.environ.get("GA_KB_IMAGE_PROMPT_VERSION", "6")),
        }










# ───────────────────────────── 图片资产适配 ─────────────────────────────

_image_assets_instance = None


def _image_assets():
    global _image_assets_instance
    if _image_assets_instance is None:
        _image_assets_instance = _ImageAssetProcessor(
            image_client_fn=_load_image_client,
            image_meta_fn=_image_analysis_meta,
            image_cache_dir_fn=_image_cache_dir,
            image_assets_path_fn=_image_assets_path,
            index_dir_fn=_index_dir,
        merge_usage_fn=_merge_image_analysis_usage,
            model_usage_delta_fn=_model_usage_delta,
            concurrency=IMAGE_ANALYSIS_CONCURRENCY,
        )
    return _image_assets_instance


def _local_ref_key(value):
    return _image_assets().local_ref_key(value)


def _build_document_index(text):
    return _image_assets().build_document_index(text)


def _asset_body(asset):
    return _image_assets().asset_body(asset)


def _write_image_assets(kb_path, assets, validation=None, pending=False):
    _image_assets().write_assets(kb_path, assets, validation=validation, pending=pending)
    # A pending write does not change the final file the retriever reads,
    # so only invalidate the retrieval asset cache on a direct/final write.
    if not pending and _retrieval_instance is not None:
        _retrieval_instance.clear_asset_cache(kb_path)


def _commit_pending_image_assets(kb_path):
    committed = _image_assets().commit_pending_assets(kb_path)
    if committed and _retrieval_instance is not None:
        _retrieval_instance.clear_asset_cache(kb_path)
    return committed


def _discard_pending_image_assets(kb_path):
    _image_assets().discard_pending_assets(kb_path)


def _load_image_assets(kb_path):
    return _image_assets().load_assets(kb_path)


def _load_image_assets_build(kb_path):
    # Build/index-time count reads the not-yet-committed pending assets so
    # meta stats reflect the assets that will be published together with
    # this index generation.  Retrieval keeps using the final-only loader.
    return _image_assets().load_assets(kb_path, prefer_pending=True)


def _image_source_fingerprint(kb_path, scanned, image_indexes=None):
    return _image_assets().image_source_fingerprint(kb_path, scanned, image_indexes=image_indexes)


def _image_records_for_document(*args, **kwargs):
    return _image_assets().image_records_for_document(*args, **kwargs)


def _apply_image_analysis(asset, analysis):
    return _image_assets().apply_image_analysis(asset, analysis)


def _analyze_image_jobs(kb, image_jobs, log):
    return _image_assets().analyze_image_jobs(kb, image_jobs, log)


# ───────────────────────────── 构建 ─────────────────────────────

_build_coordinator_instance = None


def _build_coordinator():
    global _build_coordinator_instance
    if _build_coordinator_instance is None:
        _build_coordinator_instance = _BuildCoordinator(
            _BuildServices(
                documents=_DocumentServices(
                    scan_documents=_scan,
                    extract_text=extract_text,
                    chunk_document_records=chunk_document_records,
                    build_sources=_build_sources,
                    display_names=_imported_document_titles,
                ),
                images=_ImageServices(
                    build_document_index=_build_document_index,
                    image_records_for_document=_image_records_for_document,
                    analyze_image_jobs=_analyze_image_jobs,
                    apply_image_analysis=_apply_image_analysis,
                    write_image_assets=_write_image_assets,
                    load_image_assets=_load_image_assets,
                    commit_pending_image_assets=_commit_pending_image_assets,
                    discard_pending_image_assets=_discard_pending_image_assets,
                ),
                index=_IndexServices(
                    zvec_path=_zvec_path,
                    require_zvec=_require_zvec,
                    zvec_meta=_zvec_meta,
                    zvec_is_quickly_fresh=_zvec_is_quickly_fresh,
                    build_zvec_index=_build_zvec_index,
                    append_zvec_image_index=_append_zvec_image_index,
                ),
                usage=_UsageServices(
                    set_usage=_set_usage,
                    empty_usage=_empty_usage,
                    usage=_usage,
                    write_build_usage=_write_build_usage,
                    build_usage_summary=_build_usage_summary,
                ),
                update_build_state=_update_build_state,
                load_config=load_config,
            ),
            lock_port=_LOCK_PORT,
            build_lock=_build_lock,
        )
    return _build_coordinator_instance


def _records(
    kb,
    scanned,
    log,
    include_text=True,
    include_images=True,
    progressfn=None,
    image_indexes=None,
):
    return _build_coordinator().records(
        kb,
        scanned,
        log,
        include_text=include_text,
        include_images=include_images,
        progressfn=progressfn,
        image_indexes=image_indexes,
    )


def build_kb(kb, force=False, verbose=True, logfn=None, mode="full"):
    """Compatibility facade for the build coordinator."""
    return _build_coordinator().build_kb(
        kb, force=force, verbose=verbose, logfn=logfn, mode=mode
    )


def _build_summary(results):
    return _BuildCoordinator.build_summary(results)


def build_all(force=False, verbose=True, logfn=None, kb_id=None, mode="full"):
    """Compatibility facade for building configured knowledge bases."""
    return _build_coordinator().build_all(
        force=force, verbose=verbose, logfn=logfn, kb_id=kb_id, mode=mode
    )


# ───────────────────────────── 检索 ─────────────────────────────

_zvec_store_instance = None
_zvec_store_lock = threading.RLock()


def _configured_embedding_dimension() -> int:
    """Return the active embedding dimension used by both provider and Zvec."""
    try:
        from .providers import provider_settings
    except ImportError:  # pragma: no cover - supports direct CLI execution
        from providers import provider_settings
    try:
        value = provider_settings.embedding_config().get("dimension")
        return max(1, int(value or ZVEC_DIM))
    except (TypeError, ValueError):
        return ZVEC_DIM


def _zvec_store():
    global _zvec_store_instance
    dimension = _configured_embedding_dimension()
    with _zvec_store_lock:
        if _zvec_store_instance is None or _zvec_store_instance.dimension != dimension:
            if _zvec_store_instance is not None:
                _zvec_store_instance.clear_cache()
            _zvec_store_instance = _ZvecIndex(
                dimension=dimension,
                batch_size=ZVEC_BATCH,
                schema_version=ZVEC_SCHEMA_VERSION,
                path_fn=_zvec_path,
                meta_path_fn=_zvec_meta_path,
                embedding_fn=_embed_texts,
                sparse_embedding_fn=_embed_sparse_texts,
                embedding_meta_fn=_embedding_meta,
                sparse_embedding_meta_fn=_sparse_embedding_meta,
                chunking_meta_fn=_chunking_meta,
                image_analysis_meta_fn=_image_build_fingerprint_meta,
                usage_fn=_usage,
                load_assets_fn=_load_image_assets_build,
                document_fingerprint_fn=_fingerprint,
            )
    return _zvec_store_instance


def _zvec_conn(path, *, create=False, read_only=True):
    return _zvec_store().connect(path, create=create, read_only=read_only)


def _clear_zvec_cache(path=None):
    return _zvec_store().clear_cache(path)


def _embed_texts(texts):
    return _embed_texts_with_provider(texts)


def _embed_sparse_texts(texts, text_type="document"):
    client = _load_embeddings_provider()
    return client.embed_sparse_texts(texts, text_type=text_type)


def _embedding_provider():
    return "dashscope"


def _embedding_meta():
    provider = _embedding_provider()
    meta = {"provider": provider, "dimension": _configured_embedding_dimension()}
    try:
        client = _load_embeddings_provider()
        return client.embedding_meta()
    except Exception as e:
        meta["error"] = str(e)
    return meta


def _sparse_embedding_meta():
    provider = _embedding_provider()
    meta = {"provider": provider, "type": "sparse"}
    try:
        client = _load_embeddings_provider()
        return client.sparse_embedding_meta()
    except Exception as e:
        meta["error"] = str(e)
    return meta


def _load_embeddings_provider():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from .providers import embeddings
    except ImportError:  # pragma: no cover - supports direct CLI execution
        from providers import embeddings
    return embeddings


def _embed_texts_with_provider(texts):
    client = _load_embeddings_provider()
    return client.embed_texts(texts)


def _zvec_meta(kb_path):
    return _zvec_store().meta(kb_path)


def _zvec_is_quickly_fresh(kb_path, scanned, meta=None, mode="full"):
    return _zvec_store().is_quickly_fresh(kb_path, scanned, meta=meta, mode=mode)


def _zvec_doc_id(data_id, chunk_index):
    return _ZvecIndex.doc_id(data_id, chunk_index)


def _record_search_error(kb, source, error):
    errors = getattr(_local, "search_errors", None)
    if errors is not None:
        errors.append({
            "kb_id": kb.get("id", ""),
            "source": source,
            "error": str(error),
        })


def search_diagnostics():
    """Return non-fatal errors from the latest search in this thread."""
    return list(getattr(_local, "search_errors", []) or [])


def _require_zvec():
    return _zvec_store().require()


def _build_zvec_index(kb, records, sources, force=False, logfn=None):
    return _zvec_store().build(kb, records, sources, force=force, logfn=logfn)


def _append_zvec_image_index(kb, records, sources, force=False, logfn=None):
    return _zvec_store().append_images(kb, records, sources, force=force, logfn=logfn)


_ZVEC_OUTPUT_FIELDS = [
    "data_id", "chunk_index", "kb_id", "file_name", "title", "kind",
    "image_path", "source_data_id", "source_chunk_index", "header_path", "body",
]










def _zvec_fetch_doc(kb, data_id, chunk_index, output_fields=None):
    return _zvec_store().fetch(
        kb, data_id, chunk_index, output_fields=output_fields or _ZVEC_OUTPUT_FIELDS
    )


# ───────────────────────── 状态 / 预加载上下文 ─────────────────────────

def _imported_file_counts(kb_path):
    """Read source-level document/asset counts from the import manifest."""
    manifest = os.path.join(os.path.dirname(kb_path), "import_manifest.json")
    try:
        with open(manifest, encoding="utf-8") as handle:
            entries = (json.load(handle) or {}).get("files") or []
    except Exception:
        return None
    documents = sum(
        1 for item in entries
        if isinstance(item, dict)
        and item.get("kind") == "document"
        and item.get("status") == "ready"
    )
    assets = sum(
        1 for item in entries
        if isinstance(item, dict) and item.get("kind") == "asset"
    )
    return {"documents": documents, "assets": assets}


def kb_status(kb):
    zpath = _zvec_path(kb["path"])
    zm = _zvec_meta(kb["path"])
    zvec_ready = os.path.isdir(zpath)
    index_meta = zm
    info = {"id": kb["id"], "name": kb["name"], "path": kb["path"],
            "raw_path": kb.get("raw_path", kb["path"]),
            "preload": kb["preload"], "exists": kb["exists"],
            "ready": bool(index_meta and zvec_ready), "n_docs": 0, "n_chunks": 0,
            "image_assets": 0,
            "built_at": index_meta.get("built_at") if index_meta else None, "up_to_date": None,
            "zvec_ready": zvec_ready, "zvec_status": None,
            "index_backend": "zvec" if zvec_ready else None,
            "embedding": None}
    if not kb["exists"]:
        return info
    imported_counts = _imported_file_counts(kb["path"])
    scanned = None
    try:
        scanned = _scan(kb["path"])
    except Exception:
        pass
    if imported_counts is not None:
        info["n_docs"] = imported_counts["documents"]
        info["image_assets"] = imported_counts["assets"]
    elif scanned is not None:
        info["n_docs"] = len(scanned)
    if info["ready"]:
        st = index_meta.get("stats", {})
        info["n_docs"] = st.get("n_docs", info["n_docs"])
        info["n_chunks"] = st.get("n_chunks", 0)
        info["zvec_status"] = "ready"
        info["embedding"] = index_meta.get("embedding")
        info["sparse_embedding"] = index_meta.get("sparse_embedding")
        info["zvec_chunks"] = st.get("n_chunks", 0)
        info["image_assets"] = st.get("image_assets", info["image_assets"])
        if info["image_assets"] is None:
            info["image_assets"] = len(_load_image_assets(kb["path"]))
        info["zvec_bytes"] = st.get("zvec_bytes", 0)
        bu = _load_build_usage(kb["path"])
        if bu:
            ia = bu.get("image_analysis", {})
            em = bu.get("embedding", {})
            sem = bu.get("sparse_embedding", {})
            info["usage"] = {
                "image_calls": ia.get("calls", 0),
                "image_cached": ia.get("cached", 0),
                "image_failed": ia.get("failed", 0),
                "image_models": ia.get("models", {}),
                "image_cached_models": ia.get("cached_models", {}),
                "embedding_calls": em.get("calls", 0),
                "embedding_texts": em.get("texts", 0),
                "embedding_estimated_input_tokens": em.get("estimated_input_tokens", 0),
                "sparse_embedding_calls": sem.get("calls", 0),
                "sparse_embedding_texts": sem.get("texts", 0),
                "sparse_embedding_estimated_input_tokens": sem.get("estimated_input_tokens", 0),
                "cost": bu.get("cost"),
            }
        try:
            if scanned is None:
                scanned = _scan(kb["path"])
            info["up_to_date"] = _zvec_is_quickly_fresh(kb["path"], scanned, zm)
        except Exception:
            info["up_to_date"] = None
    return info


def status():
    kbs = [kb_status(kb) for kb in load_config()]
    state = _build_state_snapshot()
    return {
        "knowledge_bases": kbs,
        "building": state["running"],
        "build_kb": state.get("kb"),
        "last_build": state.get("result"),
        "last_build_summary": state.get("last_build_summary"),
        "build_progress": {
            "phase": state.get("phase"),
            "message": state.get("message", ""),
            "processed": state.get("processed", 0),
            "total": state.get("total", 0),
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
        },
        "configured": bool(kbs),
    }


def _folder_breakdown(kb, limit=8):
    """按顶层目录统计文档数（用于 preload 概览）。"""
    counts = {}
    for rel, _ap, _st in _scan(kb["path"]):
        rel = rel.replace("\\", "/")
        folder = rel.split("/", 1)[0] if "/" in rel else "(根目录)"
        counts[folder] = counts.get(folder, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def preload_context():
    """为 preload=true 的知识库生成「知识库目录概览」，在开启会话时注入上下文。"""
    kbs = [kb for kb in load_config() if kb.get("preload")]
    ready = [kb for kb in kbs if os.path.isdir(_zvec_path(kb["path"]))]
    if not kbs:
        return ""
    lines = ["[本地知识库已加载]（preload）"]
    if not ready:
        if _build_state_snapshot()["running"]:
            lines.append("索引正在后台构建中，稍候即可检索。")
        else:
            lines.append("索引尚未就绪（可在客户端「知识库」重建，或运行 python -m knowledge_base.backend --build）。")
    total_docs = 0
    for kb in ready:
        m = _zvec_meta(kb["path"])
        nd = m.get("stats", {}).get("n_docs", 0)
        total_docs += nd
        fb = _folder_breakdown(kb)
        fb_s = "；".join(f"{name}({cnt})" for name, cnt in fb)
        lines.append(f"- {kb['id']}：{kb['path']}（{nd} 篇文档）" + (f"｜主要目录：{fb_s}" if fb_s else ""))
    if ready:
        lines.append(f"合计约 {total_docs} 篇文档。回答前服务器会自动注入「[本地知识库命中]」检索结果；")
        lines.append("如需进一步核对原文，可在 code_run 中：")
        lines.append("  from knowledge_base import backend")
        lines.append('  backend.search("关键词", top_k=5)  # 检索；backend.read_chunk(data_id, chunk_index)  # 读原文核对')
    return "\n".join(lines)


def is_preload_enabled():
    return any(kb.get("preload") for kb in load_config())


def _imported_document_titles(kb_path):
    """Map generated Markdown paths back to source filenames when present."""
    titles = {}
    for entry in _imported_document_entries(kb_path):
        source = str(entry.get("source") or "")
        title = os.path.basename(source) or source
        for rel in entry.get("processed") or []:
            normalized = str(rel).replace("\\", "/").lstrip("/")
            if not normalized:
                continue
            titles[normalized] = title
            if normalized.startswith("processed/"):
                titles[normalized[len("processed/"):]] = title
            else:
                titles[f"processed/{normalized}"] = title
    return titles


def _imported_document_entries(kb_path):
    """Return manifest document entries without exposing source file paths."""
    manifest = os.path.join(os.path.dirname(kb_path), "import_manifest.json")
    try:
        with open(manifest, encoding="utf-8") as handle:
            entries = (json.load(handle) or {}).get("files") or []
    except Exception:
        return []
    return [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get("kind") == "document"
    ]


_retrieval_instance = None


def _retrieval():
    global _retrieval_instance
    if _retrieval_instance is None:
        _retrieval_instance = _KnowledgeBaseRetriever(
            load_config=load_config,
            kb_by_id=_kb_by_id,
            scan_documents=_scan,
            read_textfile=_read_textfile,
            zvec_path=_zvec_path,
            zvec_conn=_zvec_conn,
            zvec_doc_id=_zvec_doc_id,
            fetch_doc=_zvec_fetch_doc,
            require_zvec=_require_zvec,
            embed_texts=_embed_texts,
            embed_sparse_texts=_embed_sparse_texts,
            load_image_assets=_load_image_assets,
            local_ref_key=_local_ref_key,
            asset_body=_asset_body,
            record_search_error=_record_search_error,
            imported_document_titles=_imported_document_titles,
            imported_document_entries=_imported_document_entries,
            query_factor=ZVEC_QUERY_FACTOR,
            vector_weight=ZVEC_VECTOR_WEIGHT,
            sparse_weight=ZVEC_SPARSE_WEIGHT,
            snippet_width=_SNIPPET,
            output_fields=_ZVEC_OUTPUT_FIELDS,
        )
    return _retrieval_instance


def search(query, top_k=6, kb_id=None, snippet_chars=_SNIPPET, file_name=None, title=None, mode="rrf"):
    _local.search_errors = []
    results = _retrieval().search(
        query, top_k=top_k, kb_id=kb_id, snippet_chars=snippet_chars,
        file_name=file_name, title=title, mode=mode,
    )
    _local.last_hits = results
    return results


def document_exists(file_name=None, title=None, kb_id=None):
    return _retrieval().document_exists(file_name=file_name, title=title, kb_id=kb_id)


def read_chunk(data_id=None, chunk_index=0, kb_id=None, ref=None, max_chars=4000):
    return _retrieval().read_chunk(
        data_id=data_id, chunk_index=chunk_index, kb_id=kb_id,
        ref=ref, max_chars=max_chars,
    )


def reference_for_chunk(data_id=None, chunk_index=0, kb_id=None, ref=None):
    return _retrieval().reference_for_chunk(
        data_id=data_id, chunk_index=chunk_index, kb_id=kb_id, ref=ref,
    )


def list_chunks(data_id=None, kb_id=None, ref=None, preview_chars=80, limit=400):
    return _retrieval().list_chunks(
        data_id=data_id, kb_id=kb_id, ref=ref,
        preview_chars=preview_chars, limit=limit,
    )


def read_image(image_id=None, image_path=None, data_id=None, ref=None, kb_id=None, ref_key=None, query=None):
    return _retrieval().read_image(
        image_id=image_id, image_path=image_path, data_id=data_id, ref=ref,
        kb_id=kb_id, ref_key=ref_key, query=query,
    )


def resolve_file(cited):
    return _retrieval().resolve_file(cited)


def list_documents(kb_id=None):
    return _retrieval().list_documents(kb_id=kb_id)


def read_document(kb_id=None, data_id=None, file_name=None, ref=None, max_chars=200000):
    return _retrieval().read_document(
        kb_id=kb_id, data_id=data_id, file_name=file_name,
        ref=ref, max_chars=max_chars,
    )


def resolve_source_document(kb_id=None, data_id=None, file_name=None, ref=None):
    return _retrieval().resolve_source_document(
        kb_id=kb_id, data_id=data_id, file_name=file_name, ref=ref,
    )


def _path_is_within(root, path):
    try:
        root = os.path.realpath(root)
        path = os.path.realpath(path)
        return path == root or os.path.commonpath((root, path)) == root
    except (OSError, ValueError):
        return False


def resolve_document_asset(kb_id=None, data_id=None, image_path=None):
    """Resolve a Markdown-relative image under one validated KB document."""
    data_id = str(data_id or "").strip()
    kid = str(kb_id or "").strip()
    if "::" not in data_id:
        return None
    data_kb_id, document_rel = data_id.split("::", 1)
    if kid and data_kb_id != kid:
        return None
    kb = _kb_by_id(kid or data_kb_id)
    if kb is None or not kb.get("exists"):
        return None

    raw_image = str(image_path or "").strip().strip("<>")
    raw_image = unquote(raw_image.split("?", 1)[0].split("#", 1)[0]).replace("\\", "/")
    if (
        not raw_image
        or raw_image.startswith("/")
        or re.match(r"^[a-z][a-z0-9+.-]*:", raw_image, re.I)
        or os.path.splitext(raw_image)[1].lower() not in DOCUMENT_IMAGE_EXTS
    ):
        return None

    root = os.path.realpath(kb["path"])
    document = os.path.realpath(os.path.join(root, document_rel.replace("/", os.sep)))
    if not _path_is_within(root, document) or not os.path.isfile(document):
        return None
    target = os.path.realpath(
        os.path.join(os.path.dirname(document), raw_image.replace("/", os.sep))
    )
    if not _path_is_within(root, target) or not os.path.isfile(target):
        return None
    return target


def resolve_source_asset(kb_id=None, data_id=None, ref=None, image_path=None):
    """Resolve an image linked from an original source document."""
    source = resolve_source_document(kb_id=kb_id, data_id=data_id, ref=ref)
    if source.get("error") or not source.get("is_original"):
        return None
    kb = _kb_by_id(source.get("kb_id") or kb_id)
    source_root_value = str((kb or {}).get("source_path") or "").strip()
    if kb is None or not source_root_value:
        return None
    raw_image = str(image_path or "").strip().strip("<>")
    raw_image = unquote(raw_image.split("?", 1)[0].split("#", 1)[0]).replace("\\", "/")
    if (
        not raw_image
        or raw_image.startswith("/")
        or re.match(r"^[a-z][a-z0-9+.-]*:", raw_image, re.I)
        or os.path.splitext(raw_image)[1].lower() not in DOCUMENT_IMAGE_EXTS
    ):
        return None
    source_root = os.path.realpath(source_root_value)
    target = os.path.realpath(
        os.path.join(os.path.dirname(source["path"]), raw_image.replace("/", os.sep))
    )
    if not _path_is_within(source_root, target) or not os.path.isfile(target):
        return None
    return target


def build(kb_id=None, force=False, mode="full", logfn=None):
    """Desktop bridge wrapper around the original same-stack Zvec builder."""
    results = build_all(force=force, verbose=not callable(logfn), logfn=logfn, kb_id=kb_id, mode=mode)
    summary = _build_summary(results)
    return {
        "ok": True,
        "summary": summary,
        "has_failures": bool(summary["failed"]),
        "results": [
            {"kb_id": k, "status": v[0], "stats": v[1]}
            for k, v in results.items()
        ],
    }


def answer_image(image_id=None, image_path=None, question="", data_id=None, ref=None):
    asset = read_image(image_id=image_id, image_path=image_path, data_id=data_id, ref=ref)
    if asset.get("error"):
        return asset
    client = _load_image_client()
    return client.answer_image_question(
        asset.get("image_abspath") or os.path.join(_kb_by_id(asset.get("kb_id"))["path"], asset.get("image_path", "")),
        question=question or "",
        title=asset.get("title", ""),
        context=asset,
    )


def resolve_open_target(kb_id="", data_id="", image_id="", image_path="", ref="", ref_key=""):
    """Resolve Desktop /kb/open to an original document or indexed image."""
    is_image_target = bool(image_id or image_path or "::image::" in str(data_id or ""))
    if is_image_target:
        asset = read_image(
            image_id=image_id,
            image_path=image_path,
            data_id=data_id,
            ref=ref,
            kb_id=kb_id,
            ref_key=ref_key,
        )
        if not asset.get("error"):
            path = asset.get("image_abspath") or ""
            if path and os.path.isfile(path):
                return path
        return None

    source = resolve_source_document(kb_id=kb_id, data_id=data_id, ref=ref)
    source_path = source.get("path") or ""
    if source_path and os.path.isfile(source_path):
        return source_path
    return None


# ───────────────────────────── CLI ─────────────────────────────

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="外挂知识库引擎")
    p.add_argument("--build", action="store_true", help="增量构建所有配置库")
    p.add_argument("--rebuild", action="store_true", help="强制全量重建")
    p.add_argument("--kb", help="配合 --build/--rebuild：只构建指定知识库 ID")
    p.add_argument("--text-only", action="store_true", help="只构建文本 chunk 索引，跳过图片资产")
    p.add_argument("--images-only", action="store_true", help="只补图片资产到已有 Zvec 文本索引，不重建文本 embedding")
    p.add_argument("--status", action="store_true", help="查看各库状态")
    p.add_argument("--search", metavar="Q", help="检索测试")
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--add", nargs=2, metavar=("ID", "PATH"), help="新增/更新知识库：--add 库ID 文件夹路径")
    p.add_argument("--rm", metavar="ID", help="移除知识库配置（不删除原始文件）")
    p.add_argument("--name", help="配合 --add：设置展示名")
    p.add_argument("--preload", action="store_true", help="配合 --add：开启会话预加载")
    args = p.parse_args(argv)

    if args.add:
        kid, path = args.add
        kbs = upsert_kb(kid, path=path, preload=args.preload, name=args.name)
        print(json.dumps({"ok": True, "knowledge_bases": [k["id"] for k in kbs]}, ensure_ascii=False))
        print("提示：运行 `python -m knowledge_base.backend --build` 构建索引（或重启 Web 服务自动构建）。")
        return 0
    if args.rm:
        ok = remove_kb(args.rm)
        print(json.dumps({"ok": ok, "removed": args.rm if ok else None}, ensure_ascii=False))
        return 0
    if args.status:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    if args.search:
        for r in search(args.search, top_k=args.top_k):
            print(f"[{r['score']}] {r['ref']} #{r['chunk_index']} :: {r['snippet'][:120]}")
        return 0
    if args.build or args.rebuild:
        if args.text_only and args.images_only:
            print("错误：--text-only 和 --images-only 不能同时使用")
            return 2
        mode = "images" if args.images_only else "text" if args.text_only else "full"
        res = build_all(force=args.rebuild, verbose=True, kb_id=args.kb, mode=mode)
        print(json.dumps({k: v[0] for k, v in res.items()}, ensure_ascii=False))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
