"""Zvec collection lifecycle, metadata, and index construction."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import shutil
import threading
import time
from typing import Any, Callable, Dict


class ZvecIndex:
    """Own one knowledge-base Zvec collection lifecycle.

    The indexer receives provider, usage, document-fingerprint, and asset
    callbacks from the orchestration layer.  It therefore owns storage and
    native-handle cleanup without importing the backend module.
    """

    def __init__(
        self,
        *,
        dimension: int,
        batch_size: int,
        schema_version: int,
        path_fn: Callable[[str], str],
        meta_path_fn: Callable[[str], str],
        embedding_fn: Callable[[list[str]], list[list[float]]],
        sparse_embedding_fn: Callable[..., list[dict[int, float]]],
        embedding_meta_fn: Callable[[], Dict[str, Any]],
        sparse_embedding_meta_fn: Callable[[], Dict[str, Any]],
        chunking_meta_fn: Callable[[], Dict[str, Any]],
        image_analysis_meta_fn: Callable[[], Dict[str, Any]],
        usage_fn: Callable[[], Dict[str, Any]],
        load_assets_fn: Callable[[str], list],
        document_fingerprint_fn: Callable[[list], Dict[str, Any]],
        embedding_usage_drain_fn: Callable[[], Dict[str, int]],
    ) -> None:
        self.dimension = int(dimension)
        self.batch_size = max(1, int(batch_size))
        self.schema_version = int(schema_version)
        self._path_fn = path_fn
        self._meta_path_fn = meta_path_fn
        self._embedding_fn = embedding_fn
        self._sparse_embedding_fn = sparse_embedding_fn
        self._embedding_meta_fn = embedding_meta_fn
        self._sparse_embedding_meta_fn = sparse_embedding_meta_fn
        self._chunking_meta_fn = chunking_meta_fn
        self._image_analysis_meta_fn = image_analysis_meta_fn
        self._usage_fn = usage_fn
        self._load_assets_fn = load_assets_fn
        self._document_fingerprint_fn = document_fingerprint_fn
        self._embedding_usage_drain_fn = embedding_usage_drain_fn
        self._local = threading.local()
        self._cache_lock = threading.RLock()
        self._connection_caches = []

    @staticmethod
    def require():
        try:
            import zvec
        except Exception as exc:
            raise RuntimeError("Zvec 是知识库索引的必需依赖，请检查运行时 wheel 安装") from exc
        return zvec

    def connect(self, path: str, *, create: bool = False, read_only: bool = True):
        cache = getattr(self._local, "connections", None)
        if cache is None:
            cache = self._local.connections = {}
            with self._cache_lock:
                self._connection_caches.append(cache)
        key = (path, create, read_only)
        collection = cache.get(key)
        if collection is not None:
            return collection

        zvec = self.require()
        if create:
            schema = zvec.CollectionSchema(
                name="kb_chunks",
                fields=[
                    zvec.FieldSchema("data_id", zvec.DataType.STRING),
                    zvec.FieldSchema("chunk_index", zvec.DataType.INT64),
                    zvec.FieldSchema("kb_id", zvec.DataType.STRING),
                    zvec.FieldSchema("file_name", zvec.DataType.STRING),
                    zvec.FieldSchema("title", zvec.DataType.STRING),
                    zvec.FieldSchema("kind", zvec.DataType.STRING),
                    zvec.FieldSchema("image_path", zvec.DataType.STRING),
                    zvec.FieldSchema("source_data_id", zvec.DataType.STRING),
                    zvec.FieldSchema("source_chunk_index", zvec.DataType.INT64),
                    zvec.FieldSchema("header_path", zvec.DataType.STRING),
                    zvec.FieldSchema("body", zvec.DataType.STRING),
                ],
                vectors=[
                    zvec.VectorSchema(
                        "embedding",
                        zvec.DataType.VECTOR_FP32,
                        self.dimension,
                        index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
                    ),
                    zvec.VectorSchema("sparse_embedding", zvec.DataType.SPARSE_VECTOR_FP32),
                ],
            )
            collection = zvec.create_and_open(path=path, schema=schema)
        else:
            try:
                option = zvec.CollectionOption(read_only=bool(read_only))
            except Exception:
                option = zvec.CollectionOption()
                with contextlib.suppress(Exception):
                    option.read_only = bool(read_only)
            collection = zvec.open(path=path, option=option)
        cache[key] = collection
        return collection

    @staticmethod
    def _release_collection(collection) -> None:
        """Release Zvec native handles before replacing a collection on Windows."""
        for attr in ("_querier", "_obj", "_schema"):
            with contextlib.suppress(Exception):
                setattr(collection, attr, None)

    @staticmethod
    def _rename_with_retry(source: str, destination: str) -> None:
        """Rename an index directory across short-lived Windows file locks."""
        last_error = None
        for delay in (0, 0.1, 0.25, 0.5, 1, 2):
            if delay:
                time.sleep(delay)
            try:
                os.rename(source, destination)
                return
            except PermissionError as error:
                last_error = error
        if last_error is not None:
            raise last_error

    def clear_cache(self, path: str | None = None) -> None:
        # Tool calls run in worker threads while Bridge requests usually run
        # on the event-loop thread.  Releasing only the current thread's cache
        # leaves Windows IPC handles alive and makes KB deletion fail with
        # WinError 32.  Drain every thread cache for the requested collection.
        with self._cache_lock:
            caches = list(self._connection_caches)
        for cache in caches:
            for key in list(cache.keys()):
                if path is None or key[0] == path:
                    collection = cache.pop(key, None)
                    if collection is not None:
                        self._release_collection(collection)
                        del collection
        with self._cache_lock:
            self._connection_caches = [cache for cache in self._connection_caches if cache]
        current = getattr(self._local, "connections", None)
        if current is not None and not current:
            self._local.connections = {}
        gc.collect()

    def meta(self, kb_path: str) -> Dict[str, Any]:
        try:
            with open(self._meta_path_fn(kb_path), encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _embedding_fingerprint(meta: Dict[str, Any] | None) -> Dict[str, Any]:
        meta = meta or {}
        return {
            key: meta.get(key)
            for key in ("provider", "base_url", "model", "dimension", "output_type")
            if meta.get(key) is not None
        }

    def _embedding_config_matches(self, meta: Dict[str, Any] | None) -> bool:
        meta = meta or {}
        return (
            self._embedding_fingerprint(meta.get("embedding"))
            == self._embedding_fingerprint(self._embedding_meta_fn())
            and self._embedding_fingerprint(meta.get("sparse_embedding"))
            == self._embedding_fingerprint(self._sparse_embedding_meta_fn())
        )

    def _index_is_usable(self, kb_path: str, meta: Dict[str, Any] | None) -> bool:
        return bool(
            self._embedding_config_matches(meta)
            and os.path.isdir(self._path_fn(kb_path))
        )

    def is_fresh(self, kb_path: str, sources: Dict[str, Any], required_kinds=None) -> bool:
        meta = self.meta(kb_path)
        indexed = set(meta.get("indexed_kinds") or ["text", "image"])
        if required_kinds and not set(required_kinds).issubset(indexed):
            return False
        return bool(
            meta
            and meta.get("schema_version") == self.schema_version
            and meta.get("sources") == sources
            and self._index_is_usable(kb_path, meta)
        )

    def is_quickly_fresh(self, kb_path: str, scanned: list, meta=None, mode: str = "full") -> bool:
        meta = meta or self.meta(kb_path)
        sources = meta.get("sources") or {}
        if not meta or meta.get("schema_version") != self.schema_version:
            return False
        mode = mode if mode in ("full", "text", "images") else "full"
        required_kinds = {"text"} if mode != "images" else {"image"}
        indexed_kinds = set(meta.get("indexed_kinds") or [])
        if not required_kinds.issubset(indexed_kinds):
            return False
        if "images" not in sources and mode != "text":
            return False
        if sources.get("documents") != self._document_fingerprint_fn(scanned):
            return False
        if sources.get("chunking") != self._chunking_meta_fn():
            return False
        if mode != "text":
            if sources.get("image_analysis") != self._image_analysis_meta_fn():
                return False
            for rel, expected in (sources.get("images") or {}).items():
                path = os.path.realpath(os.path.join(kb_path, rel))
                try:
                    stat = os.stat(path)
                except OSError:
                    return False
                if {"mtime": int(stat.st_mtime), "size": stat.st_size} != expected:
                    return False
        return self._index_is_usable(kb_path, meta)

    @staticmethod
    def doc_id(data_id: str, chunk_index: int) -> str:
        return hashlib.sha1(f"{data_id}#{chunk_index}".encode("utf-8")).hexdigest()

    def insert_records(self, kb, collection, records, logfn=None, operation: str = "insert"):
        zvec = self.require()
        batch = []
        n_chunks = 0
        n_text_chunks = 0
        n_image_chunks = 0
        docs_seen = set()
        embedding_used = self._embedding_meta_fn()
        sparse_embedding_used = self._sparse_embedding_meta_fn()

        def flush():
            nonlocal batch
            if not batch:
                return
            texts = [item["body"] for item in batch]
            usage = self._usage_fn()["embedding"]
            sparse_usage = self._usage_fn()["sparse_embedding"]
            usage["calls"] += 1
            usage["texts"] += len(texts)
            sparse_usage["calls"] += 1
            sparse_usage["texts"] += len(texts)
            # Clear any stale accumulator, then attribute this flush's real
            # provider token usage (dense + sparse) after both calls return.
            # Cache hits inside the embedding client report no tokens, so
            # api_tokens reflects only text that actually reached the API.
            self._embedding_usage_drain_fn()
            vectors = self._embedding_fn(texts)
            sparse_vectors = self._sparse_embedding_fn(texts, text_type="document")
            tokens = self._embedding_usage_drain_fn()
            usage["api_tokens"] += int(tokens.get("dense") or 0)
            sparse_usage["api_tokens"] += int(tokens.get("sparse") or 0)

            docs = []
            for item, vector, sparse_vector in zip(batch, vectors, sparse_vectors):
                docs.append(zvec.Doc(
                    id=self.doc_id(item["data_id"], item["chunk_index"]),
                    vectors={"embedding": vector, "sparse_embedding": sparse_vector},
                    fields={
                        "data_id": item["data_id"],
                        "chunk_index": int(item["chunk_index"]),
                        "kb_id": kb["id"],
                        "file_name": item["file_name"],
                        "title": item["title"],
                        "kind": item.get("kind", "text"),
                        "image_path": item.get("image_path", ""),
                        "source_data_id": item.get("source_data_id", ""),
                        "source_chunk_index": int(item.get("source_chunk_index", -1)),
                        "header_path": item.get("header_path", ""),
                        "body": item["body"],
                    },
                ))
            writer = collection.upsert if operation == "upsert" else collection.insert
            writer(docs)
            if logfn:
                verb = "upsert" if operation == "upsert" else "写入"
                logfn(f"  zvec 已{verb} {n_chunks} chunk...")
            batch = []

        for record in records:
            body = record.get("body", "") or ""
            data_id = record.get("data_id") or ""
            if not body or not data_id:
                continue
            chunk_index = int(record.get("chunk_index", 0))
            kind = record.get("kind", "text") or "text"
            if kind == "image":
                n_image_chunks += 1
            else:
                n_text_chunks += 1
                docs_seen.add(data_id)
            batch.append({
                "data_id": data_id,
                "chunk_index": chunk_index,
                "file_name": record.get("file_name", "") or "",
                "title": record.get("title", "") or "",
                "kind": kind,
                "image_path": record.get("image_path", "") or "",
                "source_data_id": record.get("source_data_id", "") or "",
                "source_chunk_index": int(record.get("source_chunk_index", -1)),
                "header_path": record.get("header_path", "") or "",
                "body": body,
            })
            n_chunks += 1
            if len(batch) >= self.batch_size:
                flush()
        flush()
        return {
            "n_docs": len(docs_seen),
            "n_chunks": n_chunks,
            "text_chunks": n_text_chunks,
            "image_chunks": n_image_chunks,
        }, embedding_used, sparse_embedding_used

    @staticmethod
    def _write_json_temp(path: str, payload: Dict[str, Any]) -> str:
        temp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return temp_path
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            raise

    @classmethod
    def _write_json_atomic(cls, path: str, payload: Dict[str, Any]) -> None:
        temp_path = cls._write_json_temp(path, payload)
        try:
            os.replace(temp_path, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    def build(self, kb, records, sources, force: bool = False, logfn=None):
        path = self._path_fn(kb["path"])
        meta_path = self._meta_path_fn(kb["path"])
        if not force and self.is_fresh(kb["path"], sources):
            return "up-to-date", (self.meta(kb["path"]) or {}).get("stats", {})

        temp_path = path + ".tmp"
        backup_path = ""
        meta_temp_path = ""
        collection = None
        published = False
        started = time.time()
        try:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            self.clear_cache(path)
            self.clear_cache(temp_path)
            try:
                collection = self.connect(temp_path, create=True)
            except Exception as exc:
                return "unavailable", {"error": str(exc)}

            stats, embedding_used, sparse_embedding_used = self.insert_records(
                kb, collection, records, logfn=logfn
            )
            collection.flush()
            if os.environ.get("GA_KB_ZVEC_OPTIMIZE", "0").strip().lower() in ("1", "true", "yes", "on"):
                try:
                    collection.optimize()
                except Exception as exc:
                    if logfn:
                        logfn(f"  [warn] zvec optimize 跳过：{exc}")

            self.clear_cache(temp_path)
            with contextlib.suppress(Exception):
                del collection
            collection = None
            gc.collect()

            stats = {
                **stats,
                "image_assets": sum(1 for _asset in self._load_assets_fn(kb["path"])),
                "zvec_bytes": self.dir_size(temp_path),
                "build_seconds": round(time.time() - started, 1),
            }
            meta = {
                "schema_version": self.schema_version,
                "built_at": int(time.time()),
                "sources": sources,
                "embedding": embedding_used,
                "sparse_embedding": sparse_embedding_used,
                "vector_index": {"type": "hnsw", "metric": "cosine"},
                "sparse_index": {"field": "sparse_embedding", "type": "SPARSE_VECTOR_FP32"},
                "indexed_kinds": ["text"] if stats.get("image_chunks", 0) == 0 else ["text", "image"],
                "stats": stats,
            }
            meta_temp_path = self._write_json_temp(meta_path, meta)

            if os.path.exists(path):
                backup_path = f"{path}.bak.{os.getpid()}.{time.time_ns()}"
                self._rename_with_retry(path, backup_path)
            try:
                self._rename_with_retry(temp_path, path)
                os.replace(meta_temp_path, meta_path)
                meta_temp_path = ""
            except Exception:
                self.clear_cache(path)
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
                if backup_path and os.path.exists(backup_path):
                    self._rename_with_retry(backup_path, path)
                    backup_path = ""
                raise

            published = True
            if backup_path:
                shutil.rmtree(backup_path, ignore_errors=True)
                backup_path = ""
            self.clear_cache(path)
            if logfn:
                logfn(f"  zvec dense+sparse 索引完成：{stats['n_chunks']} chunk / {stats['zvec_bytes']//1024//1024}MB")
            return "built", stats
        except Exception:
            if not published and backup_path and os.path.exists(backup_path):
                self.clear_cache(path)
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
                self._rename_with_retry(backup_path, path)
                backup_path = ""
            raise
        finally:
            self.clear_cache(temp_path)
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
            if collection is not None:
                with contextlib.suppress(Exception):
                    del collection
            with contextlib.suppress(OSError):
                if meta_temp_path:
                    os.remove(meta_temp_path)
            gc.collect()

    def append_images(self, kb, records, sources, force: bool = False, logfn=None):
        path = self._path_fn(kb["path"])
        meta_path = self._meta_path_fn(kb["path"])
        if not os.path.isdir(path):
            return "unavailable", {"error": "text Zvec index missing; run --text-only first"}
        if not force and self.is_fresh(kb["path"], sources, required_kinds=["text", "image"]):
            return "up-to-date", (self.meta(kb["path"]) or {}).get("stats", {})

        meta = self.meta(kb["path"])
        if meta.get("schema_version") != self.schema_version:
            return "unavailable", {"error": f"schema mismatch: {meta.get('schema_version')} != {self.schema_version}"}
        if self._embedding_fingerprint(meta.get("embedding")) != self._embedding_fingerprint(self._embedding_meta_fn()):
            return "unavailable", {"error": "embedding config changed; rebuild text index first"}
        if self._embedding_fingerprint(meta.get("sparse_embedding")) != self._embedding_fingerprint(self._sparse_embedding_meta_fn()):
            return "unavailable", {"error": "sparse embedding config changed; rebuild text index first"}

        self.clear_cache(path)
        try:
            collection = self.connect(path, read_only=False)
        except Exception as exc:
            return "unavailable", {"error": str(exc)}

        started = time.time()
        try:
            # M3: the old image chunks MUST be deleted before re-inserting.
            # upsert only overwrites rows whose data_id is unchanged; any
            # image removed from the source, or whose data_id shifted, would
            # otherwise survive as an orphan chunk and silently pollute
            # retrieval.  A delete failure is therefore fatal — retry a few
            # times, then abort the append so the prior index and the pending
            # assets are both rolled back (build.py discards pending on a
            # non-"built" status) instead of publishing a corrupt index.
            delete_attempts = 3
            for attempt in range(1, delete_attempts + 1):
                try:
                    collection.delete_by_filter("kind = 'image'")
                    break
                except Exception as exc:
                    if attempt >= delete_attempts:
                        if logfn:
                            logfn(f"  [error] zvec 删除旧图片资产失败（已重试 {delete_attempts} 次），放弃追加：{exc}")
                        return "unavailable", {
                            "error": f"failed to delete stale image chunks after {delete_attempts} attempts: {exc}"
                        }
                    if logfn:
                        logfn(f"  [warn] zvec 删除旧图片资产失败（第 {attempt}/{delete_attempts} 次），重试：{exc}")
                    time.sleep(0.5 * attempt)
            stats, embedding_used, sparse_embedding_used = self.insert_records(
                kb, collection, records, logfn=logfn, operation="upsert"
            )
            collection.flush()
            if os.environ.get("GA_KB_ZVEC_OPTIMIZE", "0").strip().lower() in ("1", "true", "yes", "on"):
                try:
                    collection.optimize()
                except Exception as exc:
                    if logfn:
                        logfn(f"  [warn] zvec optimize 跳过：{exc}")
        finally:
            self.clear_cache(path)
            with contextlib.suppress(Exception):
                del collection
            gc.collect()

        previous = meta.get("stats") or {}
        text_chunks = int(previous.get("text_chunks") or max(0, int(previous.get("n_chunks") or 0) - int(previous.get("image_chunks") or 0)))
        merged_stats = {
            "n_docs": int(previous.get("n_docs") or stats.get("n_docs") or 0),
            "n_chunks": text_chunks + int(stats.get("image_chunks") or 0),
            "text_chunks": text_chunks,
            "image_chunks": int(stats.get("image_chunks") or 0),
            "image_assets": sum(1 for _asset in self._load_assets_fn(kb["path"])),
            "zvec_bytes": self.dir_size(path),
            "build_seconds": round(time.time() - started, 1),
        }
        meta.update({
            "schema_version": self.schema_version,
            "built_at": int(time.time()),
            "sources": sources,
            "embedding": embedding_used,
            "sparse_embedding": sparse_embedding_used,
            "vector_index": {"type": "hnsw", "metric": "cosine"},
            "sparse_index": {"field": "sparse_embedding", "type": "SPARSE_VECTOR_FP32"},
            "indexed_kinds": ["text", "image"],
            "stats": merged_stats,
        })
        self._write_json_atomic(meta_path, meta)
        if logfn:
            logfn(f"  zvec 图片资产追加完成：{merged_stats['image_chunks']} image chunk / 总 {merged_stats['n_chunks']} chunk")
        return "built", merged_stats

    @staticmethod
    def dir_size(path: str) -> int:
        total = 0
        for directory, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(directory, filename))
                except OSError:
                    pass
        return total

    def fetch(self, kb, data_id: str, chunk_index: int, output_fields=None):
        if not kb or not data_id:
            return None
        path = self._path_fn(kb["path"])
        if not os.path.isdir(path):
            return None
        try:
            collection = self.connect(path)
            document_id = self.doc_id(data_id, int(chunk_index))
            got = collection.fetch(
                document_id,
                output_fields=output_fields,
                include_vector=False,
            )
            return got.get(document_id) if isinstance(got, dict) else None
        except Exception:
            return None
