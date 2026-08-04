"""Concrete Zvec repository for unified text and image records."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import shutil
import threading
import time
from typing import Any, Callable

from .providers import embeddings
from .schema import INDEX_SCHEMA_VERSION, OUTPUT_FIELDS, collection_schema, normalize_record
from .cancellation import check_cancelled


class ZvecIndex:
    """Own Zvec schema, embeddings, native handles, and atomic index builds."""

    def __init__(self, *, dimension: int, batch_size: int, usage_tracker) -> None:
        self.dimension = int(dimension)
        self.batch_size = max(1, int(batch_size))
        self.schema_version = INDEX_SCHEMA_VERSION
        self.usage = usage_tracker

    @staticmethod
    def index_dir(kb_path: str) -> str:
        return os.path.join(kb_path, ".kb_index")

    @classmethod
    def path(cls, kb_path: str) -> str:
        return os.path.join(cls.index_dir(kb_path), "zvec")

    @classmethod
    def meta_path(cls, kb_path: str) -> str:
        return os.path.join(cls.index_dir(kb_path), "zvec_meta.json")

    @staticmethod
    def require():
        try:
            import zvec
        except Exception as exc:
            raise RuntimeError("Zvec 是知识库索引的必需依赖，请检查运行时 wheel 安装") from exc
        return zvec

    @contextlib.contextmanager
    def open_collection(
        self,
        path: str,
        *,
        create: bool = False,
        read_only: bool = True,
    ):
        zvec = self.require()
        if create:
            collection = zvec.create_and_open(
                path=path,
                schema=collection_schema(zvec, self.dimension),
            )
        else:
            try:
                option = zvec.CollectionOption(read_only=bool(read_only))
            except Exception:
                option = zvec.CollectionOption()
                with contextlib.suppress(Exception):
                    option.read_only = bool(read_only)
            collection = zvec.open(path=path, option=option)
        try:
            yield collection
        finally:
            self._release_collection(collection)
            del collection
            gc.collect()

    @staticmethod
    def _release_collection(collection) -> None:
        for attr in ("_querier", "_obj", "_schema"):
            with contextlib.suppress(Exception):
                setattr(collection, attr, None)

    @staticmethod
    def _rename_with_retry(source: str, destination: str) -> None:
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

    def meta(self, kb_path: str) -> dict[str, Any]:
        try:
            with open(self.meta_path(kb_path), encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _embedding_fingerprint(meta: dict | None) -> dict:
        return {
            key: (meta or {}).get(key)
            for key in ("provider", "base_url", "model", "dimension", "output_type")
            if (meta or {}).get(key) is not None
        }

    def embedding_config_matches(self, meta: dict | None) -> bool:
        value = meta or {}
        return (
            self._embedding_fingerprint(value.get("embedding"))
            == self._embedding_fingerprint(embeddings.embedding_meta())
            and self._embedding_fingerprint(value.get("sparse_embedding"))
            == self._embedding_fingerprint(embeddings.sparse_embedding_meta())
        )

    def probe(self, kb_path: str) -> dict:
        path = self.path(kb_path)
        meta = self.meta(kb_path)
        present = os.path.isdir(path)
        schema_valid = bool(meta and meta.get("schema_version") == self.schema_version)
        openable = False
        error = ""
        if present and schema_valid:
            try:
                with self.open_collection(path):
                    pass
                openable = True
            except Exception as exc:
                error = str(exc)
        return {
            "present": present,
            "openable": openable,
            "schema_valid": schema_valid,
            "embedding_matches": bool(meta and self.embedding_config_matches(meta)),
            "error": error,
            "meta": meta,
        }

    @staticmethod
    def doc_id(data_id: str, chunk_index: int) -> str:
        return hashlib.sha1(f"{data_id}#{chunk_index}".encode("utf-8")).hexdigest()

    def embed_dense(
        self,
        texts: list[str],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[list[float]]:
        return embeddings.embed_texts(texts, cancelled=cancelled)

    def embed_sparse(
        self,
        texts: list[str],
        *,
        text_type: str = "document",
        cancelled: Callable[[], bool] | None = None,
    ):
        return embeddings.embed_sparse_texts(
            texts,
            text_type=text_type,
            cancelled=cancelled,
        )

    @staticmethod
    def _record_cache_key(item: dict) -> str:
        # Embeddings depend on the actual model input, not the record's current
        # ordinal. Rechunking may move an unchanged semantic unit to another
        # chunk_index; hashing the input lets that vector remain reusable.
        text = str(item.get("search_text") or item.get("body") or "")
        return hashlib.sha256(("search-v2\x00" + text).encode("utf-8")).hexdigest()

    def _load_record_vector_cache(self, kb_path: str) -> tuple[str, dict]:
        path = os.path.join(self.index_dir(kb_path), "record_embeddings.json")
        current = {
            "embedding": embeddings.embedding_meta(),
            "sparse_embedding": embeddings.sparse_embedding_meta(),
        }
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or payload.get("version") != 2:
                return path, {}
            if (
                self._embedding_fingerprint(payload.get("embedding"))
                != self._embedding_fingerprint(current["embedding"])
                or self._embedding_fingerprint(payload.get("sparse_embedding"))
                != self._embedding_fingerprint(current["sparse_embedding"])
            ):
                return path, {}
            rows = payload.get("records")
            if not isinstance(rows, dict):
                return path, {}
            valid = {}
            for key, value in rows.items():
                if not isinstance(value, dict):
                    continue
                dense = value.get("dense")
                sparse = value.get("sparse")
                if not isinstance(dense, list) or len(dense) != self.dimension:
                    continue
                if not isinstance(sparse, dict):
                    continue
                valid[str(key)] = value
            return path, valid
        except Exception:
            return path, {}

    def insert_records(
        self,
        kb: dict,
        collection,
        records,
        logfn=None,
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
    ):
        zvec = self.require()
        batch: list[dict] = []
        counts = {
            "n_chunks": 0,
            "text_chunks": 0,
            "image_chunks": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
        }
        docs_seen: set[str] = set()
        embedding_used = embeddings.embedding_meta()
        sparse_embedding_used = embeddings.sparse_embedding_meta()
        cache_path, vector_cache = self._load_record_vector_cache(kb["path"])
        cache_dirty = False
        total_records = len(records) if hasattr(records, "__len__") else 0
        flushed = 0

        def flush() -> None:
            nonlocal batch, cache_dirty, flushed
            if not batch:
                return
            check_cancelled(cancelled)
            missing = [
                item for item in batch
                if self._record_cache_key(item) not in vector_cache
            ]
            if missing:
                texts = [str(item.get("search_text") or item.get("body") or "") for item in missing]
                dense_usage = self.usage.current()["embedding"]
                sparse_usage = self.usage.current()["sparse_embedding"]
                dense_usage["calls"] += 1
                dense_usage["texts"] += len(texts)
                sparse_usage["calls"] += 1
                sparse_usage["texts"] += len(texts)
                embeddings.drain_usage()
                vectors = self.embed_dense(texts, cancelled=cancelled)
                sparse_vectors = self.embed_sparse(
                    texts,
                    text_type="document",
                    cancelled=cancelled,
                )
                tokens = embeddings.drain_usage()
                dense_usage["api_tokens"] += int(tokens.get("dense") or 0)
                sparse_usage["api_tokens"] += int(tokens.get("sparse") or 0)
                dense_usage["input_tokens"] += int(tokens.get("dense_input_tokens") or 0)
                sparse_usage["input_tokens"] += int(tokens.get("sparse_input_tokens") or 0)
                dense_usage["output_tokens"] += int(tokens.get("dense_output_tokens") or 0)
                sparse_usage["output_tokens"] += int(tokens.get("sparse_output_tokens") or 0)
                dense_usage["token_usage_reported"] = bool(
                    dense_usage.get("token_usage_reported") or tokens.get("dense_reported")
                )
                sparse_usage["token_usage_reported"] = bool(
                    sparse_usage.get("token_usage_reported") or tokens.get("sparse_reported")
                )
                dense_usage["input_token_usage_reported"] = bool(
                    dense_usage.get("input_token_usage_reported")
                    or tokens.get("dense_input_reported")
                )
                sparse_usage["input_token_usage_reported"] = bool(
                    sparse_usage.get("input_token_usage_reported")
                    or tokens.get("sparse_input_reported")
                )
                dense_usage["output_token_usage_reported"] = bool(
                    dense_usage.get("output_token_usage_reported")
                    or tokens.get("dense_output_reported")
                )
                sparse_usage["output_token_usage_reported"] = bool(
                    sparse_usage.get("output_token_usage_reported")
                    or tokens.get("sparse_output_reported")
                )
                dense_usage["cache_hits"] = int(dense_usage.get("cache_hits") or 0) + int(
                    tokens.get("dense_cache_hits") or 0
                )
                sparse_usage["cache_hits"] = int(sparse_usage.get("cache_hits") or 0) + int(
                    tokens.get("sparse_cache_hits") or 0
                )
                dense_usage["api_calls"] = int(dense_usage.get("api_calls") or 0) + int(
                    tokens.get("dense_api_calls") or 0
                )
                sparse_usage["api_calls"] = int(sparse_usage.get("api_calls") or 0) + int(
                    tokens.get("sparse_api_calls") or 0
                )
                for item, vector, sparse_vector in zip(missing, vectors, sparse_vectors):
                    vector_cache[self._record_cache_key(item)] = {
                        "dense": list(vector),
                        "sparse": {str(key): float(value) for key, value in sparse_vector.items()},
                    }
                cache_dirty = True
            cache_hits = len(batch) - len(missing)
            counts["embedding_cache_hits"] += cache_hits
            counts["embedding_cache_misses"] += len(missing)
            if cache_hits:
                self.usage.current()["embedding"]["cache_hits"] += cache_hits
                self.usage.current()["sparse_embedding"]["cache_hits"] += cache_hits
            check_cancelled(cancelled)
            if cache_dirty:
                self._write_json_atomic(
                    cache_path,
                    {
                        "version": 2,
                        "embedding": embedding_used,
                        "sparse_embedding": sparse_embedding_used,
                        "records": vector_cache,
                    },
                )
                cache_dirty = False

            docs = [
                zvec.Doc(
                    id=self.doc_id(item["data_id"], item["chunk_index"]),
                    vectors={
                        "embedding": vector_cache[self._record_cache_key(item)]["dense"],
                        "sparse_embedding": {
                            int(key): float(value)
                            for key, value in vector_cache[self._record_cache_key(item)]["sparse"].items()
                        },
                    },
                    fields={field: item[field] for field in OUTPUT_FIELDS},
                )
                for item in batch
            ]
            collection.insert(docs)
            flushed += len(batch)
            if callable(progress):
                progress({"phase": "indexing", "processed": flushed, "total": total_records})
            if logfn:
                logfn(f"  Zvec 已写入 {counts['n_chunks']} 条记录")
            batch = []

        for raw in records:
            check_cancelled(cancelled)
            item = normalize_record(raw, kb_id=kb["id"])
            if not item["body"] or not item["data_id"]:
                continue
            if item["kind"] == "image":
                counts["image_chunks"] += 1
            else:
                counts["text_chunks"] += 1
                docs_seen.add(item["data_id"])
            counts["n_chunks"] += 1
            batch.append(item)
            if len(batch) >= self.batch_size:
                flush()
        flush()
        check_cancelled(cancelled)
        counts["n_docs"] = len(docs_seen)
        return counts, embedding_used, sparse_embedding_used

    @staticmethod
    def _write_json_atomic(path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    def build(
        self,
        kb: dict,
        records,
        sources: dict,
        logfn=None,
        *,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
    ):
        path = self.path(kb["path"])
        meta_path = self.meta_path(kb["path"])
        temp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        backup_path = ""
        started = time.time()
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with self.open_collection(
                temp_path,
                create=True,
                read_only=False,
            ) as collection:
                stats, embedding_used, sparse_embedding_used = self.insert_records(
                    kb,
                    collection,
                    records,
                    logfn=logfn,
                    cancelled=cancelled,
                    progress=progress,
                )
                if not stats["n_chunks"]:
                    raise RuntimeError("没有可写入索引的图文记录")
                check_cancelled(cancelled)
                collection.flush()
                check_cancelled(cancelled)
                if os.environ.get("GA_KB_ZVEC_OPTIMIZE", "0").strip().lower() in (
                    "1", "true", "yes", "on"
                ):
                    check_cancelled(cancelled)
                    with contextlib.suppress(Exception):
                        collection.optimize()

            stats.update({
                "image_assets": int(stats["image_chunks"]),
                "zvec_bytes": self.dir_size(temp_path),
                "build_seconds": round(time.time() - started, 1),
                "embedding": embedding_used,
                "sparse_embedding": sparse_embedding_used,
            })
            meta = {
                "schema_version": self.schema_version,
                "built_at": int(time.time()),
                "sources": sources,
                "embedding": embedding_used,
                "sparse_embedding": sparse_embedding_used,
                "indexed_kinds": ["text", "image"] if stats["image_chunks"] else ["text"],
                "stats": stats,
            }

            if os.path.exists(path):
                backup_path = f"{path}.rollback.{time.time_ns()}"
                self._rename_with_retry(path, backup_path)
            try:
                self._rename_with_retry(temp_path, path)
                self._write_json_atomic(meta_path, meta)
            except Exception:
                shutil.rmtree(path, ignore_errors=True)
                if backup_path and os.path.exists(backup_path):
                    self._rename_with_retry(backup_path, path)
                    backup_path = ""
                raise
            if backup_path:
                shutil.rmtree(backup_path, ignore_errors=True)
            check_cancelled(cancelled)
            if logfn:
                logfn(
                    f"  Zvec 索引完成：{stats['n_chunks']} 条记录 / "
                    f"{stats['zvec_bytes'] // 1024 // 1024} MB"
                )
            return stats
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
            gc.collect()

    def fetch(self, kb: dict, data_id: str, chunk_index: int, output_fields=None):
        if not kb or not data_id:
            return None
        path = self.path(kb["path"])
        if not os.path.isdir(path):
            return None
        document_id = self.doc_id(data_id, int(chunk_index))
        with self.open_collection(path) as collection:
            got = collection.fetch(
                document_id,
                output_fields=output_fields or OUTPUT_FIELDS,
                include_vector=False,
            )
        return got.get(document_id) if isinstance(got, dict) else None

    @staticmethod
    def dir_size(path: str) -> int:
        total = 0
        for directory, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                with contextlib.suppress(OSError):
                    total += os.path.getsize(os.path.join(directory, filename))
        return total


__all__ = ["ZvecIndex"]
