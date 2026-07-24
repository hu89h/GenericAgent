"""Knowledge-base search and source-reading services.

This module owns the read-only side of the knowledge-base runtime.  It does
not import :mod:`backend`; storage, embedding, asset, and configuration
operations are supplied as callbacks so the service stays independent from
the build orchestrator and remains easy to exercise through the real API.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable


_SPLIT = re.compile(r"\s+")
_FIGURE_REF = re.compile(
    r"[图表]\s*[0-9０-９一二三四五六七八九十百]+"
    r"(?:[-－—–.．][0-9０-９一二三四五六七八九十百]+)+"
)
_DEFAULT_OUTPUT_FIELDS = [
    "data_id", "chunk_index", "kb_id", "file_name", "title", "kind",
    "image_path", "parent_data_id", "parent_chunk_index", "header_path",
    "chunk_role", "body",
]


class KnowledgeBaseRetriever:
    """Read-only retrieval facade for configured Zvec knowledge bases.

    The callbacks are deliberately narrow.  In particular, this class does
    not hold a backend module reference, which avoids a circular import and
    makes the retrieval path independent from build-state mutation.
    """

    def __init__(
        self,
        *,
        load_config: Callable[[], list],
        kb_by_id: Callable[[str], dict | None],
        scan_documents: Callable[[str], list],
        read_textfile: Callable[[str], str],
        zvec_path: Callable[[str], str],
        zvec_conn: Callable[..., Any],
        zvec_doc_id: Callable[[str, int], str],
        fetch_doc: Callable[..., Any],
        require_zvec: Callable[[], Any],
        embed_texts: Callable[[list[str]], list],
        embed_sparse_texts: Callable[..., list],
        load_image_assets: Callable[[str], list],
        local_ref_key: Callable[[str], str],
        asset_body: Callable[[dict], str],
        parent_chunk_body: Callable[[str, str, int], str],
        record_search_error: Callable[[dict, str, Any], None],
        imported_document_titles: Callable[[str], dict],
        imported_document_entries: Callable[[str], list[dict]] | None = None,
        query_factor: int = 4,
        vector_weight: float = 1.2,
        snippet_width: int = 220,
        output_fields: list[str] | None = None,
    ) -> None:
        self._load_config = load_config
        self._kb_by_id = kb_by_id
        self._scan_documents = scan_documents
        self._read_textfile = read_textfile
        self._zvec_path = zvec_path
        self._zvec_conn = zvec_conn
        self._zvec_doc_id = zvec_doc_id
        self._fetch_doc = fetch_doc
        self._require_zvec = require_zvec
        self._embed_texts = embed_texts
        self._embed_sparse_texts = embed_sparse_texts
        self._load_image_assets = load_image_assets
        self._local_ref_key = local_ref_key
        self._asset_body = asset_body
        self._parent_chunk_body = parent_chunk_body
        self._record_search_error = record_search_error
        self._imported_document_titles = imported_document_titles
        self._imported_document_entries = imported_document_entries or (lambda _path: [])
        self._query_factor = max(1, int(query_factor))
        self._vector_weight = float(vector_weight)
        self._snippet_width = max(1, int(snippet_width))
        self._output_fields = list(output_fields or _DEFAULT_OUTPUT_FIELDS)

    @staticmethod
    def _path_is_within(root: str, path: str) -> bool:
        root_real = os.path.realpath(root)
        path_real = os.path.realpath(path)
        return path_real == root_real or path_real.startswith(root_real + os.sep)

    @staticmethod
    def _folder_from_file_name(file_name: str) -> str:
        rel = str(file_name or "").replace("\\", "/")
        return rel.split("/", 1)[0] if "/" in rel else ""

    @staticmethod
    def _reference_terms(query: str) -> list[str]:
        out, seen = [], set()
        for match in _FIGURE_REF.finditer(query or ""):
            term = re.sub(r"\s+", "", match.group(0))
            if term and term not in seen:
                out.append(term)
                seen.add(term)
        return out

    def _snippet(self, text: str, query: str, width: int | None = None) -> str:
        if not text:
            return ""
        width = max(1, int(width or self._snippet_width))
        flat = re.sub(r"\s+", " ", text).strip()
        low = flat.lower()
        pos = -1
        for term in _SPLIT.split((query or "").strip()):
            if len(term) < 2:
                continue
            index = low.find(term.lower())
            if index != -1 and (pos == -1 or index < pos):
                pos = index
        if pos < 0:
            return (flat[:width] + " …") if len(flat) > width else flat
        start = max(0, pos - width // 3)
        end = min(len(flat), start + width)
        return ("… " if start > 0 else "") + flat[start:end] + (" …" if end < len(flat) else "")

    def _assets_for_kb(self, kb: dict, cache: dict[str, list] | None = None) -> list:
        path = str(kb.get("path") or "")
        if cache is None:
            return self._load_image_assets(path)
        if path not in cache:
            cache[path] = self._load_image_assets(path)
        return cache[path]

    def _asset_by_data_id(self, kb: dict, data_id: str, cache: dict[str, list] | None = None) -> dict:
        for asset in self._assets_for_kb(kb, cache):
            if asset.get("data_id") == data_id:
                return asset
        return {}

    def _asset_image_abspath(self, kb: dict, asset: dict) -> str:
        """Resolve an image only under the configured knowledge-base root."""
        rel = str((asset or {}).get("image_path") or "").replace("\\", "/").lstrip("/")
        if not rel:
            return ""
        root = os.path.realpath(kb["path"])
        path = os.path.realpath(os.path.join(root, rel))
        return path if self._path_is_within(root, path) else ""

    def _enrich_hit_with_asset(
        self, kb: dict, hit: dict, asset_cache: dict[str, list] | None = None
    ) -> dict:
        asset = self._asset_by_data_id(kb, hit.get("data_id") or "", asset_cache)
        if not asset:
            hit.setdefault("kind", "text")
            return hit
        hit.update({
            "kind": "image",
            "image_id": asset.get("image_id", ""),
            "image_path": asset.get("image_path", ""),
            "image_abspath": self._asset_image_abspath(kb, asset),
            "parent_data_id": asset.get("parent_data_id", ""),
            "parent_chunk_index": asset.get("parent_chunk_index", -1),
            "description": asset.get("description", ""),
            "table_markdown": asset.get("table_markdown", ""),
        })
        return hit

    @staticmethod
    def _doc_name_candidates(file_name: str | None = None, title: str | None = None) -> list[str]:
        vals = []
        for raw in (file_name, title):
            value = str(raw or "").strip().replace("\\", "/")
            if not value:
                continue
            vals.append(value)
            base, ext = os.path.splitext(value)
            if not ext:
                vals.extend([value + ".md", value + ".markdown"])
        out, seen = [], set()
        for value in vals:
            if value and value not in seen:
                out.append(value)
                seen.add(value)
        return out

    @staticmethod
    def _doc_matches_candidates(file_name: str, title: str, candidates: set[str]) -> bool:
        if not candidates:
            return True
        rel = str(file_name or "").replace("\\", "/")
        ttl = str(title or "").replace("\\", "/")
        values = {rel, ttl}
        if rel:
            values.add(os.path.basename(rel))
        if ttl:
            values.add(os.path.basename(ttl))
        if any(value in candidates for value in values if value):
            return True
        return any(
            (rel and rel.endswith("/" + candidate))
            or (ttl and ttl.endswith("/" + candidate))
            for candidate in candidates
        )

    @staticmethod
    def _zvec_quote(value: str) -> str:
        return "'" + str(value or "").replace("\\", "\\\\").replace("'", "''") + "'"

    def _zvec_doc_filter(self, candidates: set[str]) -> str | None:
        if not candidates:
            return None
        parts, seen = [], set()
        for candidate in sorted(candidates):
            value = str(candidate or "").replace("\\", "/").strip()
            if not value:
                continue
            exact = f"file_name = {self._zvec_quote(value)}"
            suffix = f"file_name LIKE {self._zvec_quote('%/' + value)}"
            for expression in (exact, suffix):
                if expression not in seen:
                    parts.append(expression)
                    seen.add(expression)
        return "(" + " OR ".join(parts) + ")" if parts else None

    def _doc_path_candidates(self, file_name: str | None = None, title: str | None = None) -> list[str]:
        out, seen = [], set()
        for candidate in self._doc_name_candidates(file_name=file_name, title=title):
            rel = str(candidate or "").replace("\\", "/").strip()
            if not rel:
                continue
            variants = [rel]
            if "/" not in rel:
                base, ext = os.path.splitext(rel)
                if ext:
                    variants.append(f"{base}/final/{base}{ext}")
                else:
                    variants.extend([f"{rel}/final/{rel}.md", f"{rel}/final/{rel}.markdown"])
            for variant in variants:
                if variant and variant not in seen:
                    out.append(variant)
                    seen.add(variant)
        return out

    def _search_one_zvec_field(
        self,
        kb: dict,
        query: str,
        top_k: int,
        snippet_chars: int,
        *,
        file_name: str | None,
        title: str | None,
        vector_field: str,
        score_type: str,
        error_source: str,
        asset_cache: dict[str, list],
    ) -> list[dict]:
        path = self._zvec_path(kb["path"])
        if not os.path.isdir(path):
            self._record_search_error(kb, error_source, "Zvec index directory is missing")
            return []
        doc_candidates = set(self._doc_name_candidates(file_name=file_name, title=title))
        doc_filter = self._zvec_doc_filter(doc_candidates)
        query_topk = max(top_k * self._query_factor, top_k)
        try:
            zvec = self._require_zvec()
            col = self._zvec_conn(path)
            if vector_field == "sparse_embedding":
                query_vector = self._embed_sparse_texts([query], text_type="query")[0]
            else:
                query_vector = self._embed_texts([query])[0]
            rows = col.query(
                zvec.Query(vector_field, vector=query_vector),
                topk=query_topk,
                filter=doc_filter,
                output_fields=self._output_fields,
                include_vector=False,
            )
        except Exception as error:
            self._record_search_error(kb, error_source, error)
            return []

        out, seen = [], set()
        for row in rows or []:
            fields = getattr(row, "fields", None) or {}
            rel = fields.get("file_name") or ""
            ttl = fields.get("title") or ""
            if doc_candidates and not self._doc_matches_candidates(rel, ttl, doc_candidates):
                continue
            key = (fields.get("data_id"), int(fields.get("chunk_index") or 0))
            if key in seen:
                continue
            seen.add(key)
            score = float(getattr(row, "score", 0.0) or 0.0)
            data_id = fields.get("data_id") or ""
            chunk_index = int(fields.get("chunk_index") or 0)
            body = fields.get("body") or ""
            hit = {
                "kb_id": kb["id"],
                "score": round(score, 6),
                "score_type": score_type,
                "rank": len(out) + 1,
                "data_id": data_id,
                "chunk_index": chunk_index,
                "title": ttl[:160],
                "file_name": rel,
                "ref": f"{kb['id']}/{rel}",
                "abspath": os.path.join(kb["path"], rel),
                "folder": self._folder_from_file_name(rel),
                "format": fields.get("kind") or "text",
                "n_chunks": 0,
                "kind": fields.get("kind") or "text",
                "image_path": fields.get("image_path") or "",
                "parent_data_id": fields.get("parent_data_id") or "",
                "parent_chunk_index": int(
                    fields.get("parent_chunk_index")
                    if fields.get("parent_chunk_index") is not None else -1
                ),
                "header_path": fields.get("header_path") or "",
                "chunk_role": fields.get("chunk_role") or "leaf",
                "snippet": self._snippet(body, query, snippet_chars),
                "body": body,
            }
            out.append(self._enrich_hit_with_asset(kb, hit, asset_cache))
            if len(out) >= top_k:
                break
        return out

    def _search_one_zvec(
        self, kb: dict, query: str, top_k: int, snippet_chars: int,
        *, file_name: str | None, title: str | None, asset_cache: dict[str, list]
    ) -> list[dict]:
        return self._search_one_zvec_field(
            kb, query, top_k, snippet_chars, file_name=file_name, title=title,
            vector_field="embedding", score_type="zvec", error_source="dense",
            asset_cache=asset_cache,
        )

    def _search_one_zvec_sparse(
        self, kb: dict, query: str, top_k: int, snippet_chars: int,
        *, file_name: str | None, title: str | None, asset_cache: dict[str, list]
    ) -> list[dict]:
        return self._search_one_zvec_field(
            kb, query, top_k, snippet_chars, file_name=file_name, title=title,
            vector_field="sparse_embedding", score_type="zvec_sparse", error_source="sparse",
            asset_cache=asset_cache,
        )

    def _search_exact_image_refs(
        self,
        kb: dict,
        query: str,
        top_k: int,
        snippet_chars: int,
        *,
        file_name: str | None,
        title: str | None,
        asset_cache: dict[str, list],
    ) -> list[dict]:
        refs = []
        for term in self._reference_terms(query):
            key = self._local_ref_key(term)
            if key and key not in refs:
                refs.append(key)
        if not refs:
            return []
        doc_candidates = set(self._doc_name_candidates(file_name=file_name, title=title))
        out, seen = [], set()
        for asset in self._assets_for_kb(kb, asset_cache):
            rel = asset.get("file_name") or ""
            ttl = asset.get("title") or ""
            if doc_candidates and not self._doc_matches_candidates(rel, ttl, doc_candidates):
                continue
            asset_keys = []
            for text in (ttl, asset.get("alt_text") or ""):
                key = self._local_ref_key(text)
                if key and key not in asset_keys:
                    asset_keys.append(key)
            if not any(key in refs for key in asset_keys):
                continue
            data_id = asset.get("data_id") or ""
            if not data_id or data_id in seen:
                continue
            seen.add(data_id)
            body = self._asset_body(asset)
            out.append({
                "kb_id": kb["id"],
                "score": 1.0,
                "score_type": "ref_exact",
                "rank": len(out) + 1,
                "data_id": data_id,
                "chunk_index": 0,
                "title": ttl[:160],
                "file_name": rel,
                "ref": f"{kb['id']}/{rel}",
                "abspath": os.path.join(kb["path"], rel),
                "folder": self._folder_from_file_name(rel),
                "format": "image",
                "n_chunks": 0,
                "kind": "image",
                "image_id": asset.get("image_id", ""),
                "image_path": asset.get("image_path", ""),
                "image_abspath": self._asset_image_abspath(kb, asset),
                "parent_data_id": asset.get("parent_data_id", ""),
                "parent_chunk_index": int(
                    asset.get("parent_chunk_index")
                    if asset.get("parent_chunk_index") is not None else -1
                ),
                "header_path": "",
                "chunk_role": "leaf",
                "description": asset.get("description", ""),
                "table_markdown": asset.get("table_markdown", ""),
                "snippet": self._snippet(body, query, snippet_chars),
                "body": body,
            })
            if len(out) >= top_k:
                break
        return out

    def search(
        self,
        query: str,
        top_k: int = 6,
        kb_id: str | None = None,
        snippet_chars: int = 220,
        file_name: str | None = None,
        title: str | None = None,
        mode: str = "rrf",
    ) -> list[dict]:
        """Search configured knowledge bases with dense/sparse RRF retrieval."""
        top_k = max(1, min(int(top_k), 30))
        mode = str(mode or "rrf").strip().lower()
        if mode not in ("rrf", "vector", "sparse"):
            raise ValueError("mode must be one of: rrf, vector, sparse")
        by_key = {}
        asset_cache: dict[str, list] = {}

        def add_hits(hits: list[dict], source_weight: float) -> None:
            for index, result in enumerate(hits or [], 1):
                key = (result.get("data_id"), result.get("chunk_index"))
                if not key[0]:
                    continue
                item = by_key.get(key)
                if item is None:
                    item = dict(result)
                    item["_rrf"] = 0.0
                    item["_sources"] = []
                    by_key[key] = item
                item["_rrf"] += source_weight / (60.0 + index)
                item["_sources"].append(result.get("score_type") or "unknown")
                if result.get("score_type") == "zvec_sparse" or item.get("score_type") not in ("zvec", "zvec_sparse"):
                    item.update(result)

        for kb in self._load_config():
            if kb_id and kb["id"] != kb_id:
                continue
            if not kb.get("exists"):
                self._record_search_error(kb, "config", "knowledge base directory is missing")
                continue
            add_hits(
                self._search_exact_image_refs(
                    kb, query, top_k, snippet_chars,
                    file_name=file_name, title=title, asset_cache=asset_cache,
                ),
                4.0,
            )
            if not os.path.isdir(self._zvec_path(kb["path"])):
                self._record_search_error(kb, "zvec", "Zvec index directory is missing")
                continue
            if mode in ("rrf", "vector"):
                add_hits(
                    self._search_one_zvec(
                        kb, query, top_k, snippet_chars,
                        file_name=file_name, title=title, asset_cache=asset_cache,
                    ),
                    self._vector_weight,
                )
            if mode in ("rrf", "sparse"):
                add_hits(
                    self._search_one_zvec_sparse(
                        kb, query, top_k, snippet_chars,
                        file_name=file_name, title=title, asset_cache=asset_cache,
                    ),
                    1.0,
                )
        results = list(by_key.values())
        for result in results:
            sources = []
            for source in result.pop("_sources", []):
                if source not in sources:
                    sources.append(source)
            result["score_type"] = "+".join(sources) if sources else result.get("score_type")
            result["score"] = round(result.pop("_rrf", 0.0), 6)
        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]

    def _resolve_zvec_target(
        self, data_id: str | None = None, kb_id: str | None = None, ref: str | None = None
    ) -> tuple[dict | None, str]:
        knowledge_bases = self._load_config()
        target = None
        data_id = str(data_id or "").strip()
        ref = str(ref or "").strip().replace("\\", "/")
        if data_id and "::" in data_id:
            kid = data_id.split("::", 1)[0]
            target = next((kb for kb in knowledge_bases if kb["id"] == kid), None)
        if target is None and kb_id:
            target = next((kb for kb in knowledge_bases if kb["id"] == kb_id), None)
        if target is None and ref:
            for kb in knowledge_bases:
                if ref.startswith(kb["id"] + "/"):
                    target = kb
                    rel = ref[len(kb["id"]) + 1:]
                    if not data_id:
                        data_id = f"{kb['id']}::{rel}"
                    break
        if target is None:
            target = next(
                (kb for kb in knowledge_bases if kb.get("exists") and os.path.isdir(self._zvec_path(kb["path"]))),
                None,
            )
        return target, data_id

    def _zvec_fetch_doc(self, kb: dict, data_id: str, chunk_index: int, output_fields: list[str] | None = None):
        return self._fetch_doc(
            kb, data_id, chunk_index,
            output_fields=output_fields or self._output_fields,
        )

    def document_exists(
        self, file_name: str | None = None, title: str | None = None, kb_id: str | None = None
    ) -> bool:
        for kb in self._load_config():
            if kb_id and kb["id"] != kb_id:
                continue
            if not kb.get("exists") or not os.path.isdir(self._zvec_path(kb["path"])):
                continue
            for rel in self._doc_path_candidates(file_name=file_name, title=title):
                try:
                    if self._zvec_fetch_doc(kb, f"{kb['id']}::{rel}", 0, ["data_id"]):
                        return True
                except Exception:
                    continue
        return False

    def read_chunk(
        self,
        data_id: str | None = None,
        chunk_index: int = 0,
        kb_id: str | None = None,
        ref: str | None = None,
        max_chars: int = 4000,
    ) -> str:
        target, data_id = self._resolve_zvec_target(data_id=data_id, kb_id=kb_id, ref=ref)
        if target is None or not data_id:
            return f"[未找到] data_id={data_id}"
        if not os.path.isdir(self._zvec_path(target["path"])):
            return "[索引未就绪] Zvec index directory is missing"
        try:
            doc = self._zvec_fetch_doc(target, data_id, chunk_index)
        except Exception as error:
            return f"[Zvec 读取失败] {error}"
        if not doc:
            return f"[未找到] data_id={data_id} chunk_index={chunk_index}"
        fields = getattr(doc, "fields", None) or doc or {}
        body = fields.get("body") or ""
        parent_body = ""
        if (fields.get("kind") or "text") == "text":
            parent_body = self._parent_chunk_body(
                target["path"], data_id, fields.get("parent_chunk_index", -1)
            )
        parts = []
        if parent_body and parent_body.strip() and parent_body.strip() != body.strip():
            parts.extend(["## 父级上下文\n" + parent_body.strip(), "## 命中 leaf chunk\n" + body.strip()])
        else:
            parts.append(body.strip())
        content = "\n\n".join(part for part in parts if part)
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n…[已截断，本次读取共 {len(content)} 字]"
        parent_index = int(
            fields.get("parent_chunk_index")
            if fields.get("parent_chunk_index") is not None else -1
        )
        head = (
            f"# {fields.get('title', '')}\n"
            f"file={fields.get('file_name', '')} chunk={int(fields.get('chunk_index') or chunk_index)} "
            f"role={fields.get('chunk_role') or 'leaf'} parent={parent_index}\n"
            f"header_path={fields.get('header_path') or ''}\n"
            f"{'-' * 40}\n"
        )
        return head + content

    def list_chunks(
        self,
        data_id: str | None = None,
        kb_id: str | None = None,
        ref: str | None = None,
        preview_chars: int = 80,
        limit: int = 400,
    ) -> dict:
        target, data_id = self._resolve_zvec_target(data_id=data_id, kb_id=kb_id, ref=ref)
        if target is None or not data_id:
            return {"error": f"[未找到] data_id={data_id}"}
        path = self._zvec_path(target["path"])
        if not os.path.isdir(path):
            return {"error": "[索引未就绪]"}
        try:
            col = self._zvec_conn(path)
            n = max(1, int(limit))
            ids = [self._zvec_doc_id(data_id, index) for index in range(n)]
            got = col.fetch(
                ids,
                output_fields=["data_id", "chunk_index", "file_name", "title", "body"],
                include_vector=False,
            )
        except Exception as error:
            return {"error": f"[Zvec 读取失败] {error}"}
        chunks, title, file_name = [], "", ""
        for index in range(max(1, int(limit))):
            doc = got.get(self._zvec_doc_id(data_id, index)) if isinstance(got, dict) else None
            if not doc:
                continue
            fields = getattr(doc, "fields", None) or {}
            body = fields.get("body") or ""
            title = title or fields.get("title") or ""
            file_name = file_name or fields.get("file_name") or ""
            preview = re.sub(r"\s+", " ", body).strip()[:preview_chars]
            chunks.append({
                "chunk_index": int(fields.get("chunk_index") or index),
                "chars": len(body),
                "preview": preview,
            })
        if not chunks:
            return {"error": f"[未找到] data_id={data_id} 无 chunk"}
        return {"title": title, "file_name": file_name, "n_chunks": len(chunks), "chunks": chunks}

    def read_image(
        self,
        image_id: str | None = None,
        image_path: str | None = None,
        data_id: str | None = None,
        ref: str | None = None,
        kb_id: str | None = None,
    ) -> dict:
        image_id = str(image_id or "").strip()
        image_path = str(image_path or "").strip().replace("\\", "/")
        data_id = str(data_id or "").strip()
        ref = str(ref or "").strip().replace("\\", "/")
        for kb in self._load_config():
            if kb_id and kb["id"] != kb_id:
                continue
            if not kb.get("exists"):
                continue
            for asset in self._assets_for_kb(kb):
                if image_id and asset.get("image_id") != image_id:
                    continue
                if data_id and asset.get("data_id") != data_id:
                    continue
                if image_path and asset.get("image_path") != image_path and not str(asset.get("image_path", "")).endswith(image_path):
                    continue
                if ref:
                    full_ref = f"{kb['id']}/{asset.get('file_name', '')}"
                    if ref not in (full_ref, asset.get("file_name"), asset.get("image_path")):
                        continue
                out = dict(asset)
                out["kb_id"] = kb["id"]
                out["ref"] = f"{kb['id']}/{asset.get('file_name', '')}"
                out["image_abspath"] = self._asset_image_abspath(kb, asset)
                return out
        return {"error": f"[未找到图片资产] image_id={image_id} image_path={image_path} data_id={data_id} ref={ref}"}

    def resolve_file(self, cited: str | None) -> str | None:
        """Resolve a citation to a file under one configured KB root."""
        if not cited:
            return None
        cited = str(cited).strip().strip("[]").replace("\\", "/")
        cited = re.sub(r"^kb:", "", cited)
        if "::" in cited:
            cited = cited.replace("::", "/", 1)
        cited = cited.split("#", 1)[0].strip()
        candidates = [cited]
        for kb in self._load_config():
            prefix = kb["id"] + "/"
            if cited.startswith(prefix):
                candidates.append(cited[len(prefix):])
        for kb in self._load_config():
            root = os.path.realpath(kb["path"])
            for candidate in candidates:
                full = os.path.realpath(os.path.join(root, candidate))
                if self._path_is_within(root, full) and os.path.isfile(full):
                    return full
        return None

    def list_documents(self, kb_id: str | None = None) -> list[dict]:
        docs = []
        for kb in self._load_config():
            if kb_id and kb["id"] != kb_id:
                continue
            if not kb.get("exists"):
                continue
            imported_titles = self._imported_document_titles(kb["path"])
            manifest_by_processed = {}
            for entry in self._imported_document_entries(kb["path"]):
                source_rel = str(entry.get("source") or "").replace("\\", "/")
                for processed_rel in entry.get("processed") or []:
                    rel = str(processed_rel or "").replace("\\", "/")
                    if rel:
                        manifest_by_processed[rel] = source_rel
            source_root_value = str(kb.get("source_path") or "").strip()
            source_root = os.path.realpath(source_root_value) if source_root_value else ""
            for rel, absolute_path, _mtime, size in self._scan_documents(kb["path"]):
                rel = rel.replace("\\", "/")
                source_rel = manifest_by_processed.get(rel, "")
                source_path = os.path.realpath(os.path.join(source_root, source_rel)) if source_root and source_rel else ""
                source_is_safe = bool(source_root and source_rel and self._path_is_within(source_root, source_path))
                source_exists = bool(source_is_safe and os.path.isfile(source_path))
                source_size = os.path.getsize(source_path) if source_exists else 0
                display_name = os.path.basename(source_rel) if source_rel else imported_titles.get(rel, os.path.basename(rel))
                docs.append({
                    "kb_id": kb["id"],
                    "data_id": f"{kb['id']}::{rel}",
                    "title": display_name,
                    "file_name": rel,
                    "folder": (source_rel or rel).split("/", 1)[0] if "/" in (source_rel or rel) else "",
                    "size": size,
                    "source_file_name": source_rel,
                    "source_size": source_size,
                    "source_exists": source_exists,
                    "abspath": absolute_path,
                    "ref": f"{kb['id']}/{rel}",
                })
        return docs

    def read_document(
        self,
        kb_id: str | None = None,
        data_id: str | None = None,
        file_name: str | None = None,
        ref: str | None = None,
        max_chars: int = 200000,
    ) -> dict:
        data_id = str(data_id or "")
        kb = self._kb_by_id(kb_id or (data_id.split("::", 1)[0] if "::" in data_id else ""))
        if kb is None and ref:
            kb = self._kb_by_id(str(ref).split("/", 1)[0])
        if kb is None:
            return {"error": "[未找到知识库]"}
        if data_id and "::" in data_id:
            rel = data_id.split("::", 1)[1].split("::image::", 1)[0]
        elif ref:
            ref_value = str(ref).replace("\\", "/")
            prefix = kb["id"] + "/"
            rel = ref_value[len(prefix):] if ref_value.startswith(prefix) else ref_value
        elif file_name:
            rel = str(file_name).replace("\\", "/")
        else:
            return {"error": "[未指定文档]"}
        target = os.path.realpath(os.path.join(kb["path"], rel))
        if not self._path_is_within(kb["path"], target) or not os.path.isfile(target):
            return {"error": "[未找到文档]"}
        body = self._read_textfile(target)
        truncated = len(body) > int(max_chars)
        if truncated:
            body = body[:int(max_chars)]
        return {
            "kb_id": kb["id"],
            "data_id": f"{kb['id']}::{rel}",
            "title": self._imported_document_titles(kb["path"]).get(rel, os.path.basename(rel)),
            "file_name": rel,
            "content": body,
            "truncated": truncated,
            "path": target,
            "ref": f"{kb['id']}/{rel}",
        }

    def resolve_source_document(
        self,
        kb_id: str | None = None,
        data_id: str | None = None,
        file_name: str | None = None,
        ref: str | None = None,
    ) -> dict:
        document = self.read_document(
            kb_id=kb_id, data_id=data_id, file_name=file_name, ref=ref, max_chars=1
        )
        if document.get("error"):
            return document
        kb = self._kb_by_id(document["kb_id"])
        if kb is None:
            return {"error": "[未找到知识库]"}
        processed_rel = str(document["file_name"]).replace("\\", "/")
        processed_path = os.path.realpath(os.path.join(kb["path"], processed_rel))
        package_root = os.path.realpath(os.path.dirname(kb["path"]))
        source_rel = ""
        manifest = os.path.join(package_root, "import_manifest.json")
        try:
            with open(manifest, encoding="utf-8") as handle:
                entries = (json.load(handle) or {}).get("files") or []
            for entry in entries:
                processed = [str(item).replace("\\", "/") for item in (entry.get("processed") or [])]
                if processed_rel in processed:
                    source_rel = str(entry.get("source") or "").replace("\\", "/")
                    break
        except Exception:
            pass
        source_root_value = str(kb.get("source_path") or "").strip()
        source_root = os.path.realpath(source_root_value) if source_root_value else ""
        external_path = os.path.realpath(os.path.join(source_root, source_rel)) if source_root and source_rel else ""
        external_is_safe = bool(source_root and source_rel and self._path_is_within(source_root, external_path))
        originals_root = os.path.realpath(os.path.join(package_root, "originals"))
        legacy_original_path = os.path.realpath(os.path.join(originals_root, source_rel)) if source_rel else ""
        legacy_original_is_safe = bool(source_rel and self._path_is_within(originals_root, legacy_original_path))
        if external_is_safe and os.path.isfile(external_path):
            source_path = external_path
        elif legacy_original_is_safe and os.path.isfile(legacy_original_path):
            source_path = legacy_original_path
        else:
            source_path = processed_path
        if not os.path.isfile(source_path):
            return {"error": "[未找到原始文档]"}
        return {
            "kb_id": document["kb_id"],
            "data_id": document["data_id"],
            "file_name": document["file_name"],
            "source_file_name": source_rel or processed_rel,
            "title": os.path.basename(source_rel or processed_rel),
            "path": source_path,
            "is_original": bool(source_rel and source_path in {external_path, legacy_original_path}),
            "ref": document["ref"],
        }
