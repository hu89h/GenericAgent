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

from .references import clean_public_text, public_reference, section_label, with_reference


_SPLIT = re.compile(r"\s+")
_FIGURE_REF = re.compile(
    r"[图表]\s*[0-9０-９一二三四五六七八九十百]+"
    r"(?:[-－—–.．][0-9０-９一二三四五六七八九十百]+){0,3}"
)
_DEFAULT_OUTPUT_FIELDS = [
    "data_id", "chunk_index", "kb_id", "file_name", "title", "kind",
    "image_path", "source_data_id", "source_chunk_index", "header_path", "body",
    # Image-asset columns (empty for text records) — let search/read_image
    # resolve images straight from the zvec row, no catalog join.
    "image_id", "ref_key", "display_label", "caption", "description",
    "table_markdown", "source_file_name", "uncertain", "analysis_error",
    "related_text", "near_text",
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
        local_ref_key: Callable[[str], str],
        record_search_error: Callable[[dict, str, Any], None],
        imported_document_titles: Callable[[str], dict],
        imported_document_entries: Callable[[str], list[dict]] | None = None,
        query_factor: int = 4,
        vector_weight: float = 1.2,
        sparse_weight: float = 1.0,
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
        self._local_ref_key = local_ref_key
        self._record_search_error = record_search_error
        self._imported_document_titles = imported_document_titles
        self._imported_document_entries = imported_document_entries or (lambda _path: [])
        self._query_factor = max(1, int(query_factor))
        self._vector_weight = float(vector_weight)
        self._sparse_weight = float(sparse_weight)
        self._snippet_width = max(1, int(snippet_width))
        self._output_fields = list(output_fields or _DEFAULT_OUTPUT_FIELDS)

    @staticmethod
    def _path_is_within(root: str, path: str) -> bool:
        try:
            root_real = os.path.realpath(root)
            path_real = os.path.realpath(path)
            return path_real == root_real or os.path.commonpath((root_real, path_real)) == root_real
        except (OSError, ValueError):
            return False

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

    def clear_asset_cache(self, path: str | None = None) -> None:
        """No-op retained for the build/backend call sites.

        Route 甲 removed the in-memory ``_AssetCatalog`` — image metadata now
        lives in the widened zvec row, so there is no per-KB asset cache left to
        invalidate.  The method stays so backend build/delete flows need not
        special-case its absence.
        """

    def _image_abspath(self, kb: dict, image_path: str) -> str:
        """Resolve an image only under the configured knowledge-base root."""
        rel = str(image_path or "").replace("\\", "/").lstrip("/")
        if not rel:
            return ""
        root = os.path.realpath(kb["path"])
        path = os.path.realpath(os.path.join(root, rel))
        return path if self._path_is_within(root, path) else ""

    def _image_fields_from_row(self, kb: dict, fields: dict) -> dict:
        """Project an image zvec row onto the public image field set.

        Route 甲: the widened zvec row is self-sufficient, so every image field
        is read straight from it — no ``image_assets.json`` catalog join.
        ``uncertain`` was JSON-encoded on insert (zvec columns are scalar
        STRING), so it is decoded back to its list/dict form here.
        """
        uncertain = fields.get("uncertain") or ""
        if isinstance(uncertain, str) and uncertain:
            try:
                uncertain = json.loads(uncertain)
            except (ValueError, TypeError):
                uncertain = [uncertain]
        image_path = str(fields.get("image_path") or "")
        return {
            "kind": "image",
            "image_id": fields.get("image_id") or "",
            "image_path": image_path,
            "image_abspath": self._image_abspath(kb, image_path),
            "source_data_id": fields.get("source_data_id") or "",
            "source_chunk_index": int(
                fields.get("source_chunk_index")
                if fields.get("source_chunk_index") is not None else -1
            ),
            "description": fields.get("description") or "",
            "table_markdown": fields.get("table_markdown") or "",
            "caption": fields.get("caption") or "",
            "display_label": fields.get("display_label") or "",
            "ref_key": fields.get("ref_key") or "",
            "source_file_name": fields.get("source_file_name") or "",
            "related_text": fields.get("related_text") or "",
            "near_text": fields.get("near_text") or "",
            "uncertain": uncertain,
            "analysis_error": fields.get("analysis_error") or "",
        }

    def _enrich_hit_with_asset(self, kb: dict, hit: dict, fields: dict) -> dict:
        """Fold image columns from the zvec row into a search hit in place.

        Text rows leave ``kind`` untouched; image rows (``kind == "image"``)
        gain the full image field set and a display-label title.  The hit's
        ``ref`` already equals ``{kb_id}/{file_name}`` (the old ``source_ref``),
        so no ref override is needed.
        """
        if (fields.get("kind") or "") != "image":
            hit.setdefault("kind", "text")
            return hit
        hit.update(self._image_fields_from_row(kb, fields))
        if hit.get("display_label"):
            hit["title"] = hit["display_label"]
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

    def _query_image_rows(
        self,
        kb: dict,
        *,
        ref_keys: list[str] | None = None,
        source_data_id: str | None = None,
        topk: int = 100,
    ) -> list[dict]:
        """Return image rows (``kind='image'``) from a KB via a pure zvec filter.

        Route 甲: image lookup by figure number or owning document is a scalar
        filter over the widened zvec columns, so no query vector is needed.
        ``ref_keys`` are OR-joined (``ref_key IN (...)``) and AND-ed with an
        optional ``source_data_id`` constraint.  Returns each row's ``fields``
        dict; failures degrade to an empty list (logged as a search error).
        """
        path = self._zvec_path(kb["path"])
        if not os.path.isdir(path):
            return []
        clauses = ["kind = 'image'"]
        keys = [k for k in (ref_keys or []) if k]
        if keys:
            quoted = ", ".join(self._zvec_quote(k) for k in dict.fromkeys(keys))
            clauses.append(f"ref_key IN ({quoted})")
        if source_data_id:
            clauses.append(f"source_data_id = {self._zvec_quote(source_data_id)}")
        filter_expr = " AND ".join(clauses)
        try:
            col = self._zvec_conn(path)
            rows = col.query(
                topk=max(1, int(topk)),
                filter=filter_expr,
                output_fields=self._output_fields,
                include_vector=False,
            )
        except Exception as error:
            self._record_search_error(kb, "image_filter", error)
            return []
        return [getattr(row, "fields", None) or {} for row in (rows or [])]

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
        query_vector,
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
            body = clean_public_text(fields.get("body") or "")
            # Reference fields are normalized once for the whole result set in
            # search(); building this raw hit and enriching it is enough here.
            hit = {
                "kb_id": kb["id"],
                "score": round(score, 6),
                "score_type": score_type,
                "data_id": data_id,
                "chunk_index": chunk_index,
                "title": ttl[:160],
                "file_name": rel,
                "ref": f"{kb['id']}/{rel}",
                "abspath": os.path.join(kb["path"], rel),
                "kind": fields.get("kind") or "text",
                "image_path": fields.get("image_path") or "",
                "source_data_id": fields.get("source_data_id") or "",
                "source_chunk_index": int(
                    fields.get("source_chunk_index")
                    if fields.get("source_chunk_index") is not None else -1
                ),
                "header_path": fields.get("header_path") or "",
                "snippet": self._snippet(body, query, snippet_chars),
                "body": body,
            }
            out.append(self._enrich_hit_with_asset(kb, hit, fields))
            if len(out) >= top_k:
                break
        return out

    def _search_one_zvec(
        self, kb: dict, query: str, top_k: int, snippet_chars: int,
        *, file_name: str | None, title: str | None, query_vector
    ) -> list[dict]:
        return self._search_one_zvec_field(
            kb, query, top_k, snippet_chars, file_name=file_name, title=title,
            vector_field="embedding", score_type="zvec", error_source="dense",
            query_vector=query_vector,
        )

    def _search_one_zvec_sparse(
        self, kb: dict, query: str, top_k: int, snippet_chars: int,
        *, file_name: str | None, title: str | None, query_vector
    ) -> list[dict]:
        return self._search_one_zvec_field(
            kb, query, top_k, snippet_chars, file_name=file_name, title=title,
            vector_field="sparse_embedding", score_type="zvec_sparse", error_source="sparse",
            query_vector=query_vector,
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
    ) -> list[dict]:
        refs = []
        for term in self._reference_terms(query):
            key = self._local_ref_key(term)
            if key and key not in refs:
                refs.append(key)
        if not refs:
            return []
        doc_candidates = set(self._doc_name_candidates(file_name=file_name, title=title))
        # Route 甲: exact figure-number hits come straight from a zvec filter on
        # the widened ``ref_key`` column (normalized once at ingest), replacing
        # the old in-memory ``by_ref_key`` catalog index.
        rows = self._query_image_rows(kb, ref_keys=refs, topk=max(top_k * self._query_factor, top_k))
        out, seen = [], set()
        for fields in rows:
            rel = fields.get("file_name") or ""
            ttl = fields.get("title") or ""
            if doc_candidates and not self._doc_matches_candidates(rel, ttl, doc_candidates):
                continue
            data_id = fields.get("data_id") or ""
            if not data_id or data_id in seen:
                continue
            seen.add(data_id)
            out.append(self._asset_hit(kb, fields, query=query, snippet_chars=snippet_chars))
            if len(out) >= top_k:
                break
        return out

    def _asset_hit(self, kb: dict, fields: dict, *, query: str, snippet_chars: int) -> dict:
        """Build an exact-ref image hit from a zvec image row.

        Reference fields are normalized once in search(); this only assembles
        the raw payload.  ``body`` comes from the row (composed at build time by
        ``asset_body``), keeping the snippet consistent with dense/sparse hits.
        """
        rel = fields.get("file_name") or ""
        body = clean_public_text(fields.get("body") or "")
        hit = {
            "kb_id": kb["id"],
            "score": 1.0,
            "score_type": "ref_exact",
            "data_id": fields.get("data_id") or "",
            "chunk_index": 0,
            "title": (fields.get("display_label") or fields.get("title") or "图片")[:160],
            "file_name": rel,
            "ref": f"{kb['id']}/{rel}",
            "abspath": os.path.join(kb["path"], rel),
            "header_path": "",
            "snippet": self._snippet(body, query, snippet_chars),
            "body": body,
        }
        hit.update(self._image_fields_from_row(kb, fields))
        return hit

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
                # The fused RRF score always accumulates above; this only picks
                # which channel's metadata (snippet/body/title) the merged hit
                # displays.  Prefer a sparse hit, then any zvec hit, over an
                # exact-ref hit: exact-ref carries only a thin reference payload,
                # while zvec hits carry the full body/snippet.  Ordering of
                # channels in add_hits does not matter because of this rule.
                if result.get("score_type") == "zvec_sparse" or item.get("score_type") not in ("zvec", "zvec_sparse"):
                    item.update(result)

        # Embed the query once up front: the vector is identical across every
        # knowledge base, so computing it inside the per-KB loop only multiplied
        # cache reads (or network calls when the cache is disabled). A failure
        # here is KB-independent, so it disables that whole channel rather than
        # aborting the search.
        dense_vector = sparse_vector = None
        if mode in ("rrf", "vector"):
            try:
                dense_vector = self._embed_texts([query])[0]
            except Exception as error:
                self._record_search_error({"id": ""}, "dense", error)
        if mode in ("rrf", "sparse"):
            try:
                sparse_vector = self._embed_sparse_texts([query], text_type="query")[0]
            except Exception as error:
                self._record_search_error({"id": ""}, "sparse", error)

        for kb in self._load_config():
            if kb_id and kb["id"] != kb_id:
                continue
            if not kb.get("exists"):
                self._record_search_error(kb, "config", "knowledge base directory is missing")
                continue
            # Exact figure/table-number matches (图3-1, 表4.1, ...) are injected
            # for every mode, on purpose — including mode="vector" and
            # mode="sparse".  When the user names a specific figure/table, that
            # asset must pin to the top regardless of the retrieval channel, so
            # this runs before the mode gate below.  Its high weight (4.0) makes
            # such a hit outrank ordinary dense/sparse results after RRF fusion.
            add_hits(
                self._search_exact_image_refs(
                    kb, query, top_k, snippet_chars,
                    file_name=file_name, title=title,
                ),
                4.0,
            )
            if not os.path.isdir(self._zvec_path(kb["path"])):
                self._record_search_error(kb, "zvec", "Zvec index directory is missing")
                continue
            if mode in ("rrf", "vector") and dense_vector is not None:
                add_hits(
                    self._search_one_zvec(
                        kb, query, top_k, snippet_chars,
                        file_name=file_name, title=title, query_vector=dense_vector,
                    ),
                    self._vector_weight,
                )
            if mode in ("rrf", "sparse") and sparse_vector is not None:
                add_hits(
                    self._search_one_zvec_sparse(
                        kb, query, top_k, snippet_chars,
                        file_name=file_name, title=title, query_vector=sparse_vector,
                    ),
                    self._sparse_weight,
                )
        results = list(by_key.values())
        for result in results:
            sources = []
            for source in result.pop("_sources", []):
                if source not in sources:
                    sources.append(source)
            result["score_type"] = "+".join(sources) if sources else result.get("score_type")
            result["score"] = round(result.pop("_rrf", 0.0), 6)
            result.update(with_reference(result, kind=result.get("kind")))
            result.pop("abspath", None)
            result.pop("image_abspath", None)
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
        content = clean_public_text(body)
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n…[已截断，本次读取共 {len(content)} 字]"
        head = f"# {fields.get('title', '')}\n"
        section = section_label(fields.get("header_path"))
        if section:
            head += f"章节：{section}\n"
        head += f"分段：{int(fields.get('chunk_index') or chunk_index)}\n{'-' * 40}\n"
        return head + content

    def reference_for_chunk(
        self,
        data_id: str | None = None,
        chunk_index: int = 0,
        kb_id: str | None = None,
        ref: str | None = None,
    ) -> dict:
        """Return a stable citation for a retrieved text chunk."""
        target, resolved_data_id = self._resolve_zvec_target(
            data_id=data_id, kb_id=kb_id, ref=ref
        )
        fallback = {
            "kind": "document",
            "kb_id": kb_id or "",
            "data_id": resolved_data_id or data_id or "",
            "chunk_index": chunk_index,
            "ref": ref or "",
        }
        if target is None or not resolved_data_id:
            return public_reference(fallback, kind="document")
        try:
            doc = self._zvec_fetch_doc(
                target,
                resolved_data_id,
                chunk_index,
                output_fields=["data_id", "chunk_index", "file_name", "title", "header_path"],
            )
        except Exception:
            doc = None
        fields = getattr(doc, "fields", None) or doc or {}
        file_name = str(fields.get("file_name") or "")
        source_title = self._imported_document_titles(target["path"]).get(file_name, "")
        return public_reference({
            **fallback,
            "kb_id": target["id"],
            "data_id": fields.get("data_id") or resolved_data_id,
            "chunk_index": fields.get("chunk_index") if fields else chunk_index,
            "title": source_title or fields.get("title") or os.path.basename(file_name),
            "file_name": file_name,
            "source_file_name": source_title or fields.get("title") or file_name,
            "header_path": fields.get("header_path") or "",
            "ref": f"{target['id']}/{file_name}" if file_name else (ref or ""),
        }, kind="document")

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
        # Chunks are written contiguously as 0..n-1 (build.py enumerates them;
        # image assets carry their own ::image:: data_id and never interleave),
        # so probe in windows and stop at the first window that comes back short
        # instead of always fetching `limit` (400) doc ids for a small document.
        limit = max(1, int(limit))
        window = min(64, limit)
        chunks, title, file_name = [], "", ""
        try:
            col = self._zvec_conn(path)
            start = 0
            while start < limit:
                end = min(start + window, limit)
                ids = [self._zvec_doc_id(data_id, index) for index in range(start, end)]
                got = col.fetch(
                    ids,
                    output_fields=["data_id", "chunk_index", "file_name", "title", "body"],
                    include_vector=False,
                )
                found = 0
                for index in range(start, end):
                    doc = got.get(self._zvec_doc_id(data_id, index)) if isinstance(got, dict) else None
                    if not doc:
                        continue
                    found += 1
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
                if found < len(ids):
                    break
                start = end
        except Exception as error:
            return {"error": f"[Zvec 读取失败] {error}"}
        if not chunks:
            return {"error": f"[未找到] data_id={data_id} 无 chunk"}
        return {"title": title, "file_name": file_name, "n_chunks": len(chunks), "chunks": chunks}

    def read_image(
        self,
        data_id: str | None = None,
        ref_key: str | None = None,
        kb_id: str | None = None,
    ) -> dict:
        """Resolve one image from the widened zvec rows.

        Route 甲/Q3: the only locators are ``data_id`` (occurrence-level
        ``::image::`` id → O(1) primary-key fetch; or document-level id →
        ``source_data_id`` filter) and ``ref_key`` (figure number → ``ref_key``
        filter, normalized once at ingest).  ``kb_id`` narrows the scope.
        """
        data_id = str(data_id or "").strip()
        ref_key = self._local_ref_key(str(ref_key or "").strip())
        if not data_id and not ref_key:
            return {
                "error_code": "image_target_missing",
                "error": "[图片目标缺失] 请提供知识库返回的 data_id 或图表编号。",
            }

        matches = []  # list[(kb, fields)]
        for kb in self._load_config():
            if kb_id and kb["id"] != kb_id:
                continue
            if not kb.get("exists"):
                continue
            if data_id and "::image::" in data_id:
                # Occurrence-level id encodes its owning KB, so only the right
                # KB's index holds this primary key — a direct fetch is enough.
                doc = self._fetch_doc(kb, data_id, 0)
                fields = (getattr(doc, "fields", None) or {}) if doc else {}
                if not fields or (fields.get("kind") or "") != "image":
                    continue
                if ref_key and self._local_ref_key(fields.get("ref_key") or "") != ref_key:
                    continue
                matches.append((kb, fields))
            else:
                # Document-level data_id and/or figure number: a scalar filter
                # over the widened image columns (AND-ed when both are given).
                for fields in self._query_image_rows(
                    kb,
                    ref_keys=[ref_key] if ref_key else None,
                    source_data_id=data_id or None,
                ):
                    matches.append((kb, fields))
        if not matches:
            return {
                "error_code": "image_not_found",
                "error": f"[未找到图片资产] data_id={data_id} ref_key={ref_key}",
            }
        if len(matches) > 1:
            candidates = []
            for kb, fields in matches[:8]:
                candidates.append(public_reference({
                    "kb_id": kb["id"],
                    "data_id": fields.get("data_id", ""),
                    "image_id": fields.get("image_id", ""),
                    "ref_key": fields.get("ref_key", ""),
                    "display_label": fields.get("display_label") or fields.get("title") or "图片",
                    "source_file_name": fields.get("source_file_name") or fields.get("file_name") or "",
                }, kind="image"))
            return {
                "error_code": "image_ambiguous",
                "error": "[图片目标不明确] 请使用 kb_search 返回的完整 data_id。",
                "candidates": candidates,
            }
        kb, fields = matches[0]
        rel = fields.get("file_name") or ""
        out = {
            "kb_id": kb["id"],
            "data_id": fields.get("data_id") or "",
            "chunk_index": 0,
            "title": fields.get("title") or "",
            "file_name": rel,
            "ref": f"{kb['id']}/{rel}",
            "source_ref": f"{kb['id']}/{rel}",
        }
        out.update(self._image_fields_from_row(kb, fields))
        if not out.get("source_file_name"):
            source_titles = self._imported_document_titles(kb["path"])
            out["source_file_name"] = source_titles.get(rel, "")
        if not out.get("ref_key"):
            for value in (out.get("title"), out.get("near_text")):
                out["ref_key"] = self._local_ref_key(value or "")
                if out["ref_key"]:
                    break
        if not out.get("caption") and out.get("title") and str(out.get("title")).lower() != "image":
            out["caption"] = out.get("title")
        if not out.get("display_label"):
            out["display_label"] = out.get("caption") or out.get("ref_key") or out.get("title") or "图片"
        return with_reference(out, kind="image")

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
                        if rel.startswith("processed/"):
                            manifest_by_processed[rel[len("processed/"):]] = source_rel
                        else:
                            manifest_by_processed[f"processed/{rel}"] = source_rel
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
                docs.append(with_reference({
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
                }, kind="document"))
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
            return {"error_code": "knowledge_base_not_found", "error": "[未找到知识库]"}
        if data_id and "::" in data_id:
            rel = data_id.split("::", 1)[1].split("::image::", 1)[0]
        elif ref:
            ref_value = str(ref).replace("\\", "/")
            prefix = kb["id"] + "/"
            rel = ref_value[len(prefix):] if ref_value.startswith(prefix) else ref_value
        elif file_name:
            rel = str(file_name).replace("\\", "/")
        else:
            return {"error_code": "document_target_missing", "error": "[未指定文档]"}
        target = os.path.realpath(os.path.join(kb["path"], rel))
        if not self._path_is_within(kb["path"], target) or not os.path.isfile(target):
            return {"error_code": "document_not_found", "error": "[未找到文档]"}
        body = self._read_textfile(target)
        truncated = len(body) > int(max_chars)
        if truncated:
            body = body[:int(max_chars)]
        return with_reference({
            "kb_id": kb["id"],
            "data_id": f"{kb['id']}::{rel}",
            "title": self._imported_document_titles(kb["path"]).get(rel, os.path.basename(rel)),
            "file_name": rel,
            "content": body,
            "truncated": truncated,
            "path": target,
            "ref": f"{kb['id']}/{rel}",
        }, kind="document")

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
        if external_is_safe and os.path.isfile(external_path):
            source_path = external_path
        else:
            source_path = processed_path
        if not os.path.isfile(source_path):
            return {"error": "[未找到原始文档]"}
        return with_reference({
            "kb_id": document["kb_id"],
            "data_id": document["data_id"],
            "file_name": document["file_name"],
            "source_file_name": source_rel or processed_rel,
            "title": os.path.basename(source_rel or processed_rel),
            "path": source_path,
            "is_original": bool(source_rel and source_path == external_path),
            "ref": document["ref"],
        }, kind="document")
