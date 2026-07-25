"""GenericAgent tools backed by the unified knowledge-base facade.

This module deliberately contains only the agent-facing adapter.  Retrieval,
asset validation, and index lifecycle remain owned by :mod:`knowledge_base.backend`.
"""

import base64
import json
import mimetypes
import os

from agent_loop import StepOutcome

from .references import clean_public_text, public_reference


KB_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "在当前知识库范围内检索相关文本和图片资产，返回引用、命中正文及图片描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言问题或检索词"},
                    "mode": {
                        "type": "string",
                        "enum": ["rrf", "vector", "sparse"],
                        "default": "rrf",
                        "description": "检索通道：rrf 融合向量+稀疏（默认，通用问答用它）；vector 仅语义向量（找相近含义）；sparse 仅稀疏向量（偏关键词/术语）。注意：无论选哪种 mode，当 query 里出现明确的图表编号（如 图3-1、表4.1）时，对应图表都会被精确匹配并置顶，不受 mode 影响。",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_read",
            "description": "读取知识库命中的一个或多个连续文本分段。必须使用 kb_search 返回的 data_id 或 ref。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "chunk_index": {"type": "integer", "default": 0},
                    "span": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 8000, "default": 4000},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_list",
            "description": "列出当前范围内的知识库文档；提供 data_id 或 ref 时列出该文档的分段目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "preview_chars": {"type": "integer", "minimum": 20, "maximum": 200, "default": 80},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_image_read",
            "description": "读取知识库定位图片的图题、VLM 描述、表格 Markdown、关联正文和引用信息，不发送原图。只能使用 kb_search 返回的完整 image_id/data_id，或在当前文档范围内提供图1-1等图表编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "data_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "ref_key": {"type": "string", "description": "图1-1、表8-1等知识库返回的图表编号"},
                    "query": {"type": "string", "description": "包含图表编号的定位查询"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_image_view",
            "description": "查看已由知识库定位的原图，并将其加入下一轮多模态模型输入；不能读取任意本地路径。只能使用 kb_search 返回的完整 image_id/data_id，或在当前文档范围内提供图1-1等图表编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "data_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "ref_key": {"type": "string", "description": "图1-1、表8-1等知识库返回的图表编号"},
                    "query": {"type": "string", "description": "包含图表编号的定位查询"},
                },
            },
        },
    },
]

_MAX_INLINE_IMAGES = 3
_MAX_INLINE_IMAGE_BYTES = 12 * 1024 * 1024


class KnowledgeBaseToolsMixin:
    """Tool implementations mixed into ``GenericAgentHandler``."""

    @staticmethod
    def _kb_backend():
        from . import backend

        return backend

    @staticmethod
    def _scope_ref(value):
        return str(value or "").replace("\\", "/").strip().lstrip("/")

    @classmethod
    def _document_key(cls, data_id=None, ref=None, file_name=None):
        raw = str(data_id or "").strip()
        if "::" in raw:
            return cls._scope_ref(raw.split("::", 1)[1].split("::image::", 1)[0])
        value = cls._scope_ref(ref or file_name)
        if "/" in value:
            return value.split("/", 1)[1]
        return value

    @classmethod
    def _target_parts(cls, data_id=None, ref=None, file_name=None, kb_id=None):
        raw_id = str(data_id or "").strip()
        if "::" in raw_id:
            return raw_id.split("::", 1)[0], cls._document_key(data_id=raw_id)
        value = cls._scope_ref(ref)
        if "/" in value:
            return value.split("/", 1)[0], value.split("/", 1)[1]
        return str(kb_id or "").strip(), cls._document_key(file_name=file_name)

    def _knowledge_scope(self):
        raw = getattr(self.parent, "knowledge_scope", None)
        scope = dict(raw) if isinstance(raw, dict) else {}
        mode = str(scope.get("mode") or scope.get("kind") or "all").strip().lower()
        if mode not in {"all", "kb", "document"}:
            mode = "all"
        scope["mode"] = mode
        return scope

    def _scope_kb_id(self):
        scope = self._knowledge_scope()
        return str(scope.get("kb_id") or "").strip() or None

    def _scope_allows_target(self, data_id=None, ref=None, file_name=None, kb_id=None):
        scope = self._knowledge_scope()
        if scope["mode"] == "all":
            return True

        target_kb, target_doc = self._target_parts(
            data_id=data_id, ref=ref, file_name=file_name, kb_id=kb_id
        )
        expected_kb = str(scope.get("kb_id") or "").strip()
        if not expected_kb or not target_kb or target_kb != expected_kb:
            return False
        if scope["mode"] == "kb":
            return True

        expected_doc = self._document_key(
            data_id=scope.get("data_id"),
            ref=scope.get("ref"),
            file_name=scope.get("file_name"),
        )
        return bool(expected_doc and target_doc and expected_doc == target_doc)

    def _scope_search_kwargs(self):
        scope = self._knowledge_scope()
        if scope["mode"] == "all":
            return {}
        out = {"kb_id": self._scope_kb_id()}
        if scope["mode"] == "document":
            document = self._document_key(
                data_id=scope.get("data_id"),
                ref=scope.get("ref"),
                file_name=scope.get("file_name"),
            )
            if document:
                out["file_name"] = document
        return out

    @staticmethod
    def _clean_hit(hit):
        keep = (
            "score", "score_type", "rank", "folder", "format", "occurrence_id",
            "header_path", "snippet", "body", "description", "table_markdown",
            "related_text", "related_text_refs", "near_text", "uncertain", "caption",
            "display_label",
        )
        result = public_reference(hit, kind=hit.get("kind"))
        result.update({key: hit[key] for key in keep if key in hit and hit[key] is not None})
        for key in ("body", "snippet", "related_text"):
            if key in result:
                result[key] = clean_public_text(result[key])
        return result

    def _record_knowledge_citations(self, *items):
        """Keep stable knowledge references for the Desktop message metadata.

        The model-facing tool result can contain large text, while the UI only
        needs identifiers it can send back to the bridge.  Never copy a local
        path into this metadata.
        """
        citations = getattr(self, "_knowledge_citations", None)
        if citations is None:
            self._knowledge_citations = citations = []
        seen = {
            (
                item.get("kb_id", ""), item.get("data_id", ""),
                item.get("image_id", ""), item.get("chunk_index", -1),
            )
            for item in citations
        }
        for value in items:
            if isinstance(value, dict):
                values = [value]
            elif isinstance(value, (list, tuple)):
                values = value
            else:
                continue
            for raw in values:
                if not isinstance(raw, dict):
                    continue
                citation = public_reference(raw, kind=raw.get("kind"))
                if raw.get("display_label"):
                    citation["display_label"] = str(raw.get("display_label")).strip()
                if not any(citation.get(key) for key in ("data_id", "ref", "image_id")):
                    continue
                key = (
                    citation.get("kb_id", ""), citation.get("data_id", ""),
                    citation.get("image_id", ""), citation.get("chunk_index", -1),
                )
                if key not in seen:
                    citations.append(citation)
                    seen.add(key)
                if len(citations) >= 24:
                    return

    def take_knowledge_citations(self):
        citations = list(getattr(self, "_knowledge_citations", []) or [])
        self._knowledge_citations = []
        return citations

    def _anchor_outcome(self, args, data):
        return StepOutcome(
            data,
            next_prompt=self._get_anchor_prompt(skip=args.get("_index", 0) > 0),
        )

    def do_kb_search(self, args, response):
        query = str(args.get("query") or "").strip()
        if not query:
            return self._anchor_outcome(args, "[Error] kb_search 需要 query 参数。")
        mode = str(args.get("mode") or "rrf").strip().lower()
        if mode not in {"rrf", "vector", "sparse"}:
            return self._anchor_outcome(args, "[Error] kb_search mode 只允许 rrf、vector 或 sparse。")
        try:
            top_k = max(1, min(int(args.get("top_k", 5)), 10))
        except (TypeError, ValueError):
            top_k = 5
        try:
            hits = self._kb_backend().search(
                query, top_k=top_k, mode=mode, **self._scope_search_kwargs()
            ) or []
        except Exception as error:
            return self._anchor_outcome(args, f"[Error] kb_search 失败: {error}")
        result = {
            "query": query,
            "scope": self._knowledge_scope(),
            "hits": [self._clean_hit(hit) for hit in hits],
        }
        # Search results are candidates.  An image becomes a citation only
        # after kb_image_read/view successfully resolves it, so unrelated
        # vector hits cannot leak into the final answer's citation list.
        self._record_knowledge_citations(
            [hit for hit in hits if hit.get("kind") != "image"]
        )
        yield "[Info] kb_search done.\n"
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def do_kb_read(self, args, response):
        data_id = str(args.get("data_id") or "").strip() or None
        ref = str(args.get("ref") or "").strip() or None
        if not data_id and not ref:
            return self._anchor_outcome(args, "[Error] kb_read 需要 data_id 或 ref。")
        if not self._scope_allows_target(data_id=data_id, ref=ref):
            return self._anchor_outcome(args, "[Error] kb_read 目标不在当前会话的知识库范围内。")
        try:
            start = max(0, int(args.get("chunk_index", 0)))
            span = max(1, min(int(args.get("span", 1)), 5))
            max_chars = max(500, min(int(args.get("max_chars", 4000)), 8000))
        except (TypeError, ValueError):
            start, span, max_chars = 0, 1, 4000
        parts = []
        backend = self._kb_backend()
        for index in range(start, start + span):
            content = backend.read_chunk(
                data_id=data_id,
                ref=ref,
                chunk_index=index,
                kb_id=self._scope_kb_id(),
                max_chars=max_chars,
            )
            if str(content).startswith("[未找到]"):
                break
            parts.append(str(content))
        if parts:
            self._record_knowledge_citations(
                backend.reference_for_chunk(
                    data_id=data_id,
                    ref=ref,
                    kb_id=self._scope_kb_id(),
                    chunk_index=start,
                )
            )
        yield "[Info] kb_read done.\n"
        return self._anchor_outcome(args, "\n\n".join(parts) or "[kb_read] 未读到内容。")

    def do_kb_list(self, args, response):
        data_id = str(args.get("data_id") or "").strip() or None
        ref = str(args.get("ref") or "").strip() or None
        if data_id or ref:
            if not self._scope_allows_target(data_id=data_id, ref=ref):
                return self._anchor_outcome(args, "[Error] kb_list 目标不在当前会话的知识库范围内。")
            try:
                preview = max(20, min(int(args.get("preview_chars", 80)), 200))
            except (TypeError, ValueError):
                preview = 80
            result = self._kb_backend().list_chunks(
                data_id=data_id,
                ref=ref,
                kb_id=self._scope_kb_id(),
                preview_chars=preview,
            )
        else:
            documents = self._kb_backend().list_documents(kb_id=self._scope_kb_id())
            if self._knowledge_scope()["mode"] == "document":
                expected = self._document_key(
                    data_id=self._knowledge_scope().get("data_id"),
                    ref=self._knowledge_scope().get("ref"),
                    file_name=self._knowledge_scope().get("file_name"),
                )
                documents = [
                    document for document in documents
                    if self._document_key(
                        data_id=document.get("data_id"),
                        ref=document.get("ref"),
                        file_name=document.get("file_name"),
                    ) == expected
                ]
            self._record_knowledge_citations(documents)
            result = {"documents": documents}
        yield "[Info] kb_list done.\n"
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def _read_image_asset(self, args):
        lookup = {
            key: str(args.get(key) or "").strip() or None
            for key in ("image_id", "data_id", "ref", "ref_key", "query")
        }
        if not any(lookup.values()):
            return None, "[Error] 需要 kb_search 返回的 image_id/data_id，或图1-1等图表编号。"
        # image_id alone does not carry the owning document.  In a
        # restricted scope the backend query is first constrained by kb_id;
        # the returned asset is then checked against the document scope below.
        if (lookup["data_id"] or lookup["ref"]) and not self._scope_allows_target(
            data_id=lookup["data_id"], ref=lookup["ref"]
        ):
            return None, "[Error] 图片目标不在当前会话的知识库范围内。"
        try:
            info = self._kb_backend().read_image(kb_id=self._scope_kb_id(), **lookup)
        except Exception as error:
            return None, f"[Error] 知识库图片读取失败: {error}"
        if not isinstance(info, dict) or info.get("error"):
            return None, (info or {}).get("error", "[未找到图片资产]")
        if not self._scope_allows_target(
            data_id=info.get("data_id"),
            ref=info.get("ref"),
            file_name=info.get("file_name"),
            kb_id=info.get("kb_id"),
        ):
            return None, "[Error] 图片目标不在当前会话的知识库范围内。"
        return info, None

    @staticmethod
    def _public_image(info):
        result = public_reference(info, kind="image")
        for key in (
            "description", "table_markdown", "related_text", "related_text_refs",
            "near_text", "uncertain", "analysis_error", "caption", "display_label",
        ):
            if key in info:
                result[key] = info.get(key)
        return result

    def do_kb_image_read(self, args, response):
        info, error = self._read_image_asset(args)
        if error:
            return self._anchor_outcome(args, error)
        self._record_knowledge_citations(info)
        yield "[Info] kb_image_read done.\n"
        return self._anchor_outcome(args, json.dumps(self._public_image(info), ensure_ascii=False, indent=2))

    @staticmethod
    def _image_data_url(path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _queue_image_view(self, info):
        pending = getattr(self, "_pending_inline_blocks", None)
        if pending is None:
            self._pending_inline_blocks = pending = []
        if len(pending) // 2 >= _MAX_INLINE_IMAGES:
            return "[kb_image_view] 本轮最多查看 3 张图片。"
        path = str(info.get("image_abspath") or "")
        if not os.path.isfile(path):
            return "[kb_image_view] 图片文件不存在，无法注入原图；可使用已返回的图片描述回答。"
        try:
            size = os.path.getsize(path)
            if size > _MAX_INLINE_IMAGE_BYTES:
                return "[kb_image_view] 图片文件过大，无法安全注入当前模型上下文；可使用图片描述回答。"
            data_url = self._image_data_url(path)
        except OSError as error:
            return f"[kb_image_view] 图片读取失败: {error}"
        pending.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "[知识库图片原图]\n"
                        f"图题: {info.get('display_label') or info.get('title') or info.get('alt_text') or ''}\n"
                        f"来源: {info.get('source_file_name') or ''}\n"
                        f"引用: {info.get('citation_label') or info.get('title') or '图片'}\n"
                        "请直接查看紧随其后的原图，并结合工具结果回答用户问题。"
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        return None

    def do_kb_image_view(self, args, response):
        info, error = self._read_image_asset(args)
        if error:
            return self._anchor_outcome(args, error)
        error = self._queue_image_view(info)
        if error:
            return self._anchor_outcome(args, error)
        self._record_knowledge_citations(info)
        result = public_reference(info, kind="image")
        result.update({
            "status": "attached",
            "message": "原图已加入下一轮模型输入。",
            "description": info.get("description", ""),
            "table_markdown": info.get("table_markdown", ""),
            "related_text": info.get("related_text", ""),
        })
        yield "[Info] kb_image_view done.\n"
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))
