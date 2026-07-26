"""GenericAgent tools backed by the unified knowledge-base facade.

This module deliberately contains only the agent-facing adapter.  Retrieval,
asset validation, and index lifecycle remain owned by :mod:`knowledge_base.backend`.
"""

import json
import os

from agent_loop import StepOutcome

from .references import clean_public_text, public_reference


KB_RESPONSE_SOURCE_INSTRUCTIONS = """
[SYSTEM] 知识库回答来源规则：
- 如果最终回答实际使用了知识库信息，在答案末尾自行添加“**信息来源**”段。
- 来源只能使用工具结果中的 source_hint，只列出实际支撑回答的来源，不要罗列所有搜索命中。
- 同一原始文档必须合并为一条项目并对章节或图号去重，禁止重复书写文档名。格式固定为：
  **信息来源**
  - 《原始文档名》：“章节一”“章节二”
- data_id、ref、chunk_index、内部路径、处理后 Markdown 文件名、哈希前缀和检索诊断仅供工具调用，禁止出现在面向用户的回答中。
- 未使用知识库信息时不要添加“信息来源”段。
""".strip()


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
                        "description": "必须显式选择检索通道：rrf 融合向量+稀疏（通用问答）；vector 仅语义向量（找相近含义）；sparse 仅稀疏向量（偏关键词/术语）。无论选哪种 mode，当 query 里出现明确的图表编号（如 图3-1、表4.1）时，对应图表都会作为独立确定性信号参与排序。",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["query", "mode"],
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
            "description": "读取知识库定位图片的原图、图题、VLM 描述、表格 Markdown、关联正文和引用信息。调用成功后原图会自动加入下一轮多模态输入。只有当用户要求查看图像细节，或 kb_search 返回的描述不足以回答时才调用，避免浪费视觉 token。不能读取任意本地路径；只能使用 kb_search 返回的完整 data_id，或当前文档范围内的图表编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "ref_key": {"type": "string", "description": "图1-1、表8-1等知识库返回的图表编号"},
                },
            },
        },
    },
]

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
        if mode not in {"none", "all", "kb", "document"}:
            mode = "all"
        scope["mode"] = mode
        return scope

    def _scope_kb_id(self):
        scope = self._knowledge_scope()
        return str(scope.get("kb_id") or "").strip() or None

    def _scope_allows_target(self, data_id=None, ref=None, file_name=None, kb_id=None):
        scope = self._knowledge_scope()
        if scope["mode"] == "none":
            return False
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
        if scope["mode"] == "none":
            return {"kb_id": "__knowledge_disabled__"}
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
            "score", "score_type",
            "matched_by", "channel_ranks", "final_rank",
            "snippet", "body", "description", "table_markdown",
            "related_text", "near_text", "uncertain", "caption",
            "display_label",
        )
        result = public_reference(hit, kind=hit.get("kind"))
        result.update({key: hit[key] for key in keep if key in hit and hit[key] is not None})
        result["source_hint"] = KnowledgeBaseToolsMixin._source_hint(result)
        for key in ("body", "snippet", "related_text"):
            if key in result:
                result[key] = clean_public_text(result[key])
        return result

    @staticmethod
    def _source_hint(reference):
        name = str(
            reference.get("source_file_name")
            or reference.get("title")
            or ""
        ).strip()
        if not name:
            return ""
        if reference.get("kind") == "image":
            section = str(
                reference.get("display_label")
                or reference.get("caption")
                or reference.get("ref_key")
                or reference.get("source_section")
                or ""
            ).strip()
        else:
            section = str(reference.get("source_section") or "").strip()
        return f"《{name}》：“{section}”" if section else f"《{name}》"

    @classmethod
    def _clean_document(cls, document):
        result = public_reference(document, kind="document")
        for key in ("folder", "size", "source_size", "source_exists"):
            if key in document:
                result[key] = document.get(key)
        result["source_hint"] = cls._source_hint(result)
        return result

    def _record_knowledge_citations(self, *items):
        """Keep only image references for Desktop open-image actions.

        Text provenance is written by the agent as a safe source footer.  The
        Desktop metadata channel remains only for real image assets that have
        a meaningful open action.
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
                if citation.get("kind") != "image" or not citation.get("image_id"):
                    continue
                if raw.get("display_label"):
                    citation["display_label"] = str(raw.get("display_label")).strip()
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
        next_prompt = self._get_anchor_prompt(skip=args.get("_index", 0) > 0)
        return StepOutcome(
            data,
            next_prompt=f"{next_prompt}\n{KB_RESPONSE_SOURCE_INSTRUCTIONS}\n",
        )

    def do_kb_search(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(args, "[Error] 当前对话未启用知识库。")
        query = str(args.get("query") or "").strip()
        if not query:
            return self._anchor_outcome(args, "[Error] kb_search 需要 query 参数。")
        mode = str(args.get("mode") or "").strip().lower()
        if mode not in {"rrf", "vector", "sparse"}:
            return self._anchor_outcome(args, "[Error] kb_search mode 只允许 rrf、vector 或 sparse。")
        try:
            top_k = max(1, min(int(args.get("top_k", 5)), 10))
        except (TypeError, ValueError):
            top_k = 5
        try:
            search_result = self._kb_backend().search(
                query, top_k=top_k, mode=mode, **self._scope_search_kwargs()
            ) or {}
        except Exception as error:
            return self._anchor_outcome(args, f"[Error] kb_search 失败: {error}")
        hits = search_result.get("results") or []
        result = {
            "query": query,
            "mode": search_result.get("mode") or mode,
            "scope": self._knowledge_scope(),
            "hits": [self._clean_hit(hit) for hit in hits],
            "diagnostics": search_result.get("diagnostics") or [],
        }
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def do_kb_read(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(args, "[Error] 当前对话未启用知识库。")
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
        source_hints = []
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
            reference = backend.reference_for_chunk(
                data_id=data_id,
                ref=ref,
                kb_id=self._scope_kb_id(),
                chunk_index=index,
            )
            hint = self._source_hint(reference)
            if hint and hint not in source_hints:
                source_hints.append(hint)
        if parts:
            source_note = "\n".join(f"- {hint}" for hint in source_hints)
            if source_note:
                parts.insert(0, f"[可公开使用的原始来源口径]\n{source_note}")
        return self._anchor_outcome(args, "\n\n".join(parts) or "[kb_read] 未读到内容。")

    def do_kb_list(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(args, "[Error] 当前对话未启用知识库。")
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
            if isinstance(result, dict) and not result.get("error"):
                source = self._kb_backend().reference_for_chunk(
                    data_id=data_id,
                    ref=ref,
                    kb_id=self._scope_kb_id(),
                    chunk_index=0,
                )
                result = dict(result)
                result.pop("file_name", None)
                result["title"] = (
                    source.get("source_file_name")
                    or source.get("title")
                    or result.get("title")
                    or ""
                )
                result["source_hint"] = self._source_hint(source)
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
            result = {
                "documents": [
                    self._clean_document(document)
                    for document in documents
                ]
            }
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def _read_image_asset(self, args):
        if self._knowledge_scope()["mode"] == "none":
            return None, "[Error] 当前对话未启用知识库。"
        lookup = {
            key: str(args.get(key) or "").strip() or None
            for key in ("data_id", "ref_key")
        }
        if not any(lookup.values()):
            return None, "[Error] 需要 kb_search 返回的 data_id，或图1-1等图表编号。"
        # A figure-number-only lookup carries no document, so the pre-check runs
        # only when a data_id is given; the resolved asset is always re-checked
        # against the session scope below (authoritative).
        if lookup["data_id"] and not self._scope_allows_target(data_id=lookup["data_id"]):
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
            "description", "table_markdown", "related_text",
            "near_text", "uncertain", "analysis_error", "caption", "display_label",
        ):
            if key in info:
                result[key] = info.get(key)
        result["source_hint"] = KnowledgeBaseToolsMixin._source_hint(result)
        return result

    def do_kb_image_read(self, args, response):
        info, error = self._read_image_asset(args)
        if error:
            return self._anchor_outcome(args, error)
        attach_error = self._queue_image_view(info)
        if attach_error:
            return self._anchor_outcome(args, f"[Error] 原图无法加入模型输入: {attach_error}")
        self._record_knowledge_citations(info)
        result = self._public_image(info)
        result["attach_status"] = "attached"
        result["attach_message"] = "原图已加入下一轮模型输入。"
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def _queue_image_view(self, info):
        path = str(info.get("image_abspath") or "")
        if not os.path.isfile(path):
            return "[知识库图片] 原图文件不存在。"
        queue_image = getattr(self, "queue_image_for_next_turn", None)
        if not callable(queue_image):
            return "[知识库图片] 当前 Agent 不支持原生图片输入。"
        _metadata, error = queue_image(
            path,
            name=info.get("display_label") or info.get("title") or os.path.basename(path),
            context=(
                "[知识库图片原图]\n"
                f"图题: {info.get('display_label') or info.get('title') or ''}\n"
                f"来源: {info.get('source_file_name') or ''}\n"
                f"引用: {info.get('citation_label') or info.get('title') or '图片'}\n"
                "请直接查看原图，并结合工具结果回答用户问题。"
            ),
        )
        if error:
            return f"[{error.get('error_code')}] {error.get('error')}"
        return None
