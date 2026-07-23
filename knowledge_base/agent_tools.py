"""GenericAgent tools backed by the unified knowledge-base facade.

This module deliberately contains only the agent-facing adapter.  Retrieval,
asset validation, and index lifecycle remain owned by :mod:`knowledge_base.backend`.
"""

import json

from agent_loop import StepOutcome


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
            "description": "读取知识库图片的图题、VLM 描述、表格 Markdown、关联正文和引用信息，不发送原图。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "image_path": {"type": "string"},
                    "data_id": {"type": "string"},
                    "ref": {"type": "string"},
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
            "kb_id", "score", "score_type", "rank", "data_id", "chunk_index",
            "title", "file_name", "folder", "ref", "kind", "format", "image_id",
            "image_path", "parent_data_id", "parent_chunk_index", "header_path",
            "chunk_role", "snippet", "body", "description", "table_markdown",
            "related_text", "related_text_refs", "near_text", "uncertain",
        )
        return {key: hit[key] for key in keep if key in hit and hit[key] is not None}

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
            result = {"documents": self._kb_backend().list_documents(kb_id=self._scope_kb_id())}
        yield "[Info] kb_list done.\n"
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def _read_image_asset(self, args):
        lookup = {
            key: str(args.get(key) or "").strip() or None
            for key in ("image_id", "image_path", "data_id", "ref")
        }
        if not any(lookup.values()):
            return None, "[Error] 需要 image_id、image_path、data_id 或 ref。"
        # image_id/image_path alone do not carry the owning document.  In a
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
        result = dict(info)
        result.pop("image_abspath", None)
        return result

    def do_kb_image_read(self, args, response):
        info, error = self._read_image_asset(args)
        if error:
            return self._anchor_outcome(args, error)
        yield "[Info] kb_image_read done.\n"
        return self._anchor_outcome(args, json.dumps(self._public_image(info), ensure_ascii=False, indent=2))
