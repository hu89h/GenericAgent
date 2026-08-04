"""GenericAgent tools backed by the unified knowledge-base facade.

This module deliberately contains only the agent-facing adapter.  Retrieval,
asset validation, and index lifecycle remain owned by :mod:`knowledge_base.backend`.
"""

import json
import os
import re

from agent_loop import StepOutcome

from .references import clean_public_text, public_reference
from .scope import normalize_scope


KB_AGENT_SYSTEM_INSTRUCTIONS = """
[KNOWLEDGE_BASE_USAGE]
- 先搜索，再按用户指定的证据类型读取正文或原图；有依赖的工具不得在同一轮并行调用。
- 搜索命中是候选摘要。精确事实、数字、比较、因果、条件、表格和跨文档结论必须读取直接证据。
- 用户指定表、图、图表或原文时优先使用该来源；视觉证据用 kb_image_read，并在 focus 中写明需要从原图核对的问题。
- 每次 kb_read 后检查 continuation.has_more；未读完时按 next_chunk_index 继续。
- 证据冲突时按用户指定来源回答并说明冲突；证据不足时明确说明，不得用相近内容补答。
- 最终回答不得出现 data_id、chunk_index、内部路径、处理后文件名或检索诊断。实际使用知识库时，在末尾按 source_hint 生成“信息来源”，同一原始文档合并并去重章节或图号。
""".strip()


KB_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "在当前知识库范围内召回文本和图片候选。结果是摘要和稳定引用，不是完整证据；涉及精确事实、数字、比较、因果、条件、表格或跨文档结论时，必须先等待本工具返回，再用命中的 data_id 和 chunk_index 调用 kb_read。不要在一次检索失败后静默切换 mode。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言问题或检索词"},
                    "mode": {
                        "type": "string",
                        "enum": ["rrf", "vector", "sparse"],
                        "description": "必须显式选择且不能静默降级：rrf 融合向量和稀疏召回，适合综合问题；vector 适合语义改写和概念相近问题；sparse 适合专有名词、编号、型号和精确术语。明确图表编号（如 图3-1、表4.1）时，图表引用仍作为独立确定性信号参与排序。",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "evidence_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["text", "table", "image"],
                        },
                        "uniqueItems": True,
                        "description": "可选证据限制：text 为正文等文本结构，table 为文本或图片表格，image 为原图。无结果时不得静默移除此限制。",
                    },
                },
                "required": ["query", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_read",
            "description": "读取 kb_search 命中的文本证据，是文本事实的主要依据。表格、列表、代码、公式等会优先返回同一完整结构；普通正文由 span 控制相邻分段。必须检查 continuation，未读完时按 next_chunk_index 继续读取。不要与依赖搜索结果的调用并行执行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "chunk_index": {"type": "integer", "minimum": 0},
                    "span": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                    "max_chars": {
                        "type": "integer",
                        "minimum": 500,
                        "maximum": 8000,
                        "default": 4000,
                        "description": "单次正文上限。若 continuation 返回 required_max_chars，请对同一 chunk_index 提高该值后重读。",
                    },
                },
                "required": ["data_id", "chunk_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_list",
            "description": "仅用于当前范围内的文档发现、文档选择和分段导航；提供 data_id 时分页列出该文档的分段目录。不要在普通问答前默认调用，也不要把目录预览当作事实证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_image_read",
            "description": "读取 kb_search 定位图片的原图、图题、VLM 描述、表格 Markdown、必要上下文和引用信息。必须先等待搜索结果；只有用户要求查看图片细节、需要核对图表结构，或描述不足以回答时才调用。调用成功后原图会自动加入下一轮多模态输入，不要批量打开所有命中图片。不能读取任意本地路径，只能使用搜索结果中的 data_id，或当前文档范围内的精确图表编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "ref_key": {"type": "string", "description": "图1-1、表8-1等知识库返回的图表编号"},
                    "focus": {
                        "type": "string",
                        "description": "可选：补充本次需要从原图确认的视觉关注点；通常无需填写，不要重复用户问题，也不要填写内部路径或 ID。",
                        "maxLength": 500,
                    },
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
    def _document_key(data_id=None):
        raw = str(data_id or "").strip()
        if "::" in raw:
            return raw.split("::", 1)[1].split("::image::", 1)[0].replace("\\", "/").lstrip("/")
        return ""

    @classmethod
    def _target_parts(cls, data_id=None, kb_id=None):
        raw_id = str(data_id or "").strip()
        if "::" in raw_id:
            return raw_id.split("::", 1)[0], cls._document_key(raw_id)
        return str(kb_id or "").strip(), ""

    def _knowledge_scope(self):
        return normalize_scope(getattr(self.parent, "knowledge_scope", None))

    def _scope_targets(self):
        scope = self._knowledge_scope()
        if scope["mode"] != "selection":
            return []
        targets = scope.get("targets") or []
        return [target for target in targets if isinstance(target, dict)]

    def _scope_kb_ids(self):
        scope = self._knowledge_scope()
        if scope["mode"] == "selection":
            out = []
            for target in self._scope_targets():
                kb_id = str(target.get("kb_id") or "").strip()
                if kb_id and kb_id not in out:
                    out.append(kb_id)
            return out
        kb_id = str(scope.get("kb_id") or "").strip()
        return [kb_id] if kb_id else []

    def _scope_kb_id(self):
        ids = self._scope_kb_ids()
        return ids[0] if len(ids) == 1 else None

    def _scope_read_kb_id(self, data_id=None):
        target_kb, _target_doc = self._target_parts(
            data_id=data_id,
        )
        if target_kb:
            return target_kb
        return self._scope_kb_id()

    def _scope_allows_target(self, data_id=None, kb_id=None):
        scope = self._knowledge_scope()
        if scope["mode"] == "none":
            return False
        if scope["mode"] == "all":
            return True

        target_kb, target_doc = self._target_parts(
            data_id=data_id, kb_id=kb_id
        )
        if scope["mode"] == "selection":
            for target in self._scope_targets():
                expected_kb = str(target.get("kb_id") or "").strip()
                if not expected_kb or expected_kb != target_kb:
                    continue
                if target.get("all_documents") is True:
                    return True
                for document in target.get("documents") or []:
                    if not isinstance(document, dict):
                        continue
                    expected_doc = self._document_key(
                        data_id=document.get("data_id"),
                    )
                    if expected_doc and target_doc and expected_doc == target_doc:
                        return True
            return False
        expected_kb = str(scope.get("kb_id") or "").strip()
        if not expected_kb or not target_kb or target_kb != expected_kb:
            return False
        if scope["mode"] == "kb":
            return True

        expected_doc = self._document_key(
            data_id=scope.get("data_id"),
        )
        return bool(expected_doc and target_doc and expected_doc == target_doc)

    def _scope_search_kwargs(self):
        scope = self._knowledge_scope()
        if scope["mode"] == "none":
            return {}
        if scope["mode"] == "all":
            return {}
        if scope["mode"] == "selection":
            return {"scope_targets": self._scope_targets()}
        out = {"kb_id": self._scope_kb_id()}
        if scope["mode"] == "document":
            document = self._document_key(
                data_id=scope.get("data_id"),
            )
            if document:
                out["file_name"] = document
        return out

    @staticmethod
    def _clean_content(value):
        text = clean_public_text(value)
        # MinerU previews may already be truncated in the middle of an image
        # link.  Match through the end of the line as a fallback so a partial
        # processed path cannot reach the model.
        image_pattern = re.compile(r"!\[([^\]]*)\]\([^\r\n)]*(?:\)|$)")

        def replace_image(match):
            alt = re.sub(r"\s+", " ", match.group(1) or "").strip()
            return f"[图片：{alt}]" if alt else "[图片]"

        text = image_pattern.sub(replace_image, text)
        return re.sub(r"(?m)^\s*章节路径：(?:/[^/\r\n]+)+/\s*", "", text).strip()

    @staticmethod
    def _clip_text(value, limit: int, *, suffix=""):
        text = KnowledgeBaseToolsMixin._clean_content(value)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + (suffix or "\n…[检索摘要已截断]")

    @classmethod
    def _clip_structured_rows(cls, value, limit: int, *, suffix=""):
        text = cls._clean_content(value)
        if len(text) <= limit:
            return text
        cut = text.rfind("\n", 0, limit + 1)
        if cut < limit // 2:
            cut = limit
        return text[:cut].rstrip() + (suffix or "\n…[结构化内容已截断]")

    @classmethod
    def _public_reference(cls, item, *, kind=None, include_chunk=True):
        normalized = public_reference(item, kind=kind)
        resolved_kind = normalized.get("kind") or kind or "document"
        result = {"kind": resolved_kind}
        data_id = str(normalized.get("data_id") or "").strip()
        if data_id:
            result["data_id"] = data_id
        if include_chunk and normalized.get("chunk_index") is not None:
            result["chunk_index"] = normalized.get("chunk_index")
        if resolved_kind == "image":
            image_id = str(normalized.get("image_id") or "").strip()
            ref_key = str(normalized.get("ref_key") or "").strip()
            if image_id:
                result["image_id"] = image_id
            if ref_key:
                result["ref_key"] = ref_key
            label = str(
                item.get("display_label")
                or item.get("caption")
                or item.get("title")
                or ref_key
                or ""
            ).strip()
            if label:
                result["display_label"] = cls._clip_text(label, 320)
        source_name = str(normalized.get("source_file_name") or "").strip()
        if source_name:
            result["source_file_name"] = source_name
        section = str(normalized.get("source_section") or "").strip()
        if section:
            result["source_section"] = section
        result["source_hint"] = cls._source_hint(result)
        return result

    @staticmethod
    def _evidence_type(item):
        if str(item.get("kind") or "") == "image":
            return "image"
        return "table" if str(item.get("content_type") or "") == "table" else "text"

    @classmethod
    def _clean_hit(cls, hit):
        kind = str(hit.get("kind") or "document")
        reference = cls._public_reference(hit, kind=kind)
        result = {
            "data_id": reference.get("data_id", ""),
            "evidence_type": cls._evidence_type(hit),
            "source_hint": cls._source_hint({
                **hit,
                "kind": kind,
                "source_file_name": reference.get("source_file_name"),
                "source_section": reference.get("source_section"),
            }),
            "matched_by": list(hit.get("matched_by") or []),
        }
        if kind != "image":
            result["chunk_index"] = int(reference.get("chunk_index") or 0)
        else:
            ref_key = str(reference.get("ref_key") or "").strip()
            if ref_key:
                result["ref_key"] = ref_key
        if kind == "image":
            description = cls._clip_text(hit.get("description"), 1400)
            if description:
                result["description"] = description
            table = cls._clip_structured_rows(
                hit.get("table_markdown"), 2800,
                suffix="\n…[表格摘要已截断，完整内容请调用 kb_image_read]",
            )
            if table:
                result["table_markdown"] = table
            uncertain = hit.get("uncertain")
            if isinstance(uncertain, list):
                values = [cls._clip_text(item, 160) for item in uncertain[:6]]
                values = [item for item in values if item]
                if values:
                    result["uncertain"] = values
            elif uncertain:
                result["uncertain"] = [cls._clip_text(uncertain, 160)]
            if not description:
                fallback = cls._clip_text(hit.get("snippet") or hit.get("caption"), 320)
                if fallback:
                    result["description"] = fallback
        else:
            snippet = cls._clip_text(hit.get("snippet"), 320)
            clip = cls._clip_structured_rows if result["evidence_type"] == "table" else cls._clip_text
            body = clip(
                hit.get("body"), 1600,
                suffix="\n…[正文摘要已截断，完整内容请调用 kb_read]",
            )
            if snippet:
                result["snippet"] = snippet
            if body:
                result["body"] = body
        return {key: value for key, value in result.items() if value not in (None, "", [], {})}

    @classmethod
    def _clean_warnings(cls, items):
        cleaned = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            error = cls._redact_internal(item.get("error"))
            if error:
                cleaned.append(cls._clip_text(error, 320))
        return cleaned

    @staticmethod
    def _redact_internal(value):
        text = clean_public_text(value)
        text = re.sub(r"(?i)(?:[A-Za-z]:[\\/]|/)[^\s,;]+", "[内部路径]", text)
        return re.sub(r"(?i)\bkb-[a-z0-9_-]+\b", "[内部知识库]", text)

    @classmethod
    def _clean_image_focus(cls, value):
        text = cls._redact_internal(value)
        return text[:500].rstrip()

    @staticmethod
    def _safe_error(tool, code):
        messages = {
            "knowledge_disabled": "当前对话未启用知识库。",
            "invalid_argument": "参数无效。",
            "scope_denied": "目标不在当前对话的知识库范围内。",
            "retrieval_unavailable": "知识库暂时无法检索，请检查索引状态。",
            "index_unavailable": "知识库索引暂不可用。",
            "not_found": "未找到对应的知识库内容。",
            "read_failed": "读取知识库内容失败。",
            "image_unavailable": "未找到对应的图片资产。",
            "image_read_failed": "读取图片资产失败。",
            "attach_failed": "原图无法加入模型输入。",
        }
        return json.dumps({
            "error_code": code,
            "message": messages.get(code, "操作失败。"),
        }, ensure_ascii=False)

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
        structure_title = str(reference.get("structure_title") or "").strip()
        hint = f"《{name}》：“{section}”" if section else f"《{name}》"
        if (
            reference.get("kind") != "image"
            and structure_title
            and structure_title != section
        ):
            hint += (
                f"——“{structure_title}”"
                if section else f"：“{structure_title}”"
            )
        return hint

    @classmethod
    def _clean_document(cls, document):
        reference = cls._public_reference(document, kind="document", include_chunk=False)
        return {
            "data_id": reference.get("data_id", ""),
            "source_hint": reference.get("source_hint", ""),
        }

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
            next_prompt=next_prompt,
        )

    def do_kb_search(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "knowledge_disabled")
            )
        query = str(args.get("query") or "").strip()
        if not query:
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "invalid_argument")
            )
        mode = str(args.get("mode") or "").strip().lower()
        if mode not in {"rrf", "vector", "sparse"}:
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "invalid_argument")
            )
        try:
            top_k = max(1, min(int(args.get("top_k", 5)), 10))
        except (TypeError, ValueError):
            top_k = 5
        evidence_types = args.get("evidence_types") or []
        if not isinstance(evidence_types, list):
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "invalid_argument")
            )
        allowed_evidence = {"text", "table", "image"}
        evidence_types = [
            str(value).strip().lower() for value in evidence_types if str(value).strip()
        ]
        if set(evidence_types) - allowed_evidence:
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "invalid_argument")
            )
        internal_evidence = []
        for value in evidence_types:
            if value == "text":
                internal_evidence.extend(
                    ["prose", "equation", "code", "list", "quote", "note", "figure"]
                )
            else:
                internal_evidence.append(value)
        try:
            search_result = self._kb_backend().search(
                query,
                top_k=top_k,
                mode=mode,
                evidence_types=internal_evidence,
                **self._scope_search_kwargs(),
            ) or {}
        except Exception:
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "retrieval_unavailable")
            )
        hits = search_result.get("results") or []
        result = {
            "mode": search_result.get("mode") or mode,
            "hits": [self._clean_hit(hit) for hit in hits],
        }
        warnings = self._clean_warnings(search_result.get("diagnostics"))
        if warnings:
            result["warnings"] = warnings
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def do_kb_read(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "knowledge_disabled")
            )
        data_id = str(args.get("data_id") or "").strip() or None
        if not data_id:
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "invalid_argument")
            )
        if "chunk_index" not in args or args.get("chunk_index") is None:
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "invalid_argument")
            )
        if not self._scope_allows_target(data_id=data_id):
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "scope_denied")
            )
        try:
            start = max(0, int(args.get("chunk_index", 0)))
            span = max(1, min(int(args.get("span", 1)), 5))
            max_chars = max(500, min(int(args.get("max_chars", 4000)), 8000))
        except (TypeError, ValueError):
            start, span, max_chars = 0, 1, 4000
        backend = self._kb_backend()
        try:
            content = backend.read_content(
                data_id=data_id,
                chunk_index=start,
                span=span,
                kb_id=self._scope_read_kb_id(data_id=data_id),
                max_chars=max_chars,
            )
        except Exception:
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "read_failed")
            )
        if not isinstance(content, dict) or content.get("error_code"):
            code = (
                "not_found"
                if isinstance(content, dict) and content.get("error_code") == "not_found"
                else "read_failed"
            )
            return self._anchor_outcome(args, self._safe_error("kb_read", code))
        try:
            reference = backend.reference_for_chunk(
                data_id=data_id,
                kb_id=self._scope_read_kb_id(data_id=data_id),
                chunk_index=start,
            ) or {}
        except Exception:
            reference = {}
        result = {
            "data_id": str(content.get("data_id") or data_id or ""),
            "evidence_type": self._evidence_type(content),
            "content": self._clean_content(content.get("content")),
            "continuation": {
                key: value for key, value in dict(content.get("continuation") or {}).items()
                if key in {"has_more", "next_chunk_index", "required_max_chars"}
                and value is not None
            },
        }
        source_hint = self._source_hint({
            **(reference if isinstance(reference, dict) else {}),
            "structure_title": content.get("structure_title"),
        })
        if source_hint:
            result["source_hint"] = source_hint
        return self._anchor_outcome(
            args, json.dumps(result, ensure_ascii=False, indent=2)
        )

    def do_kb_list(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(
                args, self._safe_error("kb_list", "knowledge_disabled")
            )
        data_id = str(args.get("data_id") or "").strip() or None
        if data_id:
            if not self._scope_allows_target(data_id=data_id):
                return self._anchor_outcome(
                    args, self._safe_error("kb_list", "scope_denied")
                )
            try:
                offset = max(0, int(args.get("offset", 0)))
                page_limit = max(1, min(int(args.get("limit", 20)), 50))
            except (TypeError, ValueError):
                offset, page_limit = 0, 20
            try:
                raw_result = self._kb_backend().list_chunks(
                    data_id=data_id,
                    kb_id=self._scope_read_kb_id(data_id=data_id),
                    offset=offset,
                    limit=page_limit,
                )
            except Exception:
                return self._anchor_outcome(
                    args, self._safe_error("kb_list", "read_failed")
                )
            if not isinstance(raw_result, dict) or raw_result.get("error"):
                return self._anchor_outcome(
                    args, self._safe_error("kb_list", "not_found")
                )
            try:
                source = self._kb_backend().reference_for_chunk(
                    data_id=data_id,
                    kb_id=self._scope_read_kb_id(data_id=data_id),
                    chunk_index=0,
                ) or {}
            except Exception:
                source = {}
            chunks = []
            for chunk in raw_result.get("chunks") or []:
                preview_text = self._clip_text(chunk.get("preview"), 80)
                item = {
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "evidence_type": self._evidence_type(chunk),
                }
                if preview_text:
                    item["preview"] = preview_text
                chunks.append(item)
            result = {
                "data_id": str(raw_result.get("data_id") or data_id or ""),
                "has_more": bool(raw_result.get("has_more")),
                "chunks": chunks,
            }
            source_hint = self._source_hint(source)
            if source_hint:
                result["source_hint"] = source_hint
            if result["has_more"]:
                result["next_offset"] = int(
                    raw_result.get("next_offset") or offset + len(chunks)
                )
        else:
            scope = self._knowledge_scope()
            try:
                if scope["mode"] == "selection":
                    documents = []
                    seen = set()
                    for kb_id in self._scope_kb_ids():
                        for document in self._kb_backend().list_documents(kb_id=kb_id):
                            key = str(document.get("data_id") or "")
                            if key in seen or not self._scope_allows_target(
                                data_id=document.get("data_id"),
                                kb_id=kb_id,
                            ):
                                continue
                            seen.add(key)
                            documents.append(document)
                else:
                    documents = self._kb_backend().list_documents(kb_id=self._scope_kb_id())
            except Exception:
                return self._anchor_outcome(
                    args, self._safe_error("kb_list", "read_failed")
                )
            if scope["mode"] == "document":
                expected = self._document_key(
                    data_id=scope.get("data_id"),
                )
                documents = [
                    document for document in documents
                    if self._document_key(
                        data_id=document.get("data_id"),
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
            return None, self._safe_error("kb_image_read", "knowledge_disabled")
        lookup = {
            key: str(args.get(key) or "").strip() or None
            for key in ("data_id", "ref_key")
        }
        if not any(lookup.values()):
            return None, self._safe_error("kb_image_read", "invalid_argument")
        # A figure-number-only lookup carries no document, so the pre-check runs
        # only when a data_id is given; the resolved asset is always re-checked
        # against the session scope below (authoritative).
        if lookup["data_id"] and not self._scope_allows_target(data_id=lookup["data_id"]):
            return None, self._safe_error("kb_image_read", "scope_denied")
        try:
            scope = self._knowledge_scope()
            read_kb_id = self._scope_read_kb_id(data_id=lookup.get("data_id"))
            source_data_id = None
            if scope["mode"] == "document":
                source_data_id = str(scope.get("data_id") or "").strip() or None
            if scope["mode"] == "selection" and not lookup.get("data_id"):
                matches = []
                for kb_id in self._scope_kb_ids():
                    candidate = self._kb_backend().read_image(kb_id=kb_id, **lookup)
                    candidates = []
                    if isinstance(candidate, dict) and candidate.get("error_code") == "image_ambiguous":
                        candidates = candidate.get("candidates") or []
                    elif isinstance(candidate, dict) and not candidate.get("error"):
                        candidates = [candidate]
                    for item in candidates:
                        if not self._scope_allows_target(
                            data_id=item.get("data_id"),
                            kb_id=item.get("kb_id") or kb_id,
                        ):
                            continue
                        if isinstance(item, dict) and item.get("data_id"):
                            matches.append((kb_id, item))
                if len(matches) == 1:
                    matched_kb_id, info = matches[0]
                    if not info.get("image_abspath"):
                        info = self._kb_backend().read_image(
                            kb_id=matched_kb_id, data_id=info.get("data_id")
                        )
                elif len(matches) > 1:
                    return None, self._image_ambiguity_result(
                        [item for _kb_id, item in matches]
                    )
                else:
                    info = {"error": "[未找到图片资产]"}
            else:
                kwargs = {"kb_id": read_kb_id, **lookup}
                if source_data_id:
                    kwargs["source_data_id"] = source_data_id
                info = self._kb_backend().read_image(**kwargs)
        except Exception:
            return None, self._safe_error("kb_image_read", "image_read_failed")
        if isinstance(info, dict) and info.get("error_code") == "image_ambiguous":
            candidates = [
                item for item in (info.get("candidates") or [])
                if isinstance(item, dict)
                and self._scope_allows_target(data_id=item.get("data_id"))
            ]
            if len(candidates) == 1:
                candidate = candidates[0]
                try:
                    info = self._kb_backend().read_image(
                        kb_id=self._scope_read_kb_id(data_id=candidate.get("data_id")),
                        data_id=candidate.get("data_id"),
                    )
                except Exception:
                    return None, self._safe_error("kb_image_read", "image_read_failed")
            else:
                return None, self._image_ambiguity_result(candidates)
        if not isinstance(info, dict) or info.get("error"):
            return None, self._safe_error("kb_image_read", "image_unavailable")
        if not self._scope_allows_target(
            data_id=info.get("data_id"),
            kb_id=info.get("kb_id"),
        ):
            return None, self._safe_error("kb_image_read", "scope_denied")
        return info, None

    @classmethod
    def _image_ambiguity_result(cls, candidates):
        public_candidates = []
        seen = set()
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            data_id = str(candidate.get("data_id") or "").strip()
            if not data_id or data_id in seen:
                continue
            seen.add(data_id)
            item = {"data_id": data_id}
            ref_key = cls._clip_text(candidate.get("ref_key"), 80)
            label = cls._clip_text(
                candidate.get("display_label") or ref_key or "图片", 200
            )
            source_hint = cls._clip_text(
                candidate.get("source_hint") or cls._source_hint({
                    **candidate,
                    "kind": "image",
                    "display_label": label,
                }),
                320,
            )
            item["ref_key"] = ref_key
            item["image_label"] = label
            item["source_hint"] = source_hint
            public_candidates.append(item)
            if len(public_candidates) >= 10:
                break
        return json.dumps({
            "error_code": "image_ambiguous",
            "message": "图片编号对应多个候选，请选择候选的 data_id 后重试。",
            "candidates": public_candidates,
        }, ensure_ascii=False, indent=2)

    @classmethod
    def _public_image(cls, info):
        reference = cls._public_reference(info, kind="image")
        result = {
            "data_id": reference.get("data_id", ""),
            "evidence_type": "image",
            "source_hint": reference.get("source_hint", ""),
        }
        ref_key = str(reference.get("ref_key") or "").strip()
        if ref_key:
            result["ref_key"] = ref_key
        description = cls._clip_text(info.get("description"), 2400)
        if description:
            result["description"] = description
        table = cls._clip_text(
            info.get("table_markdown"), 3000,
            suffix="\n…[表格内容已截断]",
        )
        if table:
            result["table_markdown"] = table
        uncertain = info.get("uncertain")
        if isinstance(uncertain, list):
            values = [cls._clip_text(item, 200) for item in uncertain[:8]]
            values = [item for item in values if item]
            if values:
                result["uncertain"] = values
        elif uncertain:
            result["uncertain"] = [cls._clip_text(uncertain, 200)]

        near_text = cls._clip_text(info.get("near_text"), 1400)
        related_text = cls._clip_text(info.get("related_text"), 1000)
        if related_text and (
            related_text == near_text
            or related_text in near_text
            or near_text in related_text
        ):
            related_text = ""
        # A long sequence of figure/table labels is a document catalogue, not
        # useful context for the selected image.
        if related_text and len(re.findall(r"(?:图|表)\s*\d", related_text)) >= 8:
            related_text = ""
        context = "\n\n".join(value for value in (near_text, related_text) if value)
        if context:
            result["context"] = context
        return {key: value for key, value in result.items() if value not in (None, "", [], {})}

    def do_kb_image_read(self, args, response):
        info, error = self._read_image_asset(args)
        if error:
            return self._anchor_outcome(args, error)
        focus = self._clean_image_focus(args.get("focus"))
        try:
            attach_error = self._queue_image_view(info, focus=focus)
        except Exception:
            attach_error = "attach_failed"
        if attach_error:
            return self._anchor_outcome(
                args, self._safe_error("kb_image_read", "attach_failed")
            )
        self._record_knowledge_citations(info)
        result = self._public_image(info)
        result["image_attached"] = True
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def _queue_image_view(self, info, *, focus=""):
        path = str(info.get("image_abspath") or "")
        if not os.path.isfile(path):
            return "[知识库图片] 原图文件不存在。"
        queue_image = getattr(self, "queue_image_for_next_turn", None)
        if not callable(queue_image):
            return "[知识库图片] 当前 Agent 不支持原生图片输入。"
        context = "[知识库图片原图]\n"
        if focus:
            context += f"本次查看重点: {focus}\n"
        context += "请直接查看原图并结合上一条工具结果回答。"
        _metadata, error = queue_image(
            path,
            name=info.get("display_label") or info.get("title") or os.path.basename(path),
            context=context,
        )
        if error:
            return f"[{error.get('error_code')}] {error.get('error')}"
        return None
