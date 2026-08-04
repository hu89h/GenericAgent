"""GenericAgent tools backed by the unified knowledge-base facade.

This module deliberately contains only the agent-facing adapter.  Retrieval,
asset validation, and index lifecycle remain owned by :mod:`knowledge_base.backend`.
"""

import json
import os
import re

from agent_loop import StepOutcome

from .references import clean_public_text, public_reference


KB_RESPONSE_SOURCE_INSTRUCTIONS = """
[知识库来源规则]
- 如果最终回答实际使用了知识库信息，在答案末尾自行添加“**信息来源**”段。
- 来源只能使用工具结果中的 source_hint，只列出实际支撑回答的来源，不要罗列所有搜索命中。
- 同一原始文档必须合并为一条项目并对章节或图号去重，禁止重复书写文档名。格式固定为：
  **信息来源**
  - 《原始文档名》：“章节一”“章节二”
- data_id、ref、chunk_index、内部路径、处理后 Markdown 文件名、哈希前缀和检索诊断仅供工具调用，禁止出现在面向用户的回答中。
- 未使用知识库信息时不要添加“信息来源”段。
""".strip()


KB_AGENT_USAGE_INSTRUCTIONS = """
[KNOWLEDGE_BASE_USAGE]
- 先调用 kb_search，再根据搜索结果调用 kb_read 或 kb_image_read；不要在同一轮并行调用有依赖关系的工具。
- kb_search 返回的是候选摘要，不等同于完整证据。涉及精确事实、数字、比较、因果、条件或跨文档结论时，必须继续读取直接证据。
- 用户明确要求“根据某张表、图、图表或原文”时，该来源优先，不得自行改用附近摘要。文本或 HTML 表格调用 kb_read；图像表格、饼图、流程图等视觉证据调用 kb_image_read，并用 focus 写明需要核对的问题。
- kb_read 必须使用搜索结果中的 data_id（或兼容的 ref）和对应 chunk_index。结构未读完时必须按 continuation.next_chunk_index 继续读取。
- 用户未指定来源时，优先采用最接近问题的直接证据。不同来源冲突时按用户指定来源回答并说明冲突；证据不足时不得用相关但不准确的内容补答。
- kb_list 只用于文档发现、文档选择和分段导航，不是普通问答的默认检索工具，也不能把目录预览当作事实证据。
- 只有用户指定视觉来源、要求查看图片细节，或搜索描述不足以回答时，才调用 kb_image_read；不要批量打开所有命中图片。
- 必须使用当前对话提供的知识库范围，不要尝试读取其他知识库或任意本地路径。
- 当前模型没有 kb_image_read 时，只能使用已有图片文字描述，并明确没有查看原图，不能声称完成视觉核对。
- 不要把 data_id、ref、chunk_index、内部路径、处理后文件名或检索诊断写入最终回答；使用知识库信息时按来源规则在答案末尾列出原始文档来源。
""".strip()


KB_AGENT_SYSTEM_INSTRUCTIONS = f"{KB_AGENT_USAGE_INSTRUCTIONS}\n\n{KB_RESPONSE_SOURCE_INSTRUCTIONS}"


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
                            "enum": ["prose", "table", "image", "equation", "code", "list", "quote"],
                        },
                        "uniqueItems": True,
                        "description": "可选的证据结构限制。用户明确指定表格、图片、公式、代码等来源时使用；无结果时不得静默移除此限制。table 同时覆盖文本表格和图片表格，image 覆盖所有原图记录。",
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
                    "ref": {"type": "string"},
                    "chunk_index": {"type": "integer", "minimum": 0},
                    "span": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                    "max_chars": {
                        "type": "integer",
                        "minimum": 500,
                        "maximum": 8000,
                        "default": 4000,
                        "description": "单次正文上限。若 continuation.truncated_within_chunk=true，请对同一 chunk_index 提高该值后重读。",
                    },
                },
                "required": ["chunk_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_list",
            "description": "仅用于当前范围内的文档发现、文档选择和分段导航；提供 data_id 或 ref 时分页列出该文档的分段目录。不要在普通问答前默认调用，也不要把目录预览当作事实证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "preview_chars": {"type": "integer", "minimum": 20, "maximum": 200, "default": 80},
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
        if mode in {"multi", "selection", "selected"}:
            mode = "selection"
        if mode not in {"none", "all", "kb", "document", "selection"}:
            mode = "all"
        scope["mode"] = mode
        return scope

    def _scope_targets(self):
        scope = self._knowledge_scope()
        if scope["mode"] != "selection":
            return []
        targets = scope.get("targets") or scope.get("knowledge_bases") or []
        return [target for target in targets if isinstance(target, dict)]

    def _scope_kb_ids(self):
        scope = self._knowledge_scope()
        if scope["mode"] == "selection":
            out = []
            for target in self._scope_targets():
                kb_id = str(target.get("kb_id") or target.get("kbId") or target.get("id") or "").strip()
                if kb_id and kb_id not in out:
                    out.append(kb_id)
            return out
        kb_id = str(scope.get("kb_id") or "").strip()
        return [kb_id] if kb_id else []

    def _scope_kb_id(self):
        ids = self._scope_kb_ids()
        return ids[0] if len(ids) == 1 else None

    def _scope_read_kb_id(self, data_id=None, ref=None, file_name=None):
        target_kb, _target_doc = self._target_parts(
            data_id=data_id, ref=ref, file_name=file_name,
        )
        if target_kb:
            return target_kb
        return self._scope_kb_id()

    def _scope_allows_target(self, data_id=None, ref=None, file_name=None, kb_id=None):
        scope = self._knowledge_scope()
        if scope["mode"] == "none":
            return False
        if scope["mode"] == "all":
            return True

        target_kb, target_doc = self._target_parts(
            data_id=data_id, ref=ref, file_name=file_name, kb_id=kb_id
        )
        if scope["mode"] == "selection":
            for target in self._scope_targets():
                expected_kb = str(target.get("kb_id") or target.get("kbId") or target.get("id") or "").strip()
                if not expected_kb or expected_kb != target_kb:
                    continue
                if bool(target.get("all_documents", target.get("allDocuments", False))):
                    return True
                for document in target.get("documents") or target.get("docs") or []:
                    if not isinstance(document, dict):
                        continue
                    expected_doc = self._document_key(
                        data_id=document.get("data_id") or document.get("dataId"),
                        ref=document.get("ref"),
                        file_name=document.get("file_name") or document.get("fileName"),
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
        if scope["mode"] == "selection":
            return {"scope_targets": self._scope_targets()}
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
        return re.sub(r"(?m)^\s*章节路径：/[^\r\n]*?(?=\s*!\[|$)", "", text).strip()

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

    @classmethod
    def _clean_hit(cls, hit):
        kind = str(hit.get("kind") or "document")
        result = cls._public_reference(hit, kind=kind)
        content_type = str(hit.get("content_type") or ("image" if kind == "image" else "prose"))
        result["content_type"] = content_type
        structure_title = cls._clip_text(hit.get("structure_title"), 240)
        if structure_title:
            result["structure_title"] = structure_title
            if content_type != "prose":
                result["source_section"] = structure_title
        part_count = max(1, int(hit.get("structure_part_count") or 1))
        if part_count > 1:
            result["structure_part_index"] = int(hit.get("structure_part_index") or 0)
            result["structure_part_count"] = part_count
        for key in ("score", "score_type", "matched_by", "channel_ranks", "final_rank"):
            value = hit.get(key)
            if value not in (None, "", [], {}):
                result[key] = value
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
            clip = cls._clip_structured_rows if content_type == "table" else cls._clip_text
            body = clip(
                hit.get("body"), 1600,
                suffix="\n…[正文摘要已截断，完整内容请调用 kb_read]",
            )
            if snippet:
                result["snippet"] = snippet
            if body:
                result["body"] = body
        result["source_hint"] = cls._source_hint(result)
        return result

    @classmethod
    def _clean_diagnostics(cls, items):
        cleaned = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            error = cls._redact_internal(item.get("error"))
            source = cls._redact_internal(item.get("source"))
            row = {}
            if source:
                row["source"] = cls._clip_text(source, 120)
            if error:
                row["error"] = cls._clip_text(error, 400)
            if row:
                cleaned.append(row)
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
        return f"[Error] {tool} {messages.get(code, '操作失败。')}（{code}）"

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
        return cls._public_reference(document, kind="document", include_chunk=False)

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
        allowed_evidence = {"prose", "table", "image", "equation", "code", "list", "quote"}
        evidence_types = [
            str(value).strip().lower() for value in evidence_types if str(value).strip()
        ]
        if set(evidence_types) - allowed_evidence:
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "invalid_argument")
            )
        try:
            search_result = self._kb_backend().search(
                query,
                top_k=top_k,
                mode=mode,
                evidence_types=evidence_types,
                **self._scope_search_kwargs(),
            ) or {}
        except Exception:
            return self._anchor_outcome(
                args, self._safe_error("kb_search", "retrieval_unavailable")
            )
        hits = search_result.get("results") or []
        result = {
            "query": query,
            "mode": search_result.get("mode") or mode,
            "hits": [self._clean_hit(hit) for hit in hits],
        }
        if evidence_types:
            result["evidence_types"] = evidence_types
        diagnostics = self._clean_diagnostics(search_result.get("diagnostics"))
        if diagnostics:
            result["diagnostics"] = diagnostics
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def do_kb_read(self, args, response):
        if self._knowledge_scope()["mode"] == "none":
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "knowledge_disabled")
            )
        data_id = str(args.get("data_id") or "").strip() or None
        ref = str(args.get("ref") or "").strip() or None
        if not data_id and not ref:
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "invalid_argument")
            )
        if "chunk_index" not in args or args.get("chunk_index") is None:
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "invalid_argument")
            )
        if not self._scope_allows_target(data_id=data_id, ref=ref):
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
                ref=ref,
                chunk_index=start,
                span=span,
                kb_id=self._scope_read_kb_id(data_id=data_id, ref=ref),
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
                ref=ref,
                kb_id=self._scope_read_kb_id(data_id=data_id, ref=ref),
                chunk_index=start,
            ) or {}
        except Exception:
            reference = {}
        result = {
            "data_id": str(content.get("data_id") or data_id or ""),
            "content_type": str(content.get("content_type") or "prose"),
            "start_chunk_index": int(content.get("start_chunk_index") or 0),
            "end_chunk_index": int(content.get("end_chunk_index") or 0),
            "content": self._clean_content(content.get("content")),
            "continuation": dict(content.get("continuation") or {}),
        }
        structure_title = self._clip_text(content.get("structure_title"), 240)
        if structure_title:
            result["structure_title"] = structure_title
            if isinstance(reference, dict):
                reference = {**reference, "source_section": structure_title}
        source_hint = self._source_hint(reference)
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
        ref = str(args.get("ref") or "").strip() or None
        if data_id or ref:
            if not self._scope_allows_target(data_id=data_id, ref=ref):
                return self._anchor_outcome(
                    args, self._safe_error("kb_list", "scope_denied")
                )
            try:
                preview = max(20, min(int(args.get("preview_chars", 60)), 120))
                offset = max(0, int(args.get("offset", 0)))
                page_limit = max(1, min(int(args.get("limit", 20)), 50))
            except (TypeError, ValueError):
                preview, offset, page_limit = 60, 0, 20
            try:
                raw_result = self._kb_backend().list_chunks(
                    data_id=data_id,
                    ref=ref,
                    kb_id=self._scope_read_kb_id(data_id=data_id, ref=ref),
                    preview_chars=preview,
                    limit=400,
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
                    ref=ref,
                    kb_id=self._scope_read_kb_id(data_id=data_id, ref=ref),
                    chunk_index=0,
                ) or {}
            except Exception:
                source = {}
            all_chunks = raw_result.get("chunks") or []
            total = int(raw_result.get("n_chunks") or len(all_chunks))
            chunks = []
            for chunk in all_chunks[offset:offset + page_limit]:
                preview_text = self._clip_text(chunk.get("preview"), preview)
                item = {
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "chars": int(chunk.get("chars") or 0),
                }
                if preview_text:
                    item["preview"] = preview_text
                content_type = str(chunk.get("content_type") or "prose").strip()
                if content_type:
                    item["content_type"] = content_type
                source_section = self._clip_text(chunk.get("source_section"), 200)
                if source_section:
                    item["source_section"] = source_section
                structure_title = self._clip_text(chunk.get("structure_title"), 200)
                if structure_title:
                    item["structure_title"] = structure_title
                chunks.append(item)
            result = {
                "data_id": data_id,
                "total": total,
                "offset": offset,
                "limit": page_limit,
                "has_more": offset + len(chunks) < total,
                "chunks": chunks,
            }
            source_name = str(source.get("source_file_name") or "").strip()
            source_hint = self._source_hint(source)
            if source_name:
                result["source_file_name"] = source_name
            if source_hint:
                result["source_hint"] = source_hint
            if result["has_more"]:
                result["next_offset"] = offset + len(chunks)
        else:
            scope = self._knowledge_scope()
            try:
                if scope["mode"] == "selection":
                    documents = []
                    seen = set()
                    for kb_id in self._scope_kb_ids():
                        for document in self._kb_backend().list_documents(kb_id=kb_id):
                            key = str(document.get("data_id") or document.get("ref") or document.get("file_name") or "")
                            if key in seen or not self._scope_allows_target(
                                data_id=document.get("data_id"),
                                ref=document.get("ref"),
                                file_name=document.get("file_name"),
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
                    ref=scope.get("ref"),
                    file_name=scope.get("file_name"),
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
                if not source_data_id:
                    target_kb, target_doc = self._target_parts(
                        ref=scope.get("ref"), kb_id=scope.get("kb_id")
                    )
                    if target_kb and target_doc:
                        source_data_id = f"{target_kb}::{target_doc}"
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
                            ref=item.get("ref"),
                            file_name=item.get("file_name"),
                            kb_id=item.get("kb_id") or kb_id,
                        ):
                            continue
                        if item is not candidate and item.get("data_id"):
                            item = self._kb_backend().read_image(
                                kb_id=kb_id, data_id=item.get("data_id")
                            )
                        if isinstance(item, dict) and not item.get("error"):
                            matches.append(item)
                if len(matches) == 1:
                    info = matches[0]
                elif len(matches) > 1:
                    return None, self._safe_error("kb_image_read", "invalid_argument")
                else:
                    info = {"error": "[未找到图片资产]"}
            else:
                kwargs = {"kb_id": read_kb_id, **lookup}
                if source_data_id:
                    kwargs["source_data_id"] = source_data_id
                info = self._kb_backend().read_image(**kwargs)
        except Exception:
            return None, self._safe_error("kb_image_read", "image_read_failed")
        if not isinstance(info, dict) or info.get("error"):
            return None, self._safe_error("kb_image_read", "image_unavailable")
        if not self._scope_allows_target(
            data_id=info.get("data_id"),
            ref=info.get("ref"),
            file_name=info.get("file_name"),
            kb_id=info.get("kb_id"),
        ):
            return None, self._safe_error("kb_image_read", "scope_denied")
        return info, None

    @classmethod
    def _public_image(cls, info):
        result = cls._public_reference(info, kind="image")
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

        near_text = cls._clip_text(info.get("near_text"), 1800)
        related_text = cls._clip_text(info.get("related_text"), 1200)
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
        if near_text:
            result["near_text"] = near_text
        if related_text:
            result["related_text"] = related_text
        analysis_error = cls._clip_text(info.get("analysis_error"), 400)
        if analysis_error:
            result["analysis_error"] = analysis_error
        result["source_hint"] = cls._source_hint(result)
        return result

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
        result["attach_status"] = "attached"
        result["attach_message"] = "原图已加入下一轮模型输入。"
        return self._anchor_outcome(args, json.dumps(result, ensure_ascii=False, indent=2))

    def _queue_image_view(self, info, *, focus=""):
        path = str(info.get("image_abspath") or "")
        if not os.path.isfile(path):
            return "[知识库图片] 原图文件不存在。"
        queue_image = getattr(self, "queue_image_for_next_turn", None)
        if not callable(queue_image):
            return "[知识库图片] 当前 Agent 不支持原生图片输入。"
        context = (
            "[知识库图片原图]\n"
            f"图题: {info.get('display_label') or info.get('title') or ''}\n"
            f"来源: {info.get('source_file_name') or ''}\n"
            f"引用: {info.get('citation_label') or info.get('title') or '图片'}\n"
        )
        if focus:
            context += f"本次查看重点: {focus}\n"
        context += "请直接查看原图，并结合工具结果回答用户问题。"
        _metadata, error = queue_image(
            path,
            name=info.get("display_label") or info.get("title") or os.path.basename(path),
            context=context,
        )
        if error:
            return f"[{error.get('error_code')}] {error.get('error')}"
        return None
