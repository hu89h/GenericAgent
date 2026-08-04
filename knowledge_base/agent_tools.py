"""GenericAgent tools backed by the unified knowledge-base facade.

This module deliberately contains only the agent-facing adapter.  Retrieval,
asset validation, and index lifecycle remain owned by :mod:`knowledge_base.backend`.
"""

import json
import os
import re
import unicodedata

from agent_loop import StepOutcome

from .references import clean_public_text, public_reference
from .scope import normalize_scope


KB_AGENT_SYSTEM_INSTRUCTIONS = """
[KNOWLEDGE_BASE_USAGE]
- 一般问题先搜索，再按用户指定的证据来源读取必要的正文或原图；用户已经给出明确图表编号并要求查看原图时，可以直接用 ref_key 调用 kb_image_read。有依赖的工具不得在同一轮并行调用。搜索结果是处理后候选，并非无损原文；其中完整、明确、未截断且直接包含答案的内容可以作为证据使用。
- evidence_types 限制的是知识库记录的表现形式，不等同于用户所说的表格、原文或报告。指定来源可能被表现为文本、表格或图片；形式不明确时先检索全部类型并在查询中保留来源要求，不能用附近内容冒充指定来源。
- 可以为检索目的自由改写、简化、扩展或拆分用户问题，但必须保持原始信息需求、范围和证据要求，不得把未经证实的前提带入结论。后续检索应有新的调查目的；若不再产生新证据，应停止穷举相近表达。
- 正常结果无需反复核验。仅当指定来源与命中不一致、标题/编号/内容/数值/上下文明显冲突、候选歧义、内容截断或缺少直接证据时，才调整查询、证据类型或显式选择另一检索模式继续调查。
- 搜索命中带 truncated=true、缺少必要上下文、存在歧义或冲突、需要读取结构后续部分，或当前内容不足以支撑回答时，才调用 kb_read。kb_read 用于扩展内容，不代表其数据天然比搜索结果更可靠。
- 图片命中的 description/table_markdown 只表示已有辅助分析，不代表本轮查看了原图；locator_only=true 只用于定位图片，不是视觉事实。需要视觉信息时调用 kb_image_read，且不要批量打开所有定位器。文本模型只能说明发现了相关图片，不能声称已经核验原图。
- 用户明确要求讲解、查看或核对某张图片/图表时，只要搜索命中 evidence_type=image，就必须调用 kb_image_read；不得只依据搜索结果中的辅助描述声称已经看图。focus 中写明需要从原图核对的问题。
- 精确图片编号搜索返回 exact_image_references；missing 中的编号表示当前范围内确定未找到对应图片，不得换模式或移除图片限制后用相似图片冒充。只有用户要求扩大范围或寻找相关内容时才能另行搜索，并保留原编号未找到的结论。
- 使用明确 ref_key 调用 kb_image_read 返回 not_found 时，同样表示该编号在当前范围内确定未找到；不得继续用其他检索模式确认同一编号。
- 每次 kb_read 后检查 continuation：has_more=true 且返回 next_chunk_index 时只能按该值继续；返回 required_max_chars 时提高 max_chars 后重读同一分段；has_more=false 时不得自行递增 chunk_index。需要其他证据时重新搜索或通过 kb_list 导航。
- 用户指代“这些文档、所选资料、当前报告”但来源集合不明确，或询问当前资料是否覆盖、披露某事项时，先用无参数 kb_list 确认文档集合；普通问答不要默认列目录。
- 证据冲突时按用户指定来源回答并说明冲突；证据不足时明确说明，不得用相近内容补答。
- 最终回答不得出现 data_id、chunk_index、内部路径、处理后文件名或检索诊断。实际使用知识库时，在末尾按 source_hint 生成“信息来源”，同一原始文档合并并去重章节或图号。
""".strip()


KB_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "在当前知识库范围内召回文本和图片候选。query 是检索表达，不必逐字复制用户原话；可以在保持原始信息需求、范围和证据要求的前提下改写、改变粒度或拆分复杂问题，不得引入未经证实的前提。结果不足、冲突或明显错位时，可以基于上一轮结果显式调整 query、evidence_types 或 mode，但后续调用应有新的调查目的；若不再产生新证据，应停止穷举相近表达。工具本身不会静默切换 mode。evidence_types 只限制知识库记录的表现形式；用户指定的表格、原文或报告可能被表现为文本、表格或图片，形式不明确时不要预先硬过滤。图片命中带 locator_only=true 时只有图题、上下文和原图定位，不是视觉事实；需要视觉信息时再调用 kb_image_read，不要批量打开所有定位器。明确图号且仅检索 image 时，编号是硬约束并返回范围内全部同号图片，不用相似图片填充；exact_image_references.missing 表示当前范围内确定未找到，不得改用相似图片冒充。完整、明确且没有 truncated 标记的文本命中可以直接作为证据；内容截断、存在歧义、缺少必要上下文或需要结构后续部分时，再使用命中的 data_id 和 chunk_index 调用 kb_read。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言问题或检索词"},
                    "mode": {
                        "type": "string",
                        "enum": ["rrf", "vector", "sparse"],
                        "description": "必须显式选择且工具不会静默降级：rrf 融合语义和词语匹配结果；vector 适合语义表达；sparse 适合需要保留原始表达或精确符号的情况。Agent 可以在后续独立检索中基于当前结果显式选择另一模式。混合证据查询中的明确图表编号（如 图3-1、表4.1）作为独立确定性信号参与排序；仅检索 image 的精确编号查询直接执行确定性定位，不调用语义或词语检索通道。",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "最终候选数。精确定位通常使用较小值；问题范围较广、涉及多个来源或候选有歧义时可适当增大。增大只会扩大候选集合，不代表结果更可靠。仅检索 image 的精确编号查询是例外：返回当前范围内全部同号候选。",
                    },
                    "data_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 50,
                        "uniqueItems": True,
                        "description": "可选：仅在 kb_list 返回的这些文档中搜索。应原样复制文档 data_id；只能缩小当前会话范围，不接受图片 data_id。",
                    },
                    "evidence_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["text", "table", "image"],
                        },
                        "uniqueItems": True,
                        "description": "可选的记录表现形式硬过滤：text 只包含非图片、非表格的文本结构；table 包含已识别的文本表格及确有表格分析的图片；image 包含所有图片。它不等同于用户指定的证据来源：某张表或原文可能被处理为普通文本或图片，形式不明确时应不传本参数以检索全部类型。只有确实需要排除其他表现形式，或已从结果确认形式时才使用。若过滤代表用户明确要求（如查看原图），无结果时不得静默移除。",
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
            "description": "扩展读取 kb_search 命中的文本内容。表格、列表、代码、公式等会优先返回同一结构；普通正文由 span 控制。必须检查 continuation：has_more=true 时只使用返回的 next_chunk_index 续读，has_more=false 时不得自行递增 chunk_index；需要其他证据应重新搜索或通过 kb_list 导航。若正文与搜索内容明显冲突，以读取到的较完整内容为准。不要与依赖搜索结果的调用并行执行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "chunk_index": {"type": "integer", "minimum": 0},
                    "span": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 1,
                        "description": "普通正文需要读取的相邻分段数；结构化内容按同一结构读取。不得用它扫描未经搜索或导航定位的内容。",
                    },
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
            "description": "用于当前范围内的文档发现、来源集合确认和分段导航。不传 data_id 时列出允许访问的文档；传 data_id 时分页列出该文档的分段目录，并应使用返回的 next_offset 继续分页。offset/limit 仅在提供 data_id 时生效。不要在普通问答前默认调用，也不要把目录预览当作事实证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "提供 data_id 时使用的分页起点；后续分页使用工具返回的 next_offset。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20, "description": "提供 data_id 时每页返回的分段数。"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_image_read",
            "description": "读取知识库图片的原图、图题、辅助描述、必要上下文和引用信息。data_id 与 ref_key 至少提供一个；搜索已返回图片 data_id 时优先使用它，用户已给出明确图表编号并要求原图时也可以直接使用 ref_key。明确 ref_key 返回 not_found 表示该编号在当前范围内确定不存在，不要再换检索模式确认同一编号。用户明确要求讲解、查看或核对图片时必须调用本工具并在 focus 中写明视觉目标，不能只根据搜索描述回答。查看后应核对原图是否符合用户要求、编号、图题和来源；明显不一致时继续检索或说明疑点。调用成功后原图会自动加入下一轮多模态输入，不要批量打开所有命中图片。不得传入文档 ID、文本分段 ID或任意本地路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string"},
                    "ref_key": {"type": "string", "description": "图1-1、表8-1等知识库返回的图表编号"},
                    "focus": {
                        "type": "string",
                        "description": "补充需要从原图确认的视觉关注点。用户明确要求讲解或核对图片时应填写其具体关注点；不要填写内部路径或 ID。",
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
    def _document_data_id_parts(value):
        raw = str(value or "").strip()
        if not raw or "::image::" in raw or raw.count("::") != 1:
            return None
        kb_id, document = raw.split("::", 1)
        document = document.replace("\\", "/").lstrip("/")
        if not kb_id or not document:
            return None
        return kb_id, document

    @staticmethod
    def _data_id_lookup_key(value):
        normalized = unicodedata.normalize("NFKC", str(value or ""))
        normalized = normalized.replace("\\", "/")
        return re.sub(r"\s+", "", normalized)

    def _search_kwargs_for_data_ids(self, raw_data_ids):
        """Validate an Agent-requested document subset and narrow the scope.

        The session scope remains authoritative.  This helper only converts a
        verified subset into the existing ``scope_targets`` representation so
        retrieval does not gain a second filtering path.
        """
        if not isinstance(raw_data_ids, list) or not raw_data_ids or len(raw_data_ids) > 50:
            return None, "invalid_argument"

        requested = []
        seen = set()
        for value in raw_data_ids:
            parts = self._document_data_id_parts(value)
            if parts is None:
                return None, "invalid_argument"
            data_id = str(value).strip()
            if data_id in seen:
                continue
            seen.add(data_id)
            requested.append((data_id, parts[0]))

        backend = self._kb_backend()
        known_by_kb = {}
        try:
            for _data_id, kb_id in requested:
                if kb_id in known_by_kb:
                    continue
                known_by_kb[kb_id] = [
                    str(document.get("data_id") or "").strip()
                    for document in (backend.list_documents(kb_id=kb_id) or [])
                    if isinstance(document, dict) and document.get("data_id")
                ]
        except Exception:
            return None, "retrieval_unavailable"

        parsed = []
        resolved_seen = set()
        for requested_id, kb_id in requested:
            known = known_by_kb.get(kb_id, [])
            if requested_id in known:
                resolved_id = requested_id
            else:
                lookup_key = self._data_id_lookup_key(requested_id)
                matches = [
                    candidate for candidate in known
                    if self._data_id_lookup_key(candidate) == lookup_key
                ]
                if not matches:
                    return None, "not_found"
                if len(matches) != 1:
                    return None, "invalid_argument"
                resolved_id = matches[0]
            if not self._scope_allows_target(data_id=resolved_id):
                return None, "scope_denied"
            if resolved_id in resolved_seen:
                continue
            resolved_seen.add(resolved_id)
            parsed.append((resolved_id, kb_id))

        grouped = {}
        for data_id, kb_id in parsed:
            grouped.setdefault(kb_id, []).append({"data_id": data_id})
        return {
            "scope_targets": [
                {
                    "kb_id": kb_id,
                    "all_documents": False,
                    "documents": documents,
                }
                for kb_id, documents in grouped.items()
            ]
        }, None

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
        return KnowledgeBaseToolsMixin._clip_text_state(
            value, limit, suffix=suffix
        )[0]

    @staticmethod
    def _clip_text_state(value, limit: int, *, suffix=""):
        text = KnowledgeBaseToolsMixin._clean_content(value)
        if len(text) <= limit:
            return text, False
        return text[:limit].rstrip() + (suffix or "\n…[检索摘要已截断]"), True

    @classmethod
    def _clip_structured_rows(cls, value, limit: int, *, suffix=""):
        return cls._clip_structured_rows_state(value, limit, suffix=suffix)[0]

    @classmethod
    def _clip_structured_rows_state(cls, value, limit: int, *, suffix=""):
        text = cls._clean_content(value)
        if len(text) <= limit:
            return text, False
        cut = text.rfind("\n", 0, limit + 1)
        if cut < limit // 2:
            cut = limit
        return text[:cut].rstrip() + (suffix or "\n…[结构化内容已截断]"), True

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
            description, description_truncated = cls._clip_text_state(
                hit.get("description"), 1400
            )
            if description:
                result["description"] = description
            table, table_truncated = cls._clip_structured_rows_state(
                hit.get("table_markdown"), 2800,
                suffix="\n…[表格摘要已截断，完整内容请调用 kb_image_read]",
            )
            if table:
                result["table_markdown"] = table
            if description or table:
                uncertain = hit.get("uncertain")
                if isinstance(uncertain, list):
                    values = [cls._clip_text(item, 160) for item in uncertain[:6]]
                    values = [item for item in values if item]
                    if values:
                        result["uncertain"] = values
                elif uncertain:
                    result["uncertain"] = [cls._clip_text(uncertain, 160)]
            else:
                result["locator_only"] = True
            if description_truncated or table_truncated:
                result["truncated"] = True
        else:
            snippet = cls._clip_text(hit.get("snippet"), 320)
            clip = (
                cls._clip_structured_rows_state
                if result["evidence_type"] == "table"
                else cls._clip_text_state
            )
            body, body_truncated = clip(
                hit.get("body"), 1600,
                suffix="\n…[正文摘要已截断，完整内容请调用 kb_read]",
            )
            if snippet:
                result["snippet"] = snippet
            if body:
                result["body"] = body
            if body_truncated or int(hit.get("structure_part_count") or 1) > 1:
                result["truncated"] = True
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
    def _safe_error(tool, code, *, field=None, message=None, allowed_values=None):
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
            "target_conflict": "提供的图片定位参数互相冲突。",
        }
        payload = {
            "error_code": code,
            "message": message or messages.get(code, "操作失败。"),
        }
        if field:
            payload["field"] = str(field)
        if allowed_values:
            payload["allowed_values"] = list(allowed_values)
        return json.dumps(payload, ensure_ascii=False)

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
                args, self._safe_error(
                    "kb_search", "invalid_argument", field="query",
                    message="kb_search 缺少必填参数 query。",
                )
            )
        mode = str(args.get("mode") or "").strip().lower()
        if mode not in {"rrf", "vector", "sparse"}:
            message = (
                "kb_search 缺少必填参数 mode。"
                if not mode else
                "kb_search 的 mode 必须是 rrf、vector 或 sparse。"
            )
            return self._anchor_outcome(
                args, self._safe_error(
                    "kb_search", "invalid_argument", field="mode",
                    message=message,
                    allowed_values=["rrf", "vector", "sparse"],
                )
            )
        if "top_k" in args:
            try:
                top_k = int(args.get("top_k"))
            except (TypeError, ValueError):
                top_k = 0
            if not 1 <= top_k <= 10:
                return self._anchor_outcome(
                    args, self._safe_error(
                        "kb_search", "invalid_argument", field="top_k",
                        message="kb_search 的 top_k 必须是 1 到 10 之间的整数。",
                    )
                )
        else:
            top_k = 5
        evidence_types = args.get("evidence_types") or []
        if not isinstance(evidence_types, list):
            return self._anchor_outcome(
                args, self._safe_error(
                    "kb_search", "invalid_argument", field="evidence_types",
                    message="kb_search 的 evidence_types 必须是数组。",
                )
            )
        allowed_evidence = {"text", "table", "image"}
        evidence_types = [
            str(value).strip().lower() for value in evidence_types if str(value).strip()
        ]
        if set(evidence_types) - allowed_evidence:
            return self._anchor_outcome(
                args, self._safe_error(
                    "kb_search", "invalid_argument", field="evidence_types",
                    message="kb_search 的 evidence_types 只能包含 text、table 或 image。",
                    allowed_values=["text", "table", "image"],
                )
            )
        evidence_types = list(dict.fromkeys(evidence_types))
        search_kwargs = self._scope_search_kwargs()
        if "data_ids" in args:
            search_kwargs, error_code = self._search_kwargs_for_data_ids(
                args.get("data_ids")
            )
            if error_code:
                return self._anchor_outcome(
                    args, self._safe_error(
                        "kb_search", error_code, field="data_ids",
                        message=(
                            "kb_search 的 data_ids 参数无效。"
                            if error_code == "invalid_argument" else None
                        ),
                    )
                )
        try:
            search_result = self._kb_backend().search(
                query,
                top_k=top_k,
                mode=mode,
                evidence_types=evidence_types,
                **search_kwargs,
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
        exact_references = search_result.get("exact_image_references")
        if isinstance(exact_references, dict):
            result["exact_image_references"] = {
                key: [self._clip_text(value, 80) for value in exact_references.get(key, [])]
                for key in ("requested", "matched", "missing")
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
                args, self._safe_error(
                    "kb_read", "invalid_argument", field="data_id",
                    message="kb_read 缺少必填参数 data_id。",
                )
            )
        if "chunk_index" not in args or args.get("chunk_index") is None:
            return self._anchor_outcome(
                args, self._safe_error(
                    "kb_read", "invalid_argument", field="chunk_index",
                    message="kb_read 缺少必填参数 chunk_index。",
                )
            )
        if not self._scope_allows_target(data_id=data_id):
            return self._anchor_outcome(
                args, self._safe_error("kb_read", "scope_denied")
            )
        numeric_rules = (
            ("chunk_index", args.get("chunk_index"), 0, None),
            ("span", args.get("span", 1), 1, 5),
            ("max_chars", args.get("max_chars", 4000), 500, 8000),
        )
        parsed = {}
        for field, value, minimum, maximum in numeric_rules:
            try:
                parsed[field] = int(value)
            except (TypeError, ValueError):
                parsed[field] = minimum - 1
            if parsed[field] < minimum or (maximum is not None and parsed[field] > maximum):
                bounds = f"不小于 {minimum}" if maximum is None else f"{minimum} 到 {maximum} 之间"
                return self._anchor_outcome(
                    args, self._safe_error(
                        "kb_read", "invalid_argument", field=field,
                        message=f"kb_read 的 {field} 必须是{bounds}的整数。",
                    )
                )
        start, span, max_chars = parsed["chunk_index"], parsed["span"], parsed["max_chars"]
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
            values = {}
            for field, default, minimum, maximum in (
                ("offset", 0, 0, None),
                ("limit", 20, 1, 50),
            ):
                try:
                    values[field] = int(args.get(field, default))
                except (TypeError, ValueError):
                    values[field] = minimum - 1
                if values[field] < minimum or (
                    maximum is not None and values[field] > maximum
                ):
                    bounds = (
                        f"不小于 {minimum}"
                        if maximum is None else f"{minimum} 到 {maximum} 之间"
                    )
                    return self._anchor_outcome(
                        args, self._safe_error(
                            "kb_list", "invalid_argument", field=field,
                            message=f"kb_list 的 {field} 必须是{bounds}的整数。",
                        )
                    )
            offset, page_limit = values["offset"], values["limit"]
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
            if "offset" in args or "limit" in args:
                return self._anchor_outcome(
                    args, self._safe_error(
                        "kb_list", "invalid_argument", field="data_id",
                        message="kb_list 只有在提供 data_id 时才能使用 offset 或 limit。",
                    )
                )
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
            return None, self._safe_error(
                "kb_image_read", "invalid_argument", field="data_id",
                message="kb_image_read 至少需要提供图片 data_id 或 ref_key。",
            )
        if lookup["data_id"] and "::image::" not in lookup["data_id"]:
            return None, self._safe_error(
                "kb_image_read", "invalid_argument", field="data_id",
                message="kb_image_read 的 data_id 必须是搜索返回的图片级 data_id。",
            )
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
                    info = {"error_code": "image_not_found"}
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
        if isinstance(info, dict) and info.get("error_code") == "image_target_conflict":
            return None, self._safe_error(
                "kb_image_read", "target_conflict", field="ref_key",
                message="kb_image_read 的 data_id 与 ref_key 不指向同一图片。",
            )
        if isinstance(info, dict) and info.get("error_code") == "image_not_found":
            return None, self._safe_error("kb_image_read", "not_found")
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
        public = self._public_image(info)
        context = "[知识库图片原图]\n"
        if public.get("source_hint"):
            context += f"定位来源: {public['source_hint']}\n"
        if public.get("ref_key"):
            context += f"定位编号: {public['ref_key']}\n"
        if focus:
            context += f"本次查看重点: {focus}\n"
        context += (
            "请直接查看原图。工具提供的描述和上下文仅用于辅助定位；"
            "若原图内容与查看目标、定位编号或来源明显不符，不要按当前候选强行回答，"
            "应继续检索其他候选。"
        )
        _metadata, error = queue_image(
            path,
            name=info.get("display_label") or info.get("title") or os.path.basename(path),
            context=context,
        )
        if error:
            return f"[{error.get('error_code')}] {error.get('error')}"
        return None
