import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from knowledge_base.agent_tools import (
    KB_AGENT_SYSTEM_INSTRUCTIONS,
    KB_TOOL_SCHEMAS,
    KnowledgeBaseToolsMixin,
)


class _Backend:
    image_abspath = ""

    @staticmethod
    def search(query, **_kwargs):
        return {"mode": "vector", "results": []}

    @staticmethod
    def find_terms(terms, **_kwargs):
        return {
            "terms": list(terms),
            "match": _kwargs.get("match", "any"),
            "coverage": "searchable_records",
            "complete": True,
            "searched_documents": 2,
            "documents": [{
                "data_id": "kb-test::documents/source.md",
                "source_file_name": "source.pdf",
                "matched_terms": [terms[0]],
                "match_count": 1,
                "occurrences": [{
                    "data_id": "kb-test::documents/source.md",
                    "chunk_index": 7,
                    "kind": "text",
                    "content_type": "table",
                    "source_file_name": "source.pdf",
                    "header_path": "/财务/分红/",
                    "structure_title": "股利支付率",
                    "matched_terms": [terms[0]],
                    "preview": "股利支付率 70%",
                }],
                "has_more_occurrences": False,
            }],
            "unmatched_documents": [{
                "data_id": "kb-test::documents/other.md",
                "source_file_name": "other.pdf",
            }],
            "warnings": [],
        }

    @staticmethod
    def read_content(**kwargs):
        index = int(kwargs.get("chunk_index") or 0)
        return {
            "data_id": kwargs.get("data_id") or "kb-test::documents/source.md",
            "content_type": "prose",
            "start_chunk_index": index,
            "end_chunk_index": index,
            "content": "正文",
            "continuation": {
                "has_more": False,
                "next_chunk_index": None,
            },
        }

    @staticmethod
    def reference_for_chunk(**_kwargs):
        return {
            "kind": "document",
            "source_file_name": "source.pdf",
            "source_section": "方法",
            "citation_label": "source.pdf · 方法",
        }

    @staticmethod
    def read_image(**_kwargs):
        return {
            "kind": "image",
            "kb_id": "kb-test",
            "data_id": "kb-test::doc.md::image::1",
            "image_id": "image-1",
            "image_abspath": _Backend.image_abspath,
            "source_file_name": "source.pdf",
            "ref_key": "图1",
            "display_label": "图1：流程",
            "description": "diagram",
        }

    @staticmethod
    def list_documents(**_kwargs):
        return [{
            "kind": "document",
            "kb_id": "kb-test",
            "data_id": "kb-test::documents/source.md",
            "file_name": "documents/source.md",
            "source_file_name": "source.pdf",
            "folder": "files",
            "size": 100,
            "source_size": 200,
            "source_exists": True,
        }]

    @staticmethod
    def list_chunks(**kwargs):
        offset = int(kwargs.get("offset") or 0)
        limit = int(kwargs.get("limit") or 20)
        end = min(45, offset + limit)
        return {
            "data_id": "kb-test::documents/source.md",
            "title": "source.pdf",
            "file_name": "documents/source.md",
            "offset": offset,
            "limit": limit,
            "returned": max(0, end - offset),
            "has_more": end < 45,
            "next_offset": end if end < 45 else None,
            "chunks": [
                {"chunk_index": index, "chars": 100 + index, "preview": f"第 {index} 段 ![](documents/image-{index}.jpg)"}
                for index in range(offset, end)
            ],
        }


class _ScopedImageBackend:
    image_abspath = ""
    calls = []

    @classmethod
    def read_image(cls, **kwargs):
        cls.calls.append(dict(kwargs))
        return {
            "kind": "image",
            "kb_id": "kb-test",
            "data_id": "kb-test::doc.md::image::1",
            "image_id": "image-1",
            "image_abspath": cls.image_abspath,
            "source_data_id": kwargs.get("source_data_id"),
            "source_file_name": "source.pdf",
            "description": "diagram",
        }


class _Handler(KnowledgeBaseToolsMixin):
    def __init__(self):
        self.parent = SimpleNamespace(knowledge_scope={"mode": "all"})

    @staticmethod
    def _kb_backend():
        return _Backend

    @staticmethod
    def _get_anchor_prompt(skip=False):
        return "\n" if skip else "\nanchor"

    def queue_image_for_next_turn(self, path, **_kwargs):
        self.queued_image = path
        self.queued_context = _kwargs.get("context", "")
        return {"attach_status": "attached"}, None


class _ScopedImageHandler(_Handler):
    def __init__(self, scope):
        self.parent = SimpleNamespace(knowledge_scope=scope)

    @staticmethod
    def _kb_backend():
        return _ScopedImageBackend


class _SubsetBackend(_Backend):
    search_calls = []
    find_calls = []
    documents = {
        "kb-a": [
            {"data_id": "kb-a::documents/one.md"},
            {"data_id": "kb-a::documents/two.md"},
        ],
        "kb-b": [{"data_id": "kb-b::documents/three.md"}],
    }

    @classmethod
    def list_documents(cls, kb_id=None):
        return list(cls.documents.get(kb_id, []))

    @classmethod
    def search(cls, query, **kwargs):
        cls.search_calls.append({"query": query, **kwargs})
        return {"mode": kwargs.get("mode"), "results": []}

    @classmethod
    def find_terms(cls, terms, **kwargs):
        cls.find_calls.append({"terms": list(terms), **kwargs})
        return {
            "complete": True,
            "searched_documents": len(kwargs.get("scope_targets") or []),
            "documents": [],
            "unmatched_documents": [],
        }


class _SubsetHandler(_Handler):
    def __init__(self, scope):
        self.parent = SimpleNamespace(knowledge_scope=scope)

    @staticmethod
    def _kb_backend():
        return _SubsetBackend


class _FailingBackend:
    @staticmethod
    def search(**_kwargs):
        raise RuntimeError(r"C:\secret\zvec")

    @staticmethod
    def find_terms(*_args, **_kwargs):
        raise RuntimeError(r"C:\secret\records.jsonl")

    @staticmethod
    def read_content(**_kwargs):
        raise RuntimeError(r"C:\secret\zvec")

    @staticmethod
    def reference_for_chunk(**_kwargs):
        return {"source_file_name": "source.pdf"}

    @staticmethod
    def list_chunks(**_kwargs):
        return {"error": r"[Zvec 读取失败] C:\secret\zvec"}

    @staticmethod
    def read_image(**_kwargs):
        return {"error": r"[Zvec 读取失败] C:\secret\zvec"}


class _FailingHandler(_Handler):
    @staticmethod
    def _kb_backend():
        return _FailingBackend


class KnowledgeBaseAgentSchemaTests(unittest.TestCase):
    def test_tool_descriptions_define_agent_workflow(self):
        functions = {
            item["function"]["name"]: item["function"]
            for item in KB_TOOL_SCHEMAS
        }

        self.assertIn("候选", functions["kb_search"]["description"])
        self.assertIn("不是全文扫描", functions["kb_search"]["description"])
        self.assertIn("改写或拆分", functions["kb_search"]["parameters"]["properties"]["query"]["description"])
        self.assertIn("continuation", functions["kb_read"]["description"])
        self.assertEqual(
            functions["kb_read"]["parameters"]["required"],
            ["data_id", "chunk_index"],
        )
        self.assertIn(
            "原样使用",
            functions["kb_read"]["parameters"]["properties"]["chunk_index"]["description"],
        )
        self.assertIn("导航", functions["kb_list"]["description"])
        self.assertIn("完整扫描", functions["kb_find"]["description"])
        self.assertIn("不提供词频", functions["kb_find"]["description"])
        self.assertIn(
            "知识库前缀",
            functions["kb_search"]["parameters"]["properties"]["data_ids"]["description"],
        )
        self.assertIn(
            "知识库前缀",
            functions["kb_find"]["parameters"]["properties"]["data_ids"]["description"],
        )
        self.assertIn(
            "图片级 data_id",
            functions["kb_image_read"]["parameters"]["properties"]["data_id"]["description"],
        )
        self.assertIn("同一模型回合只能成功加入一张", functions["kb_image_read"]["description"])
        self.assertIn(
            "明确且唯一",
            functions["kb_image_read"]["parameters"]["properties"]["ref_key"]["description"],
        )

    def test_search_requires_agent_to_choose_mode(self):
        search = next(
            item["function"]
            for item in KB_TOOL_SCHEMAS
            if item["function"]["name"] == "kb_search"
        )

        self.assertIn("mode", search["parameters"]["required"])
        self.assertNotIn(
            "default",
            search["parameters"]["properties"]["mode"],
        )
        top_k = search["parameters"]["properties"]["top_k"]
        self.assertEqual(top_k["default"], 5)
        self.assertEqual((top_k["minimum"], top_k["maximum"]), (1, 10))
        self.assertIn("不代表结果更可靠", top_k["description"])
        self.assertIn("全部同号候选", top_k["description"])
        self.assertIn("不调用语义或词语检索通道", search["parameters"]["properties"]["mode"]["description"])
        data_ids = search["parameters"]["properties"]["data_ids"]
        self.assertEqual((data_ids["minItems"], data_ids["maxItems"]), (1, 50))
        evidence_description = search["parameters"]["properties"]["evidence_types"]["description"]
        self.assertIn("不等同于用户指定的证据来源", evidence_description)
        self.assertIn("形式不明确时应不传", evidence_description)

    def test_search_argument_errors_identify_the_field(self):
        handler = _Handler()
        cases = [
            ({"query": "test"}, "mode"),
            ({"query": "test", "mode": "auto"}, "mode"),
            ({"query": "test", "mode": "rrf", "top_k": 11}, "top_k"),
            ({"query": "test", "mode": "rrf", "evidence_types": "table"}, "evidence_types"),
        ]
        for args, field in cases:
            with self.subTest(args=args):
                payload = json.loads(handler.do_kb_search(args, None).data)
                self.assertEqual(payload["error_code"], "invalid_argument")
                self.assertEqual(payload["field"], field)
        missing_mode = json.loads(handler.do_kb_search({"query": "test"}, None).data)
        self.assertEqual(missing_mode["allowed_values"], ["rrf", "vector", "sparse"])

    def test_find_returns_compact_public_absence_contract(self):
        payload = json.loads(_Handler().do_kb_find({
            "terms": ["股利支付率"], "match": "any",
        }, None).data)

        self.assertTrue(payload["absence_supported"])
        self.assertEqual(payload["documents"][0]["source_hint"], "《source.pdf》")
        self.assertEqual(payload["documents"][0]["matched_terms"], ["股利支付率"])
        self.assertEqual(set(payload), {
            "absence_supported", "documents", "unmatched_documents",
        })
        self.assertNotIn("occurrences", json.dumps(payload))
        self.assertNotIn("searched_documents", payload)
        self.assertNotIn("kb_id", json.dumps(payload))
        self.assertNotIn("records.jsonl", json.dumps(payload))

    def test_find_does_not_return_image_locations_or_analysis(self):
        original = _Backend.find_terms
        try:
            _Backend.find_terms = staticmethod(lambda terms, **_kwargs: {
                "complete": True,
                "searched_documents": 1,
                "documents": [{
                    "data_id": "kb-test::documents/source.md",
                    "source_file_name": "source.pdf",
                    "matched_terms": list(terms),
                    "match_count": 1,
                    "occurrences": [{
                        "data_id": "kb-test::documents/source.md::image::1",
                        "kind": "image",
                        "source_file_name": "source.pdf",
                        "ref_key": "图1",
                        "matched_terms": list(terms),
                        "preview": "图1 分红流程",
                    }],
                }],
                "unmatched_documents": [],
            })
            payload = json.loads(_Handler().do_kb_find({
                "terms": ["分红"],
            }, None).data)
        finally:
            _Backend.find_terms = original

        self.assertTrue(payload["absence_supported"])
        self.assertEqual(set(payload["documents"][0]), {
            "data_id", "source_hint", "matched_terms",
        })
        self.assertNotIn("occurrences", json.dumps(payload))
        self.assertNotIn("image_matches_require_read", payload)

    def test_find_incomplete_scan_returns_only_safe_warning(self):
        original = _Backend.find_terms
        try:
            _Backend.find_terms = staticmethod(lambda *_args, **_kwargs: {
                "complete": False,
                "searched_documents": 0,
                "documents": [],
                "unmatched_documents": [],
                "warnings": [{
                    "error_code": "records_unavailable",
                    "error": r"C:\secret\records.jsonl",
                }],
            })
            output = _Handler().do_kb_find({"terms": ["分红"]}, None).data
            payload = json.loads(output)
        finally:
            _Backend.find_terms = original

        self.assertFalse(payload["absence_supported"])
        self.assertEqual(payload["warnings"], [{
            "error_code": "scan_incomplete",
            "message": "部分知识库记录无法扫描，不得据此作完整性或否定性结论。",
        }])
        self.assertNotIn("secret", output)
        self.assertNotIn("records.jsonl", output)

    def test_find_scope_too_broad_is_a_structured_error(self):
        original = _Backend.find_terms
        try:
            _Backend.find_terms = staticmethod(lambda *_args, **_kwargs: {
                "error_code": "scope_too_broad",
                "searched_documents": 51,
            })
            payload = json.loads(_Handler().do_kb_find({
                "terms": ["分红"],
            }, None).data)
        finally:
            _Backend.find_terms = original

        self.assertEqual(payload["error_code"], "scope_too_broad")
        self.assertIn("data_ids", payload["message"])

    def test_find_argument_errors_identify_the_field(self):
        handler = _Handler()
        cases = [
            ({}, "terms"),
            ({"terms": "分红"}, "terms"),
            ({"terms": [""]}, "terms"),
            ({"terms": ["分红"], "match": "near"}, "match"),
            ({"terms": ["分红"], "case_sensitive": "false"}, "case_sensitive"),
        ]
        for args, field in cases:
            with self.subTest(args=args):
                payload = json.loads(handler.do_kb_find(args, None).data)
                self.assertEqual(payload["error_code"], "invalid_argument")
                self.assertEqual(payload["field"], field)

    def test_find_document_subset_uses_the_same_scope_intersection(self):
        _SubsetBackend.find_calls = []
        handler = _SubsetHandler({
            "mode": "selection",
            "targets": [{
                "kb_id": "kb-a", "all_documents": False,
                "documents": [{"data_id": "kb-a::documents/one.md"}],
            }, {"kb_id": "kb-b", "all_documents": True}],
        })

        payload = json.loads(handler.do_kb_find({
            "terms": ["分红"],
            "data_ids": ["kb-a::documents/one.md", "kb-b::documents/three.md"],
        }, None).data)

        self.assertNotIn("error_code", payload)
        targets = _SubsetBackend.find_calls[0]["scope_targets"]
        self.assertEqual(sum(len(item["documents"]) for item in targets), 2)

    def test_exact_image_reference_status_is_forwarded(self):
        original = _Backend.search
        try:
            _Backend.search = staticmethod(lambda *_args, **_kwargs: {
                "mode": "sparse",
                "results": [
                    {
                        "kind": "image",
                        "data_id": "kb-test::documents/a.md::image::79",
                        "source_file_name": "a.pdf",
                        "ref_key": "图79",
                        "display_label": "图79 产品结构",
                        "score_type": "ref_exact",
                        "matched_by": ["ref_exact"],
                    },
                    {
                        "kind": "image",
                        "data_id": "kb-test::documents/b.md::image::79",
                        "source_file_name": "b.pdf",
                        "ref_key": "图79",
                        "display_label": "图79 市场结构",
                        "score_type": "ref_exact",
                        "matched_by": ["ref_exact"],
                    },
                ],
                "exact_image_references": {
                    "requested": ["图79", "图999"],
                    "matched": ["图79"],
                    "missing": ["图999"],
                    "ambiguous": ["图79"],
                },
            })
            payload = json.loads(_Handler().do_kb_search({
                "query": "图79和图999", "mode": "sparse",
                "evidence_types": ["image"],
            }, None).data)
        finally:
            _Backend.search = original

        self.assertEqual(payload["missing_image_refs"], ["图999"])
        self.assertEqual(payload["ambiguous_image_refs"], ["图79"])
        self.assertFalse(payload["absence_supported"])
        self.assertEqual(len(payload["hits"]), 2)
        self.assertEqual(
            {hit["data_id"] for hit in payload["hits"]},
            {
                "kb-test::documents/a.md::image::79",
                "kb-test::documents/b.md::image::79",
            },
        )
        self.assertTrue(all(hit["ref_key"] == "图79" for hit in payload["hits"]))

    def test_search_document_subset_is_validated_and_uses_scope_targets(self):
        _SubsetBackend.search_calls = []
        handler = _SubsetHandler({"mode": "all"})

        outcome = handler.do_kb_search({
            "query": "对比",
            "mode": "rrf",
            "top_k": 8,
            "data_ids": [
                "kb-b::documents/three.md",
                "kb-a::documents/one.md",
                "kb-b::documents/three.md",
            ],
        }, None)

        self.assertNotIn("error_code", json.loads(outcome.data))
        call = _SubsetBackend.search_calls[0]
        self.assertEqual(call["top_k"], 8)
        self.assertEqual(call["scope_targets"], [
            {
                "kb_id": "kb-b",
                "all_documents": False,
                "documents": [{"data_id": "kb-b::documents/three.md"}],
            },
            {
                "kb_id": "kb-a",
                "all_documents": False,
                "documents": [{"data_id": "kb-a::documents/one.md"}],
            },
        ])

    def test_search_document_subset_resolves_unique_cosmetic_whitespace(self):
        _SubsetBackend.search_calls = []
        outcome = _SubsetHandler({"mode": "all"}).do_kb_search({
            "query": "test",
            "mode": "vector",
            "data_ids": ["kb-a::documents/one - report.md"],
        }, None)

        # Add one realistic long handle for this contract: models sometimes
        # insert spaces around punctuation while copying an opaque ID.
        self.assertEqual(json.loads(outcome.data)["error_code"], "not_found")

        original = _SubsetBackend.documents["kb-a"]
        try:
            _SubsetBackend.documents["kb-a"] = [
                {"data_id": "kb-a::documents/one-report.md"},
            ]
            outcome = _SubsetHandler({"mode": "all"}).do_kb_search({
                "query": "test",
                "mode": "vector",
                "data_ids": ["kb-a::documents/one - report.md"],
            }, None)
        finally:
            _SubsetBackend.documents["kb-a"] = original

        self.assertNotIn("error_code", json.loads(outcome.data))
        self.assertEqual(
            _SubsetBackend.search_calls[-1]["scope_targets"][0]["documents"],
            [{"data_id": "kb-a::documents/one-report.md"}],
        )

    def test_search_document_subset_fails_as_a_whole_for_invalid_targets(self):
        cases = [
            (["kb-a::documents/one.md::image::1"], "invalid_argument"),
            (["not-a-document-id"], "invalid_argument"),
            (["kb-a::documents/missing.md"], "not_found"),
        ]
        for data_ids, error_code in cases:
            with self.subTest(data_ids=data_ids):
                _SubsetBackend.search_calls = []
                outcome = _SubsetHandler({"mode": "all"}).do_kb_search({
                    "query": "test", "mode": "vector", "data_ids": data_ids,
                }, None)
                self.assertEqual(json.loads(outcome.data)["error_code"], error_code)
                self.assertEqual(_SubsetBackend.search_calls, [])

    def test_search_document_subset_cannot_expand_session_scope(self):
        _SubsetBackend.search_calls = []
        handler = _SubsetHandler({
            "mode": "document",
            "kb_id": "kb-a",
            "data_id": "kb-a::documents/one.md",
        })

        outcome = handler.do_kb_search({
            "query": "test",
            "mode": "vector",
            "data_ids": ["kb-a::documents/two.md"],
        }, None)

        self.assertEqual(json.loads(outcome.data)["error_code"], "scope_denied")
        self.assertEqual(_SubsetBackend.search_calls, [])

    def test_search_document_subset_respects_kb_and_selection_scopes(self):
        scopes_and_targets = [
            (
                {"mode": "kb", "kb_id": "kb-a"},
                ["kb-a::documents/two.md"],
            ),
            (
                {
                    "mode": "selection",
                    "targets": [
                        {
                            "kb_id": "kb-a",
                            "documents": [
                                {"data_id": "kb-a::documents/one.md"},
                            ],
                        },
                        {"kb_id": "kb-b", "all_documents": True},
                    ],
                },
                [
                    "kb-a::documents/one.md",
                    "kb-b::documents/three.md",
                ],
            ),
        ]
        for scope, data_ids in scopes_and_targets:
            with self.subTest(scope=scope):
                _SubsetBackend.search_calls = []
                outcome = _SubsetHandler(scope).do_kb_search({
                    "query": "test", "mode": "rrf", "data_ids": data_ids,
                }, None)
                self.assertNotIn("error_code", json.loads(outcome.data))
                self.assertEqual(
                    sum(
                        len(target["documents"])
                        for target in _SubsetBackend.search_calls[0]["scope_targets"]
                    ),
                    len(data_ids),
                )

    def test_kb_image_read_has_no_attach_switch_and_always_queues_image(self):
        schema = next(
            item["function"]
            for item in KB_TOOL_SCHEMAS
            if item["function"]["name"] == "kb_image_read"
        )
        self.assertNotIn("attach_image", schema["parameters"]["properties"])

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "figure.png"
            Image.new("RGB", (2, 2), "red").save(image)
            _Backend.image_abspath = str(image)
            try:
                handler = _Handler()
                outcome = handler.do_kb_image_read(
                    {
                        "data_id": "kb-test::doc.md::image::1",
                        "focus": "确认图中红色流程与蓝色流程的先后关系",
                    },
                    None,
                )

                self.assertEqual(handler.queued_image, str(image))
                self.assertIn("定位来源: 《source.pdf》：“图1：流程”", handler.queued_context)
                self.assertIn("定位编号: 图1", handler.queued_context)
                self.assertIn("本次查看重点: 确认图中红色流程与蓝色流程的先后关系", handler.queued_context)
                self.assertIn("继续检索其他候选", handler.queued_context)
                self.assertNotIn("kb-test", handler.queued_context)
                self.assertNotIn(str(image), handler.queued_context)
                self.assertEqual(set(json.loads(outcome.data)), {
                    "source_hint", "ref_key", "image_attached",
                })
            finally:
                _Backend.image_abspath = ""

    def test_kb_image_read_allows_only_one_success_per_model_response(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "figure.png"
            Image.new("RGB", (2, 2), "red").save(image)
            _Backend.image_abspath = str(image)
            try:
                handler = _Handler()
                first_response = SimpleNamespace()
                first = json.loads(handler.do_kb_image_read({
                    "data_id": "kb-test::doc.md::image::1",
                }, first_response).data)
                second = json.loads(handler.do_kb_image_read({
                    "data_id": "kb-test::doc.md::image::1",
                }, first_response).data)
                third = json.loads(handler.do_kb_image_read({
                    "data_id": "kb-test::doc.md::image::1",
                }, SimpleNamespace()).data)
            finally:
                _Backend.image_abspath = ""

        self.assertTrue(first["image_attached"])
        self.assertEqual(second["error_code"], "sequential_image_read_required")
        self.assertTrue(third["image_attached"])

    def test_document_scope_passes_source_document_to_figure_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "figure.png"
            Image.new("RGB", (2, 2), "red").save(image)
            _ScopedImageBackend.image_abspath = str(image)
            _ScopedImageBackend.calls = []
            try:
                handler = _ScopedImageHandler({
                    "mode": "document",
                    "kb_id": "kb-test",
                    "data_id": "kb-test::doc.md",
                })
                outcome = handler.do_kb_image_read({"ref_key": "图1"}, None)

                self.assertIn('"image_attached": true', outcome.data)
                self.assertEqual(
                    _ScopedImageBackend.calls[0]["source_data_id"],
                    "kb-test::doc.md",
                )
            finally:
                _ScopedImageBackend.image_abspath = ""

    def test_search_returns_payload_without_info_status_line(self):
        outcome = _Handler().do_kb_search(
            {"query": "SkillOpt", "mode": "vector"}, None,
        )

        self.assertNotIn("[Info] kb_search done.", outcome.data)
        self.assertIn('"absence_supported": false', outcome.data)
        self.assertNotIn('"mode"', outcome.data)
        self.assertNotIn('"scope"', outcome.data)
        self.assertNotIn('"diagnostics"', outcome.data)

    def test_search_for_explicit_table_evidence_preserves_the_agent_constraint(self):
        captured = {}
        original = _Backend.search
        try:
            def search(query, **kwargs):
                captured.update(kwargs)
                return {"mode": kwargs["mode"], "results": []}

            _Backend.search = staticmethod(search)
            outcome = _Handler().do_kb_search({
                "query": "根据重要财务指标表查询资产负债率",
                "mode": "sparse",
                "evidence_types": ["table"],
            }, None)
        finally:
            _Backend.search = original

        payload = json.loads(outcome.data)
        self.assertEqual(captured["evidence_types"], ["table"])
        self.assertNotIn("evidence_types", payload)

    def test_search_passes_public_text_type_without_internal_expansion(self):
        captured = {}
        original = _Backend.search
        try:
            def search(query, **kwargs):
                captured.update(kwargs)
                return {"mode": kwargs["mode"], "results": []}

            _Backend.search = staticmethod(search)
            _Handler().do_kb_search({
                "query": "只检索正文",
                "mode": "vector",
                "evidence_types": ["text"],
            }, None)
        finally:
            _Backend.search = original

        self.assertEqual(captured["evidence_types"], ["text"])

    def test_truncated_table_search_returns_only_a_required_read_locator(self):
        body = "| 指标 | 2020E |\n| --- | --- |\n" + "\n".join(
            f"| 指标{i} | {i}% |" for i in range(200)
        )
        result = _Handler._clean_hit({
            "kind": "text",
            "data_id": "kb-test::documents/source.md",
            "chunk_index": 2,
            "source_file_name": "source.pdf",
            "content_type": "table",
            "structure_title": "重要财务指标",
            "body": body,
            "snippet": "资产负债率",
        })

        self.assertTrue(result["requires_read"])
        self.assertNotIn("body", result)
        self.assertNotIn("snippet", result)
        self.assertNotIn("truncated", result)
        self.assertNotIn("199%", str(result))
        self.assertEqual(result["source_hint"], "《source.pdf》：“重要财务指标”")

    def test_source_hint_keeps_section_and_structure_title(self):
        result = _Handler._clean_hit({
            "kind": "text",
            "data_id": "kb-test::documents/source.md",
            "chunk_index": 2,
            "source_file_name": "source.pdf",
            "header_path": "/公司/财务数据/",
            "content_type": "table",
            "structure_title": "重要财务指标",
            "body": "| 指标 | 值 |",
        })

        self.assertEqual(
            result["source_hint"],
            "《source.pdf》：“财务数据”——“重要财务指标”",
        )

    def test_search_payload_compacts_image_context_and_hides_scope(self):
        hit = {
            "kind": "image",
            "kb_id": "kb-test",
            "data_id": "kb-test::doc.md::image::1",
            "image_id": "image-1",
            "source_file_name": "source.pdf",
            "file_name": "documents/internal.md",
            "description": "描述 " * 1000,
            "body": "重复图片正文 " * 1000,
            "related_text": "重复关联正文 " * 1000,
            "near_text": "重复邻近正文 " * 1000,
            "display_label": "图1",
            "ref_key": "图1",
        }

        compact = _Handler._clean_hit(hit)

        self.assertLessEqual(len(compact["description"]), 1450)
        self.assertNotIn("body", compact)
        self.assertNotIn("related_text", compact)
        self.assertNotIn("near_text", compact)
        self.assertNotIn("abspath", compact)
        self.assertNotIn("kb_id", compact)
        self.assertNotIn("source_data_id", compact)
        self.assertNotIn("ref", compact)
        self.assertTrue(compact["requires_image_read"])

    def test_truncated_text_search_returns_only_a_required_read_locator(self):
        compact = _Handler._clean_hit({
            "kind": "document",
            "kb_id": "kb-test",
            "data_id": "kb-test::doc.md",
            "file_name": "documents/internal.md",
            "source_file_name": "source.pdf",
            "matched_by": ["vector"],
            "body": "正文 " * 1000,
            "snippet": "命中摘要 " * 200,
        })

        self.assertEqual(set(compact), {
            "data_id", "chunk_index", "evidence_type", "source_hint",
            "requires_read",
        })
        self.assertTrue(compact["requires_read"])
        self.assertNotIn("正文", str(compact))
        self.assertNotIn("命中摘要", str(compact))

    def test_complete_search_hit_keeps_content_and_omits_read_requirement(self):
        compact = _Handler._clean_hit({
            "kind": "document",
            "data_id": "kb-test::documents/source.md",
            "source_file_name": "source.pdf",
            "chunk_index": 4,
            "body": "完整且明确的正文",
            "snippet": "明确摘要",
            "structure_part_count": 1,
        })

        self.assertNotIn("truncated", compact)
        self.assertNotIn("requires_read", compact)
        self.assertEqual(compact["body"], "完整且明确的正文")
        self.assertNotIn("snippet", compact)

    def test_text_search_without_public_body_requires_read(self):
        compact = _Handler._clean_hit({
            "kind": "document",
            "data_id": "kb-test::documents/source.md",
            "source_file_name": "source.pdf",
            "chunk_index": 4,
            "body": "",
        })

        self.assertTrue(compact["requires_read"])
        self.assertNotIn("body", compact)
        self.assertNotIn("snippet", compact)

    def test_multi_part_structure_requires_read_without_partial_content(self):
        compact = _Handler._clean_hit({
            "kind": "document",
            "data_id": "kb-test::documents/source.md",
            "source_file_name": "source.pdf",
            "chunk_index": 4,
            "content_type": "table",
            "body": "| 指标 | 值 |",
            "structure_part_count": 3,
        })

        self.assertTrue(compact["requires_read"])
        self.assertNotIn("body", compact)
        self.assertNotIn("snippet", compact)
        self.assertNotIn("truncated", compact)

    def test_text_references_are_not_desktop_citations(self):
        handler = _Handler()

        handler._record_knowledge_citations({
            "kind": "document",
            "kb_id": "kb-test",
            "data_id": "kb-test::doc.md",
            "source_file_name": "source.pdf",
        })
        handler._record_knowledge_citations({
            "kind": "image",
            "kb_id": "kb-test",
            "data_id": "kb-test::doc.md::image::1",
            "image_id": "image-1",
            "source_file_name": "source.pdf",
        })

        citations = handler.take_knowledge_citations()
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["kind"], "image")
        self.assertEqual(citations[0]["image_id"], "image-1")

    def test_document_list_payload_hides_internal_path(self):
        result = _Handler._clean_document({
            "kind": "document",
            "kb_id": "kb-test",
            "data_id": "kb-test::documents/hash-doc.md",
            "file_name": "documents/hash-doc.md",
            "source_file_name": "original.pdf",
            "abspath": "C:/internal/processed/documents/hash-doc.md",
        })

        self.assertNotIn("abspath", result)
        self.assertEqual(result["data_id"], "kb-test::documents/hash-doc.md")
        self.assertNotIn("folder", result)
        self.assertNotIn("size", result)
        self.assertEqual(result["source_hint"], "《original.pdf》")

    def test_document_list_payload_is_compact(self):
        payload = json.loads(_Handler().do_kb_list({}, None).data)
        document = payload["documents"][0]

        self.assertEqual(
            set(document),
            {"data_id", "source_hint"},
        )
        self.assertNotIn('"file_name"', json.dumps(payload))

    def test_chunk_list_is_paginated_and_sanitized(self):
        payload = json.loads(_Handler().do_kb_list({
            "data_id": "kb-test::documents/source.md",
            "offset": 20,
            "limit": 10,
        }, None).data)

        self.assertNotIn("total", payload)
        self.assertEqual(payload["next_offset"], 30)
        self.assertEqual(len(payload["chunks"]), 10)
        self.assertNotIn("data_id", payload)
        self.assertNotIn("has_more", payload)
        self.assertNotIn("documents/image-20.jpg", json.dumps(payload))
        self.assertIn("[图片]", payload["chunks"][0]["preview"])

    def test_terminal_chunk_page_omits_pagination_state(self):
        payload = json.loads(_Handler().do_kb_list({
            "data_id": "kb-test::documents/source.md",
            "offset": 40,
            "limit": 10,
        }, None).data)

        self.assertEqual(len(payload["chunks"]), 5)
        self.assertNotIn("next_offset", payload)
        self.assertNotIn("has_more", payload)
        self.assertNotIn("data_id", payload)

    def test_image_ambiguity_returns_safe_candidates_for_retry(self):
        original = _Backend.read_image
        try:
            _Backend.read_image = staticmethod(lambda **_kwargs: {
                "error_code": "image_ambiguous",
                "error": r"C:\internal\active\processed\secret.md",
                "candidates": [
                    {
                        "data_id": "kb-test::documents/a.md::image::1",
                        "ref_key": "图1",
                        "display_label": "图1 架构",
                        "source_hint": "《a.pdf》：“图1 架构”",
                        "kb_id": "kb-test",
                        "file_name": "documents/a.md",
                    },
                    {
                        "data_id": "kb-test::documents/b.md::image::1",
                        "ref_key": "图1",
                        "display_label": "图1 流程",
                        "source_hint": "《b.pdf》：“图1 流程”",
                        "image_abspath": r"C:\internal\b.png",
                    },
                ],
            })
            outcome = _Handler().do_kb_image_read({"ref_key": "图1"}, None)
        finally:
            _Backend.read_image = original

        payload = json.loads(outcome.data)
        self.assertEqual(payload["error_code"], "image_ambiguous")
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertEqual(
            set(payload["candidates"][0]),
            {"data_id", "source_hint"},
        )
        self.assertNotIn("internal", outcome.data)

    def test_image_read_rejects_document_id_before_backend_lookup(self):
        original = _Backend.read_image
        calls = []
        try:
            _Backend.read_image = staticmethod(lambda **kwargs: calls.append(kwargs))
            payload = json.loads(_Handler().do_kb_image_read({
                "data_id": "kb-test::documents/source.md",
                "ref_key": "图1",
            }, None).data)
        finally:
            _Backend.read_image = original

        self.assertEqual(payload["error_code"], "invalid_argument")
        self.assertEqual(payload["field"], "data_id")
        self.assertEqual(calls, [])

    def test_image_read_reports_conflicting_id_and_reference(self):
        original = _Backend.read_image
        try:
            _Backend.read_image = staticmethod(lambda **_kwargs: {
                "error_code": "image_target_conflict",
                "error": r"C:\internal\must-not-leak",
            })
            payload = json.loads(_Handler().do_kb_image_read({
                "data_id": "kb-test::documents/source.md::image::1",
                "ref_key": "图2",
            }, None).data)
        finally:
            _Backend.read_image = original

        self.assertEqual(payload["error_code"], "target_conflict")
        self.assertEqual(payload["field"], "ref_key")
        self.assertNotIn("internal", json.dumps(payload))

    def test_read_removes_processing_header_and_image_path(self):
        original = _Backend.read_content
        try:
            _Backend.read_content = staticmethod(
                lambda **_kwargs: {
                    "data_id": "kb-test::documents/source.md",
                    "content_type": "prose",
                    "start_chunk_index": 0,
                    "end_chunk_index": 0,
                    "content": "正文 ![](documents/internal.jpg)",
                    "continuation": {"has_more": False},
                }
            )
            outcome = _Handler().do_kb_read({
                "data_id": "kb-test::documents/source.md",
                "chunk_index": 0,
            }, None)
        finally:
            _Backend.read_content = original

        self.assertNotIn("原始文档：", outcome.data)
        self.assertNotIn("documents/internal.jpg", outcome.data)
        self.assertIn("[图片]", outcome.data)
        self.assertEqual(set(json.loads(outcome.data)), {
            "absence_supported", "source_hint", "content",
        })

    def test_read_returns_only_actionable_continuation_fields(self):
        original = _Backend.read_content
        try:
            _Backend.read_content = staticmethod(
                lambda **_kwargs: {
                    "content_type": "table",
                    "content": "| 指标 | 值 |",
                    "continuation": {
                        "has_more": True,
                        "next_chunk_index": 8,
                        "same_structure": True,
                    },
                }
            )
            payload = json.loads(_Handler().do_kb_read({
                "data_id": "kb-test::documents/source.md",
                "chunk_index": 7,
            }, None).data)
        finally:
            _Backend.read_content = original

        self.assertFalse(payload["absence_supported"])
        self.assertEqual(payload["continuation"], {"next_chunk_index": 8})

    def test_truncated_mineru_image_link_is_hidden(self):
        cleaned = _Handler._clip_text(
            "章节路径：/内部章节/ ![](documents/processed-image.jpg",
            200,
        )

        self.assertNotIn("documents/processed-image.jpg", cleaned)
        self.assertNotIn("章节路径：", cleaned)

    def test_flattened_chunk_context_does_not_remove_preview_content(self):
        cleaned = _Handler._clip_text(
            "章节路径：/公司/财务数据/ 资产负债率为 22.3%",
            200,
        )

        self.assertEqual(cleaned, "资产负债率为 22.3%")

    def test_image_read_public_payload_is_raw_image_first(self):
        result = _Handler._public_image({
            "kind": "image",
            "kb_id": "kb-test",
            "data_id": "kb-test::documents/source.md::image::1",
            "image_id": "image-1",
            "ref_key": "图1",
            "source_file_name": "source.pdf",
            "title": "图1：标题",
            "caption": "图1：标题",
            "display_label": "图1：标题",
            "description": "图表描述",
            "table_markdown": "",
            "analysis_error": "",
            "near_text": "hash-doc.assets-abcd/secret.jpg) 图片附近正文",
            "related_text": " ".join(f"图{i}：目录项" for i in range(1, 20)),
        })

        self.assertEqual(result["source_hint"], "《source.pdf》：“图1：标题”")
        self.assertEqual(set(result), {"source_hint", "ref_key"})

    def test_image_search_without_analysis_is_locator_not_visual_description(self):
        result = _Handler._clean_hit({
            "kind": "image",
            "kb_id": "kb-test",
            "data_id": "kb-test::documents/source.md::image::1",
            "source_file_name": "source.pdf",
            "ref_key": "图1",
            "caption": "图1：正确图题",
            "display_label": "图1：正确图题",
            "description": "",
            "uncertain": ["不应作为已有视觉分析返回"],
            "snippet": "图1到图99的目录式长文本，不应作为图片描述",
            "matched_by": ["ref_exact"],
        })

        self.assertTrue(result["locator_only"])
        self.assertNotIn("description", result)
        self.assertNotIn("uncertain", result)
        self.assertEqual(result["source_hint"], "《source.pdf》：“图1：正确图题”")

    def test_image_search_with_analysis_is_not_locator_only(self):
        result = _Handler._clean_hit({
            "kind": "image",
            "data_id": "kb-test::documents/source.md::image::1",
            "source_file_name": "source.pdf",
            "ref_key": "图1",
            "description": "原图分析得到的流程说明",
            "uncertain": ["小字可能不清晰"],
        })

        self.assertEqual(result["description"], "原图分析得到的流程说明")
        self.assertEqual(result["uncertain"], ["小字可能不清晰"])
        self.assertNotIn("locator_only", result)

    def test_image_source_hint_uses_original_document_and_figure_label(self):
        hint = _Handler._source_hint({
            "kind": "image",
            "source_file_name": "original.pdf",
            "display_label": "图1 技能更新流程",
        })

        self.assertEqual(hint, "《original.pdf》：“图1 技能更新流程”")

    def test_tool_errors_do_not_expose_backend_details(self):
        handler = _FailingHandler()
        outputs = [
            handler.do_kb_search({"query": "test", "mode": "vector"}, None).data,
            handler.do_kb_find({"terms": ["test"]}, None).data,
            handler.do_kb_read({"data_id": "kb-test::documents/source.md"}, None).data,
            handler.do_kb_list({"data_id": "kb-test::documents/source.md"}, None).data,
            handler.do_kb_image_read({"data_id": "kb-test::documents/source.md::image::1"}, None).data,
        ]

        for output in outputs:
            self.assertNotIn("secret", output)
            self.assertNotIn("zvec", output)
            self.assertIn("error_code", json.loads(output))

    def test_image_focus_is_optional_and_redacted(self):
        schema = next(
            item["function"]
            for item in KB_TOOL_SCHEMAS
            if item["function"]["name"] == "kb_image_read"
        )
        self.assertNotIn("focus", schema["parameters"].get("required", []))

        cleaned = _Handler._clean_image_focus(
            r"确认 C:\\private\\processed\\image.jpg 中的结构 " + "x" * 600
        )
        self.assertLessEqual(len(cleaned), 500)
        self.assertNotIn("C:\\private\\processed", cleaned)

    def test_read_exposes_safe_source_hint_and_answer_rule(self):
        outcome = _Handler().do_kb_read(
            {
                "data_id": "kb-test::documents/hash-doc.md",
                "chunk_index": 0,
                "span": 1,
            },
            None,
        )

        self.assertIn("《source.pdf》：“方法”", outcome.data)
        self.assertNotIn("禁止出现在面向用户的回答中", outcome.next_prompt)

    def test_read_requires_explicit_chunk_index(self):
        outcome = _Handler().do_kb_read(
            {"data_id": "kb-test::documents/hash-doc.md"}, None,
        )

        payload = json.loads(outcome.data)
        self.assertEqual(payload["error_code"], "invalid_argument")
        self.assertEqual(payload["field"], "chunk_index")

    def test_agent_usage_policy_is_one_system_prompt_contract(self):
        self.assertIn("[KNOWLEDGE_BASE_USAGE]", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("信息来源", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("改写、拆分查询或显式更换检索模式", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("absence_supported=false", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("absence_supported=true", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("至少出现一次", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("不能据此推断次数、语境或重要性", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("locator_only=true", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("missing_image_refs不得用相似图片冒充", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("ambiguous_image_refs必须先消歧", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("requires_read=true", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("直接证据与推断必须分开表述", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("同一模型回合只读取一张", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("工具控制字段只用于下一步决策", KB_AGENT_SYSTEM_INSTRUCTIONS)
        self.assertIn("来源名称只能逐字使用工具返回的 source_hint", KB_AGENT_SYSTEM_INSTRUCTIONS)
        for implementation_detail in ("MinerU", "OCR", "VLM", "分块", "索引"):
            self.assertNotIn(implementation_detail, KB_AGENT_SYSTEM_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
