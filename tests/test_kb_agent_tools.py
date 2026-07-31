import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from knowledge_base.agent_tools import (
    KB_RESPONSE_SOURCE_INSTRUCTIONS,
    KB_TOOL_SCHEMAS,
    KnowledgeBaseToolsMixin,
)


class _Backend:
    image_abspath = ""

    @staticmethod
    def search(query, **_kwargs):
        return {"mode": "vector", "results": []}

    @staticmethod
    def read_chunk(**_kwargs):
        return "# 原始文档：source.pdf\n章节：方法\n正文"

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
            "description": "diagram",
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
        return {"attach_status": "attached"}, None


class _ScopedImageHandler(_Handler):
    def __init__(self, scope):
        self.parent = SimpleNamespace(knowledge_scope=scope)

    @staticmethod
    def _kb_backend():
        return _ScopedImageBackend


class KnowledgeBaseAgentSchemaTests(unittest.TestCase):
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
                    {"data_id": "kb-test::doc.md::image::1"}, None,
                )

                self.assertEqual(handler.queued_image, str(image))
                self.assertIn('"attach_status": "attached"', outcome.data)
            finally:
                _Backend.image_abspath = ""

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
                    "ref": "kb-test/doc.md",
                })
                outcome = handler.do_kb_image_read({"ref_key": "图1"}, None)

                self.assertIn('"attach_status": "attached"', outcome.data)
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
        self.assertIn('"mode": "vector"', outcome.data)

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
        self.assertEqual(result["file_name"], "original.pdf")
        self.assertEqual(result["source_hint"], "《original.pdf》")

    def test_image_source_hint_uses_original_document_and_figure_label(self):
        hint = _Handler._source_hint({
            "kind": "image",
            "source_file_name": "original.pdf",
            "display_label": "图1 技能更新流程",
        })

        self.assertEqual(hint, "《original.pdf》：“图1 技能更新流程”")

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
        self.assertIn(KB_RESPONSE_SOURCE_INSTRUCTIONS, outcome.next_prompt)
        self.assertIn("禁止出现在面向用户的回答中", outcome.next_prompt)


if __name__ == "__main__":
    unittest.main()
