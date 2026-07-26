import unittest
from types import SimpleNamespace

from agent_loop import exhaust
from knowledge_base.agent_tools import (
    KB_RESPONSE_SOURCE_INSTRUCTIONS,
    KB_TOOL_SCHEMAS,
    KnowledgeBaseToolsMixin,
)


class _Backend:
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


class _Handler(KnowledgeBaseToolsMixin):
    def __init__(self):
        self.parent = SimpleNamespace(knowledge_scope={"mode": "all"})

    @staticmethod
    def _kb_backend():
        return _Backend

    @staticmethod
    def _get_anchor_prompt(skip=False):
        return "\n" if skip else "\nanchor"


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
        outcome = exhaust(_Handler().do_kb_read(
            {
                "data_id": "kb-test::documents/hash-doc.md",
                "chunk_index": 0,
                "span": 1,
            },
            None,
        ))

        self.assertIn("《source.pdf》：“方法”", outcome.data)
        self.assertIn(KB_RESPONSE_SOURCE_INSTRUCTIONS, outcome.next_prompt)
        self.assertIn("禁止出现在面向用户的回答中", outcome.next_prompt)


if __name__ == "__main__":
    unittest.main()
