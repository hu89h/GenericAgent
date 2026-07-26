import unittest

import agentmain
from frontends.desktop_bridge import normalize_knowledge_scope
from knowledge_base.agent_tools import KnowledgeBaseToolsMixin


class _ScopeHandler(KnowledgeBaseToolsMixin):
    def __init__(self, scope):
        if isinstance(scope, str):
            scope = {"mode": scope}
        self.parent = type("Parent", (), {"knowledge_scope": scope})()

    @staticmethod
    def _get_anchor_prompt(skip=False):
        return "\n"


class KnowledgeScopeTests(unittest.TestCase):
    def test_desktop_normalizes_disabled_scope(self):
        self.assertEqual(
            normalize_knowledge_scope({"mode": "none", "origin": "chat"}),
            {"mode": "none", "origin": "chat"},
        )

    def test_disabled_scope_removes_all_kb_tools(self):
        schema = [
            {"function": {"name": "file_read"}},
            *agentmain.KB_TOOL_SCHEMAS,
        ]
        filtered = agentmain.tool_schema_for_scope(schema, {"mode": "none"})
        names = {tool["function"]["name"] for tool in filtered}

        self.assertEqual(names, {"file_read"})

    def test_scope_prompt_uses_public_labels_only(self):
        prompt = agentmain.knowledge_scope_prompt({
            "mode": "document",
            "kb_id": "internal-kb-id",
            "data_id": "internal::data-id",
            "kb_name": "论文",
            "title": "研究报告.pdf",
        })

        self.assertIn("论文", prompt)
        self.assertIn("研究报告.pdf", prompt)
        self.assertNotIn("internal-kb-id", prompt)
        self.assertNotIn("internal::data-id", prompt)

    def test_enabled_scope_keeps_kb_tools(self):
        schema = list(agentmain.KB_TOOL_SCHEMAS)
        self.assertIs(agentmain.tool_schema_for_scope(schema, {"mode": "all"}), schema)

    def test_disabled_scope_rejects_direct_tool_call(self):
        outcome = _ScopeHandler("none").do_kb_search(
            {"query": "anything", "mode": "rrf"}, None,
        )

        self.assertIn("未启用知识库", outcome.data)

    def test_kb_scope_allows_only_documents_in_selected_kb(self):
        handler = _ScopeHandler({"mode": "kb", "kb_id": "kb-a"})

        self.assertTrue(handler._scope_allows_target(data_id="kb-a::documents/one.md"))
        self.assertTrue(handler._scope_allows_target(data_id="kb-a::documents/two.md"))
        self.assertFalse(handler._scope_allows_target(data_id="kb-b::documents/one.md"))

    def test_document_scope_allows_only_selected_document(self):
        handler = _ScopeHandler({
            "mode": "document",
            "kb_id": "kb-a",
            "data_id": "kb-a::documents/one.md",
        })

        self.assertEqual(
            handler._scope_search_kwargs(),
            {"kb_id": "kb-a", "file_name": "documents/one.md"},
        )
        self.assertTrue(handler._scope_allows_target(data_id="kb-a::documents/one.md"))
        self.assertFalse(handler._scope_allows_target(data_id="kb-a::documents/two.md"))
        self.assertFalse(handler._scope_allows_target(data_id="kb-b::documents/one.md"))


if __name__ == "__main__":
    unittest.main()
