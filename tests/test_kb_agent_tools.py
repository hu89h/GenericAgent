import unittest

from knowledge_base.agent_tools import KB_TOOL_SCHEMAS


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


if __name__ == "__main__":
    unittest.main()
