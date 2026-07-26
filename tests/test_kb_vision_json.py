import unittest

from knowledge_base.providers import vision


class VisionJsonParsingTests(unittest.TestCase):
    def test_repairs_only_unsupported_json_backslash_escapes(self):
        parsed = vision._extract_json(
            '{"description":"$S_t \\checkmark$","table_markdown":"",'
            '"ref_key":"图3","uncertain":[]}'
        )

        self.assertNotIn("error", parsed)
        self.assertEqual(parsed["description"], "$S_t \\checkmark$")
        self.assertEqual(parsed["ref_key"], "图3")

    def test_structurally_invalid_json_remains_a_failure(self):
        parsed = vision._extract_json('{"description":"unfinished"')

        self.assertEqual(parsed["error"], "model did not return valid JSON")


if __name__ == "__main__":
    unittest.main()
