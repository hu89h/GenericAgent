import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledge_base import importer


class DocumentProcessorTests(unittest.TestCase):
    def test_markdown_failures_do_not_discard_successful_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            source.mkdir()
            (source / "good.md").write_text("# Good\n\nUseful content.", encoding="utf-8")
            (source / "bad.md").write_text("# Bad\n\nBroken content.", encoding="utf-8")
            original = importer._write_markdown

            def selective_failure(path, *args, **kwargs):
                if Path(path).name == "bad.md":
                    raise OSError("injected markdown failure")
                return original(path, *args, **kwargs)

            events = []
            with mock.patch.object(importer, "_write_markdown", side_effect=selective_failure):
                result = importer.DocumentProcessor().prepare(
                    str(source),
                    stage_root=str(stage),
                    kb_id="kb-test",
                    progress=events.append,
                )

            self.assertEqual(result["summary"]["ready"], 1)
            self.assertEqual(result["summary"]["failed"], 1)
            self.assertEqual(result["failures"][0]["stage"], "markdown")
            self.assertEqual(events[-1]["phase"], "prepared")
            self.assertEqual(events[-1]["completed"], 2)
            self.assertEqual(events[-1]["total"], 2)
            self.assertEqual(events[-1]["document_progress"], {
                "completed": 2,
                "total": 2,
                "failed": 1,
                "ready": 1,
            })
            self.assertTrue((stage / "manifest.json").is_file())
            self.assertTrue(any((stage / "processed").rglob("*.md")))


if __name__ == "__main__":
    unittest.main()
