import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from knowledge_base import importer
from knowledge_base.assets import ImageAssetProcessor


class DocumentProcessorTests(unittest.TestCase):
    def test_mineru_captured_concatenated_image_link_is_split_before_copying(self):
        fixture = Path(__file__).parent / "fixtures" / "mineru" / "concatenated_image_links"
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            shutil.copytree(fixture, source, ignore=shutil.ignore_patterns("README.md"))
            for index, image in enumerate(sorted(source.rglob("*.jpg"))):
                Image.new("RGB", (2, 2), (index * 40, 20, 180)).save(image, format="JPEG")

            result = importer.DocumentProcessor().prepare(
                str(source), stage_root=str(stage), kb_id="kb-test"
            )

            self.assertEqual(result["summary"]["ready"], 1)
            self.assertEqual(result["summary"]["assets"], 1)
            processed = next((stage / "processed").rglob("*.md"))
            body = processed.read_text(encoding="utf-8")
            self.assertEqual(body.count("![]("), 2)
            self.assertNotIn(".jpgAC", body)
            self.assertEqual(
                len(list(processed.parent.glob("*.assets-*/*.jpg"))), 1
            )
            image_index = ImageAssetProcessor(usage_tracker=None).build_document_index(body)
            self.assertEqual(len(image_index.occurrences), 2)

    def test_valid_local_filename_and_remote_url_are_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            source.mkdir()
            assets = source / "assets"
            assets.mkdir()
            image = assets / "one.jpg"
            image.write_bytes(b"image")
            compound = assets / "one.jpgb.jpg"
            compound.write_bytes(b"compound")
            (source / "doc.md").write_text(
                "![](assets/one.jpgb.jpg)\n"
                "![](https://example.test/a.jpgb.jpg)",
                encoding="utf-8",
            )

            result = importer.DocumentProcessor().prepare(
                str(source), stage_root=str(stage), kb_id="kb-test"
            )

            processed = next((stage / "processed").rglob("*.md"))
            body = processed.read_text(encoding="utf-8")
            self.assertEqual(result["summary"]["assets"], 1)
            self.assertIn("https://example.test/a.jpgb.jpg", body)
            self.assertEqual(len(list(processed.parent.glob("*.assets-*/*"))), 1)

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
