import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from knowledge_base import importer
from knowledge_base.assets import ImageAssetProcessor


class DocumentProcessorTests(unittest.TestCase):
    def test_resume_reprocesses_same_size_file_when_content_hash_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            source.mkdir()
            document = source / "document.md"
            document.write_text("# First\nbody", encoding="utf-8")
            processor = importer.DocumentProcessor()
            first = processor.prepare(
                str(source), stage_root=str(stage), kb_id="kb-test"
            )
            original_stat = document.stat()
            document.write_text("# First\ntext", encoding="utf-8")
            os.utime(document, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with mock.patch.object(importer, "_write_markdown", wraps=importer._write_markdown) as write_markdown:
                resumed = processor.prepare(
                    str(source),
                    stage_root=str(stage),
                    kb_id="kb-test",
                    resume_manifest=first["manifest"],
                )
            self.assertEqual(resumed["summary"]["ready"], 1)
            write_markdown.assert_called_once()
            fingerprint = resumed["manifest"]["source_fingerprint"][0]
            self.assertEqual(len(fingerprint["sha256"]), 64)

    def test_cancel_with_retention_writes_checkpoint_and_resume_reuses_ready_document(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            source.mkdir()
            (source / "first.md").write_text("# First\n", encoding="utf-8")
            (source / "second.md").write_text("# Second\n", encoding="utf-8")
            cancelled = threading.Event()
            original = importer._write_markdown
            calls = []

            def stop_after_first(*args, **kwargs):
                calls.append(Path(args[0]).name)
                result = original(*args, **kwargs)
                if len(calls) == 1:
                    cancelled.set()
                return result

            with mock.patch.object(importer, "_write_markdown", side_effect=stop_after_first):
                with self.assertRaises(importer.KnowledgeBaseCancelled):
                    importer.DocumentProcessor().prepare(
                        str(source),
                        stage_root=str(stage),
                        kb_id="kb-test",
                        cancelled=cancelled.is_set,
                        retain_on_cancel=lambda: True,
                    )

            checkpoint = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["state"], "checkpoint")
            self.assertEqual(checkpoint["checkpoint"]["ready_documents"], 1)
            self.assertTrue((stage / "processed").is_dir())

            calls.clear()
            with mock.patch.object(importer, "_write_markdown", side_effect=stop_after_first):
                resumed = importer.DocumentProcessor().prepare(
                    str(source),
                    stage_root=str(stage),
                    kb_id="kb-test",
                    resume_manifest=checkpoint,
                )
            self.assertEqual(resumed["summary"]["ready"], 2)
            self.assertEqual(calls, ["second.md"])

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

    def test_selected_file_import_keeps_only_selected_document_and_original_path(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            source.mkdir()
            image = source / "figure.jpg"
            Image.new("RGB", (2, 2), (40, 80, 120)).save(image, format="JPEG")
            document = source / "selected.md"
            document.write_text("# Selected\n\n![图1](figure.jpg)\n", encoding="utf-8")
            (source / "not-selected.md").write_text("# Ignore", encoding="utf-8")

            result = importer.DocumentProcessor().prepare_files(
                [str(document)], stage_root=str(stage), kb_id="kb-test"
            )

            self.assertEqual(result["summary"]["ready"], 1)
            entries = [
                item for item in result["manifest"]["files"]
                if item.get("kind") == "document"
            ]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["name"], "selected.md")
            self.assertEqual(Path(entries[0]["source_path"]).resolve(), document.resolve())
            self.assertEqual(
                {Path(item["path"]).name for item in result["manifest"]["source_fingerprint"]},
                {"selected.md", "figure.jpg"},
            )
            self.assertFalse((stage / ".selected_sources").exists())
            self.assertFalse(any("not-selected" in path.name for path in (stage / "processed").rglob("*.md")))
            processed = next((stage / "processed").rglob("*.md"))
            self.assertEqual(len(list(processed.parent.glob("*.assets-*/*.jpg"))), 1)

    def test_multiple_selected_files_keep_a_stable_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            stage = Path(temp) / "stage"
            source.mkdir()
            first = source / "first.md"
            second = source / "second.md"
            first.write_text("# First\n", encoding="utf-8")
            second.write_text("# Second\n", encoding="utf-8")

            result = importer.DocumentProcessor().prepare_files(
                [str(second), str(first), str(second)],
                stage_root=str(stage),
                kb_id="kb-test",
            )

            entries = [
                item for item in result["manifest"]["files"]
                if item.get("kind") == "document"
            ]
            self.assertEqual({item["name"] for item in entries}, {"first.md", "second.md"})
            fingerprints = result["manifest"]["source_fingerprint"]
            self.assertEqual(
                [item["path"] for item in fingerprints],
                sorted([str(first.resolve()), str(second.resolve())], key=str.casefold),
            )


if __name__ == "__main__":
    unittest.main()
