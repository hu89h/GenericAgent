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


class ImageCaptionBindingTests(unittest.TestCase):
    def setUp(self):
        self.assets = ImageAssetProcessor(usage_tracker=None)

    class _DisabledVision:
        @staticmethod
        def analysis_meta():
            return {"prompt_version": 1, "preprocess_version": 1, "model": "test"}

        @staticmethod
        def enabled():
            return False

        @staticmethod
        def understanding_focus(*_args, **_kwargs):
            return "general"

    def test_explicit_caption_can_precede_or_follow_its_adjacent_image(self):
        index = self.assets.build_document_index(
            "图 1：前置图题\n"
            "![](one.png)\n"
            "资料来源：示例\n\n"
            "![](two.png)\n"
            "图 2：后置图题\n"
        )

        self.assertEqual(
            [(item.path, item.ref_key, item.title) for item in index.occurrences],
            [
                ("one.png", "图1", "图 1：前置图题"),
                ("two.png", "图2", "图 2：后置图题"),
            ],
        )

    def test_image_does_not_borrow_a_non_adjacent_or_ambiguous_caption(self):
        index = self.assets.build_document_index(
            "图 7 所示趋势来自其他段落，这不是图题。\n"
            "![](unlabelled.png)\n"
            "资料来源：示例\n\n"
            "图 8：下一张图片\n"
            "![](next.png)\n"
            "图 9：另一侧候选\n"
        )

        self.assertEqual(index.occurrences[0].ref_key, "")
        self.assertEqual(index.occurrences[0].title, "image")
        self.assertEqual(index.occurrences[1].ref_key, "")

    def test_context_never_contains_generated_asset_paths(self):
        index = self.assets.build_document_index(
            "图 1：示例\n"
            "![](hash-document.assets-abcd/first.jpg)\n"
            "资料来源：示例\n\n"
            "图 2：下一张\n"
            "![](hash-document.assets-abcd/second.jpg)\n"
        )

        context = index.occurrences[0].near_text
        self.assertNotIn(".assets-", context)
        self.assertNotIn("first.jpg", context)
        self.assertNotIn("second.jpg", context)
        self.assertIn("[图片:image]", context)

    def test_figure_catalogue_is_not_attached_as_related_image_evidence(self):
        catalogue = " ".join(f"图{i}：目录项" for i in range(1, 12))
        index = self.assets.build_document_index(
            f"{catalogue}\n\n"
            "图 1：正文中的图片\n"
            "![](one.png)\n"
        )

        self.assertEqual(index.occurrences[0].related_text, "")

    def test_vlm_cannot_create_an_exact_figure_reference(self):
        asset = {
            "ref_key": "",
            "caption": "",
            "title": "image",
            "section": "",
            "description": "",
            "table_markdown": "",
            "near_text": "",
            "related_text": "",
        }

        self.assets.apply_image_analysis(
            asset,
            {
                "description": "图片内容",
                "table_markdown": "",
                "ref_key": "图 99",
                "uncertain": [],
            },
        )

        self.assertEqual(asset["ref_key"], "")

    def test_rebuild_reuses_analysis_only_when_caption_binding_is_unchanged(self):
        self.assets._image_client = self._DisabledVision()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "figure.jpg").write_bytes(b"stable image bytes")
            kb = {"id": "kb-test", "path": str(root)}

            first_body = "图 1：原图题\n![](figure.jpg)\n"
            first_index = self.assets.build_document_index(first_body)
            first = self.assets.image_records_for_document(
                kb, "doc.md", "kb-test::doc.md", first_body, "doc.pdf", lambda _msg: None,
                image_index=first_index,
            )["assets"][0]
            cached = {**first, "description": "已缓存的图片描述"}

            unchanged = self.assets.image_records_for_document(
                kb, "doc.md", "kb-test::doc.md", first_body, "doc.pdf", lambda _msg: None,
                image_index=self.assets.build_document_index(first_body),
                existing_images={first["data_id"]: cached},
                preserve_analysis_only=True,
            )["assets"][0]
            self.assertEqual(unchanged["description"], "已缓存的图片描述")

            changed_body = "图 2：修正后的图题\n![](figure.jpg)\n"
            changed = self.assets.image_records_for_document(
                kb, "doc.md", "kb-test::doc.md", changed_body, "doc.pdf", lambda _msg: None,
                image_index=self.assets.build_document_index(changed_body),
                existing_images={first["data_id"]: cached},
                preserve_analysis_only=True,
            )["assets"][0]
            self.assertEqual(changed["ref_key"], "图2")
            self.assertEqual(changed["description"], "")


if __name__ == "__main__":
    unittest.main()
