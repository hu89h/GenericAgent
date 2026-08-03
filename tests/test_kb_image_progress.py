import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_base.assets import ImageAssetProcessor, ImageContent
from knowledge_base.cancellation import KnowledgeBaseCancelled
from knowledge_base.usage import UsageTracker


class ImageProgressTests(unittest.TestCase):
    def test_image_analysis_uses_durable_cache_path_when_provided(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=1)
        processor._image_client = SimpleNamespace(enabled=lambda: True)
        paths = []
        processor.analyze_image_job = lambda path, _job, **_kwargs: (
            paths.append(path) or ({"description": "ok"}, {"calls": 1})
        )
        job = ImageContent(
            image_sha="one",
            image_path="assets/one.png",
            image_abspath="unused",
            focus="general",
            title="",
            near_text="",
            ref_candidates=[],
            analysis_meta={},
            origins=[{"key": "documents/a.md", "name": "alpha.pdf"}],
        )

        processor.analyze_image_jobs(
            {"path": "disposable-stage", "image_cache_path": "durable-cache"},
            {"one": job},
            lambda _message: None,
        )

        self.assertEqual(paths, ["durable-cache"])

    def test_progress_is_grouped_by_original_document(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=1)
        processor._image_client = SimpleNamespace(enabled=lambda: True)
        processor.analyze_image_job = lambda _path, _job, **_kwargs: (
            {"description": "ok"},
            {"calls": 1},
        )
        shared = ImageContent(
            image_sha="shared",
            image_path="assets/internal-shared.png",
            image_abspath="unused",
            focus="general",
            title="",
            near_text="",
            ref_candidates=[],
            analysis_meta={},
            origins=[
                {"key": "documents/a.md", "name": "alpha.pdf"},
                {"key": "documents/b.md", "name": "beta.pdf"},
            ],
        )
        alpha_only = ImageContent(
            image_sha="alpha-only",
            image_path="assets/internal-alpha.png",
            image_abspath="unused",
            focus="general",
            title="",
            near_text="",
            ref_candidates=[],
            analysis_meta={},
            origins=[{"key": "documents/a.md", "name": "alpha.pdf"}],
        )
        events = []

        with mock.patch.dict(os.environ, {"GA_KB_IMAGE_CONCURRENCY": "1"}):
            processor.analyze_image_jobs(
                {"path": "unused"},
                {"shared": shared, "alpha-only": alpha_only},
                lambda _message: None,
                progress=events.append,
            )

        final = events[-1]
        self.assertEqual(final["analysis_completed"], 2)
        self.assertNotIn("internal-alpha.png", final["current"])
        self.assertEqual(final["image_documents"], [
            {"key": "documents/a.md", "name": "alpha.pdf", "completed": 2, "total": 2},
            {"key": "documents/b.md", "name": "beta.pdf", "completed": 1, "total": 1},
        ])
        self.assertEqual(final["image_activity"]["total"], 2)
        self.assertEqual(final["image_activity"]["completed"], 2)
        self.assertEqual(final["image_activity"]["cached"], 0)

    def test_cancellation_stops_before_submitting_more_sequential_images(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=1)
        processor._image_client = SimpleNamespace(enabled=lambda: True)
        cancel_event = threading.Event()
        calls = []

        def analyze(_path, job, **_kwargs):
            calls.append(job.image_sha)
            cancel_event.set()
            return {"description": "unused"}, {"calls": 1}

        processor.analyze_image_job = analyze
        jobs = {
            key: ImageContent(
                image_sha=key,
                image_path=f"assets/{key}.png",
                image_abspath="unused",
                focus="general",
                title="",
                near_text="",
                ref_candidates=[],
                analysis_meta={},
                origins=[{"key": "documents/a.md", "name": "alpha.pdf"}],
            )
            for key in ("one", "two")
        }

        with mock.patch.dict(os.environ, {"GA_KB_IMAGE_CONCURRENCY": "1"}):
            with self.assertRaises(KnowledgeBaseCancelled):
                processor.analyze_image_jobs(
                    {"path": "unused"},
                    jobs,
                    lambda _message: None,
                    cancelled=cancel_event.is_set,
                )

        self.assertEqual(calls, ["one"])

    def test_explicit_vision_rejection_skips_remaining_images(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=64)
        processor._image_client = SimpleNamespace(enabled=lambda: True)
        calls = []

        def analyze(_path, job, **_kwargs):
            calls.append(job.image_sha)
            if len(calls) == 1:
                return {"error": "HTTP 400: model does not support image input"}, {"calls": 1}
            raise AssertionError("a rejected vision model must not receive more image requests")

        processor.analyze_image_job = analyze
        jobs = {
            key: ImageContent(
                image_sha=key,
                image_path=f"assets/{key}.png",
                image_abspath="unused",
                focus="general",
                title="",
                near_text="",
                ref_candidates=[],
                analysis_meta={},
                origins=[{"key": "documents/a.md", "name": "alpha.pdf"}],
            )
            for key in ("one", "two", "three")
        }

        results = processor.analyze_image_jobs(
            {"path": "unused"},
            jobs,
            lambda _message: None,
        )

        self.assertEqual(calls, ["one"])
        self.assertEqual(set(results), set(jobs))
        self.assertTrue(all(item.get("vision_skipped") for item in results.values()))
        self.assertIn("不支持图片输入", results["one"]["analysis_warning"])

    def test_vision_skipped_analysis_keeps_basic_image_record(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=1)
        asset = {
            "kind": "image",
            "ref_key": "图1",
            "caption": "图1 示例",
            "title": "图1 示例",
            "section": "章节",
            "near_text": "正文上下文",
            "related_text": "正文引用",
        }

        processor.apply_image_analysis(asset, {
            "vision_skipped": True,
            "analysis_warning": "当前模型不支持图片输入",
            "uncertain": ["当前模型不支持图片输入"],
        })

        self.assertFalse(asset["analysis_error"])
        self.assertEqual(asset["analysis_warning"], "当前模型不支持图片输入")
        self.assertIn("图1", asset["body"])


if __name__ == "__main__":
    unittest.main()
