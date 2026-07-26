import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_base.assets import ImageAssetProcessor, ImageContent
from knowledge_base.cancellation import KnowledgeBaseCancelled
from knowledge_base.usage import UsageTracker


class ImageProgressTests(unittest.TestCase):
    def test_progress_is_grouped_by_original_document(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=1)
        processor._image_client = SimpleNamespace(enabled=lambda: True)
        processor.analyze_image_job = lambda _path, _job: (
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

    def test_cancellation_stops_before_submitting_more_sequential_images(self):
        processor = ImageAssetProcessor(usage_tracker=UsageTracker(), concurrency=1)
        processor._image_client = SimpleNamespace(enabled=lambda: True)
        cancel_event = threading.Event()
        calls = []

        def analyze(_path, job):
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


if __name__ == "__main__":
    unittest.main()
