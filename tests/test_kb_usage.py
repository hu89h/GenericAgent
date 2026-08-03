import threading
import unittest
from unittest import mock

from knowledge_base.build import IndexBuilder
from knowledge_base.usage import UsageTracker


class UsageTrackerTests(unittest.TestCase):
    def test_build_usage_snapshots_the_models_used_by_maintenance(self):
        tracker = UsageTracker()
        builder = IndexBuilder(index=object(), usage_tracker=tracker)
        with mock.patch(
            "knowledge_base.build.provider_settings.vision_config",
            return_value={"model": "vision-model"},
        ), mock.patch(
            "knowledge_base.build.provider_settings.embedding_config",
            return_value={"model": "embedding-model"},
        ):
            builder.begin_build()

        summary = tracker.summary(tracker.current())
        self.assertEqual(summary["image_model"], "vision-model")
        self.assertEqual(summary["embedding_model"], "embedding-model")

    def test_image_usage_from_worker_threads_merges_into_current_build(self):
        tracker = UsageTracker()
        tracker.set_current(tracker.empty())
        threads = [
            threading.Thread(
                target=tracker.merge_image_analysis,
                args=({"calls": 1, "prompt_tokens": 10},),
            )
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        usage = tracker.current()["image_analysis"]
        self.assertEqual(usage["calls"], 3)
        self.assertEqual(usage["prompt_tokens"], 30)

    def test_summary_marks_missing_provider_tokens_as_unknown(self):
        tracker = UsageTracker()
        usage = tracker.empty()
        usage["image_analysis"].update(calls=1, prompt_tokens=0)
        usage["embedding"].update(calls=1, texts=4, api_tokens=0, api_calls=1)

        summary = tracker.summary(usage)

        self.assertTrue(summary["available"])
        self.assertIsNone(summary["image_prompt_tokens"])
        self.assertFalse(summary["image_token_usage_reported"])
        self.assertIsNone(summary["embedding_api_tokens"])
        self.assertFalse(summary["embedding_token_usage_reported"])
        self.assertIsNone(summary["embedding_input_tokens"])
        self.assertIsNone(summary["embedding_output_tokens"])

    def test_summary_combines_embedding_implementations_for_display(self):
        tracker = UsageTracker()
        usage = tracker.empty()
        usage["embedding"].update(
            calls=2, texts=8, api_tokens=100, api_calls=2,
            cache_hits=3, token_usage_reported=True,
            input_tokens=100, input_token_usage_reported=True,
        )
        usage["sparse_embedding"].update(
            calls=2, texts=8, api_tokens=120, api_calls=2,
            cache_hits=3, token_usage_reported=True,
            input_tokens=120, input_token_usage_reported=True,
        )

        summary = tracker.summary(usage)

        self.assertEqual(summary["embedding_api_tokens"], 220)
        self.assertEqual(summary["embedding_api_calls"], 4)
        self.assertEqual(summary["embedding_cache_hits"], 3)
        self.assertTrue(summary["embedding_token_usage_reported"])
        self.assertEqual(summary["embedding_input_tokens"], 220)
        self.assertIsNone(summary["embedding_output_tokens"])
        self.assertTrue(summary["embedding_input_token_usage_reported"])
        self.assertFalse(summary["embedding_output_token_usage_reported"])
        self.assertNotIn("sparse_embedding_api_tokens", summary)


if __name__ == "__main__":
    unittest.main()
