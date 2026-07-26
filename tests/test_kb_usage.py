import threading
import unittest

from knowledge_base.usage import UsageTracker


class UsageTrackerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
