import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontends"))
from frontends import desktop_bridge


class KnowledgeBaseJobRetentionTests(unittest.TestCase):
    def tearDown(self):
        with desktop_bridge._kb_jobs_lock:
            desktop_bridge._kb_jobs.clear()

    def test_prunes_expired_jobs_and_caps_completed_history(self):
        now = int(time.time())
        with desktop_bridge._kb_jobs_lock:
            desktop_bridge._kb_jobs["expired"] = {
                "state": "completed",
                "updatedAt": now - desktop_bridge._KB_JOB_RETENTION_SECONDS - 1,
            }
            desktop_bridge._kb_jobs["running"] = {
                "state": "running",
                "updatedAt": 1,
            }
            for index in range(desktop_bridge._KB_JOB_MAX_COMPLETED + 5):
                desktop_bridge._kb_jobs[f"done-{index}"] = {
                    "state": "completed",
                    "updatedAt": now - index,
                }
            desktop_bridge._kb_prune_jobs_locked(now)
            jobs = dict(desktop_bridge._kb_jobs)

        self.assertNotIn("expired", jobs)
        self.assertIn("running", jobs)
        completed = [
            job for job in jobs.values() if job.get("state") == "completed"
        ]
        self.assertEqual(len(completed), desktop_bridge._KB_JOB_MAX_COMPLETED)


if __name__ == "__main__":
    unittest.main()
