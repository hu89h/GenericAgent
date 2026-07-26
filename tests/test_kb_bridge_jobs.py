import asyncio
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_cancel_signals_running_job_and_exposes_cancelling_state(self):
        cancel_event = threading.Event()
        with desktop_bridge._kb_jobs_lock:
            desktop_bridge._kb_jobs["kbimp-cancel"] = {
                "ok": True,
                "jobId": "kbimp-cancel",
                "mode": "import",
                "state": "running",
                "phase": "image_analysis",
                "cancelEvent": cancel_event,
                "progress": {
                    "completed": 5,
                    "total": 10,
                    "unit": "images",
                    "indeterminate": False,
                },
                "startedAt": int(time.time()),
                "updatedAt": int(time.time()),
            }

        response = asyncio.run(desktop_bridge.kb_job_cancel_handler(
            SimpleNamespace(match_info={"job_id": "kbimp-cancel"})
        ))
        payload = json.loads(response.text)

        self.assertEqual(response.status, 202)
        self.assertTrue(cancel_event.is_set())
        self.assertEqual(payload["state"], "cancelling")
        self.assertEqual(payload["phase"], "cancelling")
        self.assertTrue(payload["cancelRequested"])
        self.assertTrue(payload["cancellable"])

    def test_cancel_is_rejected_after_publishing_starts(self):
        cancel_event = threading.Event()
        with desktop_bridge._kb_jobs_lock:
            desktop_bridge._kb_jobs["kbimp-publishing"] = {
                "ok": True,
                "jobId": "kbimp-publishing",
                "mode": "import",
                "state": "running",
                "phase": "publishing",
                "cancelEvent": cancel_event,
                "startedAt": int(time.time()),
                "updatedAt": int(time.time()),
            }

        response = asyncio.run(desktop_bridge.kb_job_cancel_handler(
            SimpleNamespace(match_info={"job_id": "kbimp-publishing"})
        ))
        payload = json.loads(response.text)

        self.assertEqual(response.status, 409)
        self.assertEqual(payload["error"], "kb_cancel_too_late")
        self.assertFalse(cancel_event.is_set())

    def test_reindex_job_is_not_cancellable(self):
        with desktop_bridge._kb_jobs_lock:
            desktop_bridge._kb_jobs["kbreindex-running"] = {
                "ok": True,
                "jobId": "kbreindex-running",
                "mode": "reindex",
                "state": "running",
                "phase": "indexing",
                "startedAt": int(time.time()),
                "updatedAt": int(time.time()),
            }

        response = asyncio.run(desktop_bridge.kb_job_cancel_handler(
            SimpleNamespace(match_info={"job_id": "kbreindex-running"})
        ))
        payload = json.loads(response.text)

        self.assertEqual(response.status, 409)
        self.assertEqual(payload["error"], "kb_job_not_cancellable")
        snapshot = desktop_bridge._kb_job_snapshot(
            desktop_bridge._kb_jobs["kbreindex-running"]
        )
        self.assertFalse(snapshot["cancellable"])

    def test_snapshot_exposes_final_document_and_grouped_image_progress(self):
        snapshot = desktop_bridge._kb_job_snapshot({
            "jobId": "kbimp-test",
            "mode": "import",
            "state": "completed_with_failures",
            "phase": "completed_with_failures",
            "progress": {
                "completed": 2,
                "total": 2,
                "unit": "documents",
                "indeterminate": False,
            },
            "documentProgress": {
                "completed": 2,
                "total": 2,
                "failed": 1,
                "ready": 1,
            },
            "documents": [
                {
                    "source": "books/alpha.pdf",
                    "name": "alpha.pdf",
                    "status": "succeeded_with_warnings",
                    "text_chunks": 10,
                    "images_indexed": 2,
                    "images_total": 3,
                    "failures": [{"error": "image timeout"}],
                },
            ],
            "imageDocuments": [
                {"key": "internal/a.md", "name": "alpha.pdf", "completed": 3, "total": 3},
            ],
            "failures": [
                {
                    "source": "documents/internal-a.md:assets/internal.png",
                    "source_document": "books/alpha.pdf",
                    "stage": "image_analysis",
                    "error": "image timeout",
                },
            ],
        })

        self.assertEqual(snapshot["progress"]["completed"], 2)
        self.assertEqual(snapshot["documentProgress"]["total"], 2)
        self.assertEqual(snapshot["documents"][0]["name"], "alpha.pdf")
        self.assertEqual(snapshot["documents"][0]["status"], "succeeded_with_warnings")
        self.assertEqual(snapshot["documents"][0]["warningCount"], 1)
        self.assertNotIn("source", snapshot["documents"][0])
        self.assertEqual(snapshot["imageDocuments"], [
            {"name": "alpha.pdf", "completed": 3, "total": 3},
        ])
        self.assertEqual(snapshot["failures"][0]["source"], "books/alpha.pdf")
        self.assertNotIn("internal-a.md", snapshot["failures"][0]["source"])


if __name__ == "__main__":
    unittest.main()
