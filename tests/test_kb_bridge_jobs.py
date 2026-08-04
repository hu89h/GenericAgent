import asyncio
import json
import sys
import tempfile
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

    def test_cancel_can_request_processed_checkpoint_retention(self):
        cancel_event = threading.Event()

        class Request:
            match_info = {"job_id": "kbimp-keep"}
            can_read_body = True

            @staticmethod
            async def json():
                return {"retainProcessed": True}

        with desktop_bridge._kb_jobs_lock:
            desktop_bridge._kb_jobs["kbimp-keep"] = {
                "ok": True,
                "jobId": "kbimp-keep",
                "mode": "import",
                "state": "running",
                "phase": "chunking",
                "cancelEvent": cancel_event,
                "startedAt": int(time.time()),
                "updatedAt": int(time.time()),
            }

        response = asyncio.run(desktop_bridge.kb_job_cancel_handler(Request()))
        payload = json.loads(response.text)

        self.assertEqual(response.status, 202)
        self.assertNotIn("retainProcessed", payload)
        self.assertTrue(payload["cancelRequested"])
        self.assertTrue(cancel_event.is_set())

    def test_classifies_transient_mineru_download_failures_as_network_errors(self):
        error = (
            "下载 MinerU 解析结果失败：HTTPSConnectionPool(host='cdn.example', "
            "port=443): Max retries exceeded (Caused by "
            "SSLEOFError(8, 'UNEXPECTED_EOF_WHILE_READING'))"
        )

        self.assertEqual(desktop_bridge._kb_error_code(error), "network_error")

    def test_resume_directory_checkpoint_routes_to_import_pipeline(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "report.pdf"
            source.write_bytes(b"pdf")
            calls = []

            class FakeBackend:
                def checkpoint_inputs(self, kb_id):
                    return {
                        "available": True,
                        "mode": "import",
                        "source_path": root,
                        "source_files": [str(source)],
                    }

                def status(self, kb_id=None):
                    return {"knowledge_bases": [{"id": "kb-resume"}]}

                def import_kb(self, source_dir, **kwargs):
                    calls.append(("import", source_dir))
                    return {
                        "summary": {
                            "documents_total": 1,
                            "documents_succeeded": 1,
                            "documents_failed": 0,
                        },
                        "failures": [],
                        "documents": [],
                        "usage": {},
                    }

                def add_documents(self, kb_id, source_files, **kwargs):
                    calls.append(("add_documents", list(source_files)))
                    raise AssertionError("directory checkpoint was routed to add_documents")

            class Request:
                can_read_body = True

                @staticmethod
                async def json():
                    return {"kbId": "kb-resume", "resume": True}

            original_backend = desktop_bridge._kb_backend
            desktop_bridge._kb_backend = lambda: FakeBackend()
            try:
                response = asyncio.run(desktop_bridge.kb_import_handler(Request()))
                payload = json.loads(response.text)
                self.assertEqual(response.status, 202)
                job_id = payload["jobId"]
                for _ in range(100):
                    with desktop_bridge._kb_jobs_lock:
                        job = desktop_bridge._kb_jobs.get(job_id) or {}
                        state = job.get("state")
                    if state in desktop_bridge._KB_TERMINAL_JOB_STATES:
                        break
                    time.sleep(0.01)
                self.assertEqual(calls, [("import", root)])
                self.assertEqual(state, "completed")
            finally:
                desktop_bridge._kb_backend = original_backend

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
                    "failures": [{
                        "stage": "image_analysis",
                        "error": "当前模型不支持图片输入",
                    }],
                },
            ],
            "imageDocuments": [
                {"key": "internal/a.md", "name": "alpha.pdf", "completed": 3, "total": 3},
            ],
            "imageActivity": {
                "completed": 3,
                "total": 4,
                "cached": 2,
                "active": 1,
                "retrying": 1,
                "items": [{
                    "name": "alpha.pdf",
                    "state": "retrying",
                    "attempt": 2,
                    "attempts": 4,
                    "elapsed": 42,
                    "reason": "timeout",
                }],
            },
            "failures": [
                {
                    "source": "documents/internal-a.md:assets/internal.png",
                    "source_document": "books/alpha.pdf",
                    "stage": "image_analysis",
                    "error": "当前模型不支持图片输入",
                },
            ],
        })

        self.assertEqual(snapshot["progress"]["completed"], 2)
        self.assertNotIn("documentProgress", snapshot)
        self.assertEqual(snapshot["documents"][0]["name"], "alpha.pdf")
        self.assertEqual(snapshot["documents"][0]["status"], "succeeded_with_warnings")
        self.assertEqual(snapshot["documents"][0]["warningCount"], 1)
        self.assertEqual(snapshot["recommendedActions"], ["check_configuration"])
        self.assertEqual(
            snapshot["documents"][0]["recommendedActions"],
            ["check_configuration"],
        )
        self.assertEqual(snapshot["documents"][0]["errorCodes"], ["vision_unsupported"])
        self.assertNotIn("source", snapshot["documents"][0])
        self.assertEqual(snapshot["imageDocuments"], [
            {"name": "alpha.pdf", "completed": 3, "total": 3},
        ])
        self.assertEqual(snapshot["imageActivity"]["cached"], 2)
        self.assertEqual(snapshot["imageActivity"]["items"][0]["reason"], "timeout")
        self.assertEqual(snapshot["failures"][0]["source"], "books/alpha.pdf")
        self.assertNotIn("internal-a.md", snapshot["failures"][0]["source"])

    def test_cancelled_image_retry_reports_cached_partial_results_without_checkpoint(self):
        snapshot = desktop_bridge._kb_job_snapshot({
            "jobId": "kbimageretry-cancelled",
            "mode": "retry_image_analysis",
            "state": "cancelled",
            "phase": "cancelled",
            "checkpointAvailable": False,
            "checkpoint": {},
            "partialResultsRetained": True,
        })

        self.assertNotIn("checkpointAvailable", snapshot)
        self.assertNotIn("resumeAvailable", snapshot)
        self.assertTrue(snapshot["partialResultsRetained"])
        self.assertNotIn("maintenanceCheckpointAvailable", snapshot)

    def test_failure_actions_follow_the_failed_stage(self):
        self.assertEqual(
            desktop_bridge._kb_recommended_action(
                mode="import", stage="image_analysis", error="read operation timed out"
            ),
            "retry_image_analysis",
        )
        self.assertEqual(
            desktop_bridge._kb_recommended_action(
                mode="import", stage="image_resolve", error="image missing"
            ),
            "retry_document_import",
        )
        self.assertEqual(
            desktop_bridge._kb_recommended_action(
                mode="import", stage="indexing", error="schema invalid"
            ),
            "reindex",
        )

    def test_snapshot_exposes_provider_usage_and_keeps_unknown_tokens_unknown(self):
        snapshot = desktop_bridge._kb_job_snapshot({
            "jobId": "kbimp-usage",
            "mode": "import",
            "state": "completed",
            "result": {
                "usage": {
                    "available": True,
                    "image_calls": 2,
                    "image_model": "vision-model",
                    "image_prompt_tokens": None,
                    "image_completion_tokens": None,
                    "image_token_usage_reported": False,
                    "embedding_calls": 2,
                    "embedding_model": "embedding-model",
                    "embedding_texts": 10,
                    "embedding_api_calls": 2,
                    "embedding_api_tokens": None,
                    "embedding_token_usage_reported": False,
                },
            },
        })

        self.assertEqual(snapshot["usage"]["image_calls"], 2)
        self.assertEqual(snapshot["usage"]["image_model"], "vision-model")
        self.assertEqual(snapshot["usage"]["embedding_model"], "embedding-model")
        self.assertIsNone(snapshot["usage"]["image_prompt_tokens"])
        self.assertIsNone(snapshot["usage"]["embedding_api_tokens"])
        self.assertFalse(snapshot["usage"]["embedding_token_usage_reported"])

    def test_public_status_does_not_expose_managed_document_paths(self):
        public = desktop_bridge._kb_public_status({
            "id": "kb-test",
            "source_path": r"C:\\private\\source",
            "documents": [{
                "data_id": "kb-test::documents/report.md",
                "abspath": r"C:\\private\\data\\kbs\\kb-test\\active\\processed\\documents\\report.md",
                "source_path": r"C:\\private\\source\\report.pdf",
                "title": "report.pdf",
            }],
            "failures": [{"source": "report.pdf", "error": r"C:\\private\\tmp\\failure"}],
            "index": {"present": True, "openable": True, "schema_valid": True},
        })

        self.assertNotIn("source_path", public)
        self.assertNotIn("abspath", public["documents"][0])
        self.assertNotIn("source_path", public["documents"][0])
        self.assertNotIn(r"C:\\private", json.dumps(public, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
