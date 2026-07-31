import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import requests

from knowledge_base.cancellation import KnowledgeBaseCancelled
from knowledge_base.providers import mineru


class _Response:
    def __init__(self, chunks, *, headers=None, status=200):
        self._chunks = list(chunks)
        self.headers = headers or {}
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, *, chunk_size):
        del chunk_size
        yield from self._chunks


class MinerUDownloadTests(unittest.TestCase):
    def test_complete_cache_hit_skips_remote_configuration_and_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.pdf"
            source.write_bytes(b"stable source")
            cache = root / "cache"
            cache.mkdir()
            config = {
                "base_url": "https://mineru.example/api/v4",
                "model_version": "vlm",
            }
            key = mineru._cache_key(
                source,
                "report.pdf",
                base_url=config["base_url"],
                model_version=config["model_version"],
            )
            cached = mineru._cache_path(cache, key)
            with zipfile.ZipFile(cached, "w") as archive:
                archive.writestr("result.md", "# cached")

            events = []
            with mock.patch.object(
                mineru.provider_settings,
                "mineru_config",
                return_value=config,
            ), mock.patch.object(
                mineru,
                "load_config",
                side_effect=AssertionError("cache hit must not require an API key"),
            ):
                jobs = mineru.process_batches(
                    [(source, "report.pdf")],
                    root / "downloads",
                    on_update=events.append,
                    cache_dir=cache,
                )

            self.assertEqual(len(jobs), 1)
            self.assertTrue(jobs[0].cache_hit)
            self.assertEqual(jobs[0].result_path, cached)
            self.assertEqual(events[0].state, "downloaded")
            self.assertFalse((root / "downloads").exists())

    def test_mineru_cache_cleanup_removes_partial_and_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / ".result.zip.part").write_bytes(b"partial")
            (cache / "broken.zip").write_bytes(b"not a zip")
            valid = cache / "valid.zip"
            with zipfile.ZipFile(valid, "w") as archive:
                archive.writestr("result.md", "# valid")

            summary = mineru.cleanup_cache(cache)

            self.assertEqual(summary["removed"], 2)
            self.assertFalse((cache / ".result.zip.part").exists())
            self.assertFalse((cache / "broken.zip").exists())
            self.assertTrue(valid.exists())

    def test_successful_remote_result_is_persisted_as_a_complete_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.pdf"
            source.write_bytes(b"remote source")
            cache = root / "cache"
            settings = {
                "api_key": "test-key",
                "base_url": "https://mineru.example/api/v4",
                "model_version": "vlm",
            }

            def poll(_config, _batch_id, batch, on_update, **_kwargs):
                for item in batch:
                    item.state = "done"
                    item.zip_url = "https://cdn.example.test/result.zip"
                    on_update(item)

            def download(_url, target, **_kwargs):
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target, "w") as archive:
                    archive.writestr("result.md", "# remote")

            with mock.patch.object(
                mineru.provider_settings,
                "mineru_config",
                return_value=settings,
            ), mock.patch.object(
                mineru,
                "_request_upload_urls",
                return_value="batch-1",
            ), mock.patch.object(
                mineru,
                "_upload",
                side_effect=lambda item, _cancelled=None: item,
            ), mock.patch.object(mineru, "_poll_batch", side_effect=poll), mock.patch.object(
                mineru, "_download", side_effect=download
            ) as mocked_download:
                jobs = mineru.process_batches(
                    [(source, "report.pdf")],
                    root / "downloads",
                    on_update=lambda _item: None,
                    cache_dir=cache,
                )

            self.assertEqual(mocked_download.call_count, 1)
            self.assertFalse(jobs[0].cache_hit)
            self.assertTrue(jobs[0].result_path)
            self.assertTrue(Path(jobs[0].result_path).is_relative_to(cache))
            self.assertTrue(mineru._valid_zip(Path(jobs[0].result_path)))
            self.assertEqual(len(list(cache.glob("*.zip"))), 1)

    def test_transient_tls_failure_retries_and_replaces_atomically(self):
        payload = b"complete result"
        response = _Response(
            [payload],
            headers={"Content-Length": str(len(payload))},
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            mineru.requests,
            "get",
            side_effect=[requests.exceptions.SSLError("unexpected EOF"), response],
        ) as get, mock.patch.object(mineru, "wait_with_cancellation"):
            target = Path(directory) / "result.zip"
            mineru._download("https://cdn.example.test/result.zip", target)

            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(get.call_count, 2)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_cancelled_stream_does_not_leave_a_partial_result(self):
        cancelled = True
        response = _Response([b"partial"])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            mineru.requests, "get", return_value=response
        ):
            target = Path(directory) / "result.zip"
            with self.assertRaises(KnowledgeBaseCancelled):
                mineru._download("https://cdn.example.test/result.zip", target, lambda: cancelled)

            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
