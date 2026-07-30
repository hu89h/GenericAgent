import tempfile
import unittest
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
