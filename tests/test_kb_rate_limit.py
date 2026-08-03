import io
import threading
import unittest
import urllib.error
from email.message import Message
from types import SimpleNamespace
from unittest import mock

from knowledge_base.providers import embeddings, provider_http, vision
from knowledge_base.cancellation import KnowledgeBaseCancelled
from knowledge_base.providers.rate_limit import SlidingWindowRateLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


class FakeLimiter:
    def __init__(self):
        self.reservations = []
        self.actual = []

    def acquire(self, estimated_tokens):
        reservation = object()
        self.reservations.append((reservation, estimated_tokens))
        return reservation

    def reconcile(self, reservation, actual_tokens):
        self.actual.append((reservation, actual_tokens))


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @staticmethod
    def read():
        return b'{"ok": true, "usage": {"total_tokens": 7}}'


class KnowledgeBaseRateLimitTests(unittest.TestCase):
    def test_embedding_usage_records_api_response_and_missing_usage_separately(self):
        embeddings.drain_usage()

        embeddings._add_api_tokens("dense", {"data": []})
        missing = embeddings.drain_usage()
        self.assertEqual(missing["dense_api_calls"], 1)
        self.assertFalse(missing["dense_reported"])

        embeddings._add_api_tokens("dense", {"usage": {"input_tokens": 7}})
        reported = embeddings.drain_usage()
        self.assertEqual(reported["dense_api_calls"], 1)
        self.assertTrue(reported["dense_reported"])
        self.assertEqual(reported["dense"], 7)
        self.assertTrue(reported["dense_input_reported"])
        self.assertEqual(reported["dense_input_tokens"], 7)
        self.assertFalse(reported["dense_output_reported"])

    def test_cancelled_request_does_not_start_or_retry_http(self):
        cancelled = threading.Event()
        cancelled.set()
        with mock.patch.object(provider_http.urllib.request, "urlopen") as request:
            with self.assertRaises(KnowledgeBaseCancelled):
                provider_http.post_json(
                    "/v1",
                    {"value": 1},
                    base="https://example.test",
                    key="secret",
                    retries=4,
                    cancelled=cancelled.is_set,
                )
        request.assert_not_called()

    def test_cancellation_stops_waiting_for_an_inflight_request(self):
        started = threading.Event()
        release = threading.Event()
        cancelled = threading.Event()

        def blocking_urlopen(*_args, **_kwargs):
            started.set()
            release.wait(5)
            return FakeResponse()

        result = {}
        with mock.patch.object(
            provider_http.urllib.request,
            "urlopen",
            side_effect=blocking_urlopen,
        ):
            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "error",
                    self._run_cancelled_request(cancelled),
                ),
                daemon=True,
            )
            worker.start()
            self.assertTrue(started.wait(1))
            cancelled.set()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(result.get("error"), KnowledgeBaseCancelled)
            release.set()

    @staticmethod
    def _run_cancelled_request(cancelled):
        try:
            provider_http.post_json(
                "/v1",
                {"value": 1},
                base="https://example.test",
                key="secret",
                retries=4,
                cancelled=cancelled.is_set,
            )
        except Exception as error:
            return error
        return None

    def test_uses_only_headroom_budget(self):
        fake = FakeTime()
        limiter = SlidingWindowRateLimiter(
            rpm=10,
            tpm=100,
            headroom=0.8,
            burst_window_seconds=0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        limiter.acquire(30)
        limiter.acquire(30)
        limiter.acquire(30)

        self.assertEqual(fake.now, 60.0)
        self.assertEqual(limiter.snapshot()["tpm_budget"], 80)
        self.assertEqual(limiter.snapshot()["tokens"], 30)

    def test_actual_usage_replaces_estimate(self):
        fake = FakeTime()
        limiter = SlidingWindowRateLimiter(
            rpm=10,
            tpm=100,
            headroom=0.8,
            burst_window_seconds=0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        reservation = limiter.acquire(10)
        limiter.reconcile(reservation, 70)
        limiter.acquire(20)

        self.assertEqual(fake.now, 60.0)
        self.assertEqual(limiter.snapshot()["tokens"], 20)

    def test_derived_tps_prevents_one_second_burst(self):
        fake = FakeTime()
        limiter = SlidingWindowRateLimiter(
            rpm=600,
            tpm=60_000,
            headroom=0.8,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        limiter.acquire(500)
        limiter.acquire(500)

        self.assertEqual(fake.now, 1.0)
        self.assertEqual(limiter.snapshot()["tps_budget"], 800)

    def test_retry_after_is_honored_and_each_attempt_reserves_capacity(self):
        headers = Message()
        headers["Retry-After"] = "7"
        too_many = urllib.error.HTTPError(
            "https://example.test/v1",
            429,
            "rate limited",
            headers,
            io.BytesIO(b'{"error":"limited"}'),
        )
        limiter = FakeLimiter()
        progress = []
        with mock.patch.object(
            provider_http.urllib.request,
            "urlopen",
            side_effect=[too_many, FakeResponse()],
        ), mock.patch.object(provider_http.time, "sleep") as sleep, mock.patch.object(
            provider_http.random, "uniform", return_value=0
        ):
            result = provider_http.post_json(
                "/v1",
                {"value": 1},
                base="https://example.test",
                key="secret",
                retries=2,
                rate_limiter=limiter,
                estimated_tokens=11,
                usage_tokens=lambda body: body["usage"]["total_tokens"],
                on_progress=progress.append,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(len(limiter.reservations), 2)
        self.assertEqual(limiter.actual[0][1], 7)
        sleep.assert_called_once_with(7.0)
        self.assertEqual(progress[1]["event"], "retry_scheduled")
        self.assertEqual(progress[1]["reason"], "rate_limited")
        self.assertEqual(progress[-1]["event"], "attempt_succeeded")

    def test_total_timeout_stops_a_slow_image_request_before_more_retries(self):
        progress = []

        def slow_request(*_args, **_kwargs):
            provider_http.time.sleep(0.02)
            raise TimeoutError("read operation timed out")

        with mock.patch.object(provider_http, "_request_once", side_effect=slow_request):
            with self.assertRaisesRegex(RuntimeError, "总等待上限"):
                provider_http.post_json(
                    "/v1",
                    {"value": 1},
                    base="https://example.test",
                    key="secret",
                    retries=4,
                    total_timeout=0.01,
                    on_progress=progress.append,
                )

        self.assertTrue(any(item.get("event") == "deadline_exceeded" for item in progress))

    def test_embedding_request_uses_shared_headroom_limiter(self):
        config = {
            "api_key": "secret",
            "base_url": "https://example.test",
            "model": "text-embedding-v4",
            "dimension": 2,
            "max_tokens": 8192,
            "timeout": 60,
            "retries": 4,
            "rpm_limit": 1800,
            "tpm_limit": 1_200_000,
            "rate_headroom": 0.8,
        }
        response = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"input_tokens": 4},
        }
        with mock.patch.object(
            embeddings.provider_http, "embeddings", return_value=response
        ) as request:
            vectors = embeddings._post_embeddings(["测试 text"], config)

        self.assertEqual(vectors, [[0.1, 0.2]])
        kwargs = request.call_args.kwargs
        self.assertGreater(kwargs["estimated_tokens"], 0)
        self.assertEqual(kwargs["rate_limiter"].rpm, 1440)
        self.assertEqual(kwargs["rate_limiter"].tpm, 960_000)

    def test_embedding_default_concurrency_is_thirty_two(self):
        with mock.patch.object(
            embeddings.provider_settings,
            "embedding_config",
            return_value={"apikey": "secret", "apibase": "https://example.test", "model": "text-embedding-v4"},
        ):
            config = embeddings._runtime_config()

        self.assertEqual(config["concurrency"], 32)

    def test_embedding_batches_are_split_below_derived_tps(self):
        config = {
            "max_tokens": 8192,
            "tpm_limit": 1_200_000,
            "rate_headroom": 0.8,
        }
        texts = ["中" * 3000 for _ in range(10)]
        batches = embeddings._make_batches(texts, 10, config=config)

        self.assertGreater(len(batches), 1)
        for _position, batch in batches:
            prepared = embeddings._prepared_inputs(batch, config)
            self.assertLessEqual(
                embeddings._rate_limit_token_estimate(prepared),
                16_000,
            )

    def test_vlm_request_reserves_combined_input_output_tokens(self):
        config = {
            "api_key": "secret",
            "base_url": "https://example.test",
            "model": "vlm",
            "timeout": 60,
            "retries": 4,
            "max_tokens": 8192,
            "rpm_limit": 30_000,
            "tpm_limit": 5_000_000,
            "rate_headroom": 0.8,
            "token_reserve": 12_000,
        }
        response = {
            "choices": [
                {"message": {"content": "{}"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 2000,
                "total_tokens": 3000,
            },
        }
        prepared = SimpleNamespace(
            media_type="image/png",
            data="eA==",
            data_url="data:image/png;base64,eA==",
        )
        with mock.patch.object(vision, "_config", return_value=config), mock.patch.object(
            vision.multimodal, "prepare_image", return_value=prepared
        ), mock.patch.object(
            vision.provider_http, "chat_completions", return_value=response
        ) as request:
            result = vision._vision_chat("unused.png", "describe")

        self.assertEqual(result["_usage"]["total_tokens"], 3000)
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["estimated_tokens"], 12_000)
        self.assertEqual(kwargs["rate_limiter"].rpm, 24_000)
        self.assertEqual(kwargs["rate_limiter"].tpm, 4_000_000)


if __name__ == "__main__":
    unittest.main()
