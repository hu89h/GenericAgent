import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from knowledge_base.providers import vision


class VisionJsonParsingTests(unittest.TestCase):
    def test_repairs_only_unsupported_json_backslash_escapes(self):
        parsed = vision._extract_json(
            '{"description":"$S_t \\checkmark$","table_markdown":"",'
            '"ref_key":"图3","uncertain":[]}'
        )

        self.assertNotIn("error", parsed)
        self.assertEqual(parsed["description"], "$S_t \\checkmark$")
        self.assertEqual(parsed["ref_key"], "图3")

    def test_structurally_invalid_json_remains_a_failure(self):
        parsed = vision._extract_json('{"description":"unfinished"')

        self.assertEqual(parsed["error"], "model did not return valid JSON")

    def test_unsupported_classifier_does_not_match_transport_failures(self):
        self.assertTrue(vision.is_vision_unsupported_error(
            "HTTP 400: this model does not support image input"
        ))
        self.assertTrue(vision.is_vision_unsupported_error("模型仅支持文本输入"))
        self.assertFalse(vision.is_vision_unsupported_error("TLS EOF"))
        self.assertFalse(vision.is_vision_unsupported_error("HTTP 429: rate limit"))

    def test_anthropic_vision_request_uses_native_image_source_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "probe.png"
            Image.new("RGB", (4, 3), "red").save(image)
            cfg = {
                "base_url": "https://api.anthropic.com",
                "api_key": "sk-ant-test",
                "model": "claude-sonnet-test",
                "protocol": "anthropic",
                "timeout": 10,
                "retries": 1,
                "max_tokens": 128,
                "rpm_limit": 100,
                "tpm_limit": 10000,
                "rate_headroom": 0.8,
                "token_reserve": 10,
            }
            response = {
                "id": "msg_test",
                "content": [{"type": "text", "text": '{"description":"ok"}'}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
            with mock.patch.object(vision, "_config", return_value=cfg), \
                mock.patch.object(vision.rate_limit, "get_limiter", return_value=None), \
                mock.patch.object(vision.provider_http, "anthropic_messages", return_value=response) as call:
                result = vision._vision_chat(str(image), "describe")

            self.assertEqual(result["description"], "ok")
            kwargs = call.call_args.kwargs
            self.assertEqual(kwargs["auth_mode"], "x-api-key")
            content = kwargs["messages"][0]["content"]
            self.assertEqual(content[1]["type"], "image")
            self.assertEqual(content[1]["source"]["type"], "base64")
            self.assertEqual(content[1]["source"]["media_type"], "image/png")

    def test_vision_config_can_fall_back_to_native_anthropic(self):
        with mock.patch.object(
            vision.provider_settings,
            "_load_mykey_vars",
            return_value={
                "native_claude_config": {
                    "apikey": "relay-key",
                    "apibase": "https://example.test/anthropic",
                    "model": "claude-test",
                }
            },
        ):
            config = vision.provider_settings.vision_config()

        self.assertEqual(config["protocol"], "anthropic")
        self.assertEqual(config["model"], "claude-test")


if __name__ == "__main__":
    unittest.main()
