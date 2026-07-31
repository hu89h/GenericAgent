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

    def test_vision_config_does_not_enable_unverified_native_anthropic(self):
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

        self.assertIsNone(config["enabled"])
        self.assertEqual(config["model"], "")

    def test_vision_config_uses_selected_verified_profile_not_chat_profile(self):
        with mock.patch.object(
            vision.provider_settings,
            "_load_mykey_vars",
            return_value={
                "native_oai_chat": {
                    "apikey": "chat-key",
                    "apibase": "https://chat.example/v1",
                    "model": "text-only",
                    "vision_mode": "text",
                },
                "native_oai_vision": {
                    "apikey": "vision-key",
                    "apibase": "https://vision.example/v1",
                    "model": "vision-model",
                    "vision_mode": "multimodal",
                    "vision": True,
                    "vision_verified": True,
                },
                "kb_vision_config": {
                    "enabled": True,
                    "model_profile": "native_oai_vision",
                },
            },
        ), mock.patch.dict("os.environ", {
            "GA_KB_VISION_BASE_URL": "https://wrong.example/v1",
            "GA_KB_VISION_MODEL": "wrong-model",
        }, clear=False):
            config = vision.provider_settings.vision_config()

        self.assertEqual(config["model"], "vision-model")
        self.assertEqual(config["apikey"], "vision-key")
        self.assertEqual(config["model_profile"], "native_oai_vision")

    def test_vision_config_resolves_an_unambiguous_renumbered_profile(self):
        with mock.patch.object(
            vision.provider_settings,
            "_load_mykey_vars",
            return_value={
                "native_oai_config": {
                    "apikey": "vision-key",
                    "apibase": "https://vision.example/v1",
                    "model": "vision-model",
                    "vision_mode": "multimodal",
                    "vision": True,
                    "vision_verified": True,
                },
                "kb_vision_config": {
                    "enabled": True,
                    "model_profile": "native_oai_config1",
                },
            },
        ):
            config = vision.provider_settings.vision_config()
            self.assertTrue(vision.enabled())

        self.assertEqual(config["model"], "vision-model")
        self.assertEqual(config["apikey"], "vision-key")

    def test_saved_disable_wins_over_environment_override(self):
        with mock.patch.object(
            vision.provider_settings,
            "vision_config",
            return_value={"enabled": False},
        ), mock.patch.dict("os.environ", {"GA_KB_IMAGE_ANALYSIS": "1"}, clear=False):
            self.assertFalse(vision.enabled())

    def test_enabled_without_a_resolved_verified_model_stays_off(self):
        with mock.patch.object(
            vision.provider_settings,
            "vision_config",
            return_value={"enabled": True, "apibase": "", "apikey": "", "model": ""},
        ):
            self.assertFalse(vision.enabled())


if __name__ == "__main__":
    unittest.main()
