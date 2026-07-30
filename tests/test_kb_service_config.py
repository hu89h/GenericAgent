import runpy
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontends"))
from frontends import desktop_bridge


class KnowledgeBaseServiceConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.mykey = Path(self.temp.name) / "mykey.py"
        self.mykey.write_text(
            "\n".join([
                "kb_embedding_config = {",
                "    'apikey': 'embedding-secret',",
                "    'apibase': 'https://embedding.example/v1',",
                "    'model': 'old-embedding',",
                "    'dimension': 768,",
                "}",
                "mineru_config = {",
                "    'api_key': 'mineru-secret',",
                "    'base_url': 'https://mineru.example/api/v4',",
                "    'model_version': 'vlm',",
                "}",
                "",
            ]),
            encoding="utf-8",
        )
        self.provider_settings = SimpleNamespace(
            EMBEDDING_BASE_URL="https://default-embedding.example/v1",
            EMBEDDING_MODEL="text-embedding-v4",
            EMBEDDING_DIMENSION=1024,
            MINERU_BASE_URL="https://default-mineru.example/api/v4",
            MINERU_MODEL_VERSION="vlm",
            embedding_config=lambda: {
                "apikey": "embedding-secret",
                "apibase": "https://embedding.example/v1",
                "model": "old-embedding",
                "dimension": 768,
            },
            mineru_config=lambda: {
                "api_key": "mineru-secret",
                "base_url": "https://mineru.example/api/v4",
                "model_version": "vlm",
            },
            vision_config=lambda: {
                "enabled": None,
                "model_profile": "",
            },
        )
        self.manager = object.__new__(desktop_bridge.AgentManager)
        self.manager.ensure_ga_import_path = lambda: Path(self.temp.name)
        self.manager._mykey_file = lambda: self.mykey
        self.manager._kb_provider_settings = lambda: self.provider_settings
        self.manager.list_model_profiles = lambda: []

        def save_text(text):
            self.mykey.write_text(text, encoding="utf-8")
            return []

        self.manager._save_mykey_text = save_text

    def tearDown(self):
        self.temp.cleanup()

    def test_config_cards_receive_keys_for_password_inputs(self):
        result = self.manager.get_kb_service_configs()

        self.assertTrue(result["embedding"]["apiKeyConfigured"])
        self.assertTrue(result["mineru"]["apiKeyConfigured"])
        self.assertEqual(result["embedding"]["apiKey"], "embedding-secret")
        self.assertEqual(result["mineru"]["apiKey"], "mineru-secret")

    def test_vision_card_resolves_a_stale_numeric_profile_suffix(self):
        verified = {
            "id": 1,
            "varName": "native_oai_config",
            "kind": "native",
            "name": "通义千问",
            "model": "qwen3.7-plus",
            "vision": True,
            "visionVerified": True,
            "apiMode": "chat_completions",
        }
        self.provider_settings.vision_config = lambda: {
            "enabled": True,
            "model_profile": "native_oai_config1",
        }
        self.manager.list_model_profiles = lambda: [verified]

        result = self.manager.get_kb_service_configs()

        self.assertEqual(result["vision"]["modelProfile"], "native_oai_config")
        self.assertEqual(result["vision"]["modelProfileName"], "通义千问")

    def test_partial_embedding_save_keeps_both_existing_keys(self):
        raw = runpy.run_path(str(self.mykey))
        with mock.patch("llmcore.reload_mykeys", return_value=(raw, False)):
            result = self.manager.save_kb_service_configs({
                "embedding": {
                    "apiKey": "",
                    "baseUrl": "https://new-embedding.example/v1/",
                    "model": "text-embedding-v4",
                    "dimension": "1024",
                },
            })

        saved = runpy.run_path(str(self.mykey))
        self.assertEqual(saved["kb_embedding_config"]["apikey"], "embedding-secret")
        self.assertEqual(
            saved["kb_embedding_config"]["apibase"],
            "https://new-embedding.example/v1",
        )
        self.assertEqual(saved["kb_embedding_config"]["dimension"], 1024)
        self.assertEqual(saved["mineru_config"]["api_key"], "mineru-secret")
        self.assertIn("mineru", result)

    def test_invalid_url_and_dimension_are_rejected(self):
        raw = runpy.run_path(str(self.mykey))
        invalid_payloads = [
            {
                "embedding": {
                    "baseUrl": "file:///tmp/model",
                    "model": "text-embedding-v4",
                    "dimension": "1024",
                },
            },
            {
                "embedding": {
                    "baseUrl": "https://embedding.example/v1",
                    "model": "text-embedding-v4",
                    "dimension": "0",
                },
            },
        ]

        with mock.patch("llmcore.reload_mykeys", return_value=(raw, False)):
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    with self.assertRaises(ValueError):
                        self.manager.save_kb_service_configs(payload)

    def test_vision_save_requires_and_persists_a_verified_profile(self):
        verified = {
            "id": 2,
            "varName": "native_oai_aux",
            "kind": "native",
            "name": "视觉专用组",
            "model": "qwen3.7-plus",
            "vision": True,
            "visionVerified": True,
        }
        self.manager.list_model_profiles = lambda: [verified]
        raw = runpy.run_path(str(self.mykey))
        with mock.patch("llmcore.reload_mykeys", return_value=(raw, False)):
            with self.assertRaisesRegex(ValueError, "vision_model_required"):
                self.manager.save_kb_service_configs({"vision": {"enabled": True}})
            self.manager.save_kb_service_configs({
                "vision": {"enabled": True, "modelProfile": "native_oai_aux"},
            })
        saved = runpy.run_path(str(self.mykey))
        self.assertEqual(saved["kb_vision_config"], {
            "enabled": True,
            "model_profile": "native_oai_aux",
        })

    def test_concurrent_card_saves_do_not_discard_each_other(self):
        """Two UI cards may submit at once; both changes must survive."""
        raw = runpy.run_path(str(self.mykey))

        def slow_save(text):
            # Make an unguarded read-modify-write race observable without
            # relying on a particular machine or filesystem timing.
            time.sleep(0.02)
            self.mykey.write_text(text, encoding="utf-8")
            return []

        self.manager._save_mykey_text = slow_save
        barrier = threading.Barrier(3)
        errors = []
        payloads = [
            {
                "embedding": {
                    "baseUrl": "https://embedding.concurrent/v1",
                    "model": "text-embedding-v4",
                    "dimension": "1024",
                },
            },
            {
                "mineru": {
                    "baseUrl": "https://mineru.concurrent/api/v4",
                    "modelVersion": "vlm",
                },
            },
        ]

        def save(payload):
            try:
                barrier.wait(timeout=2)
                self.manager.save_kb_service_configs(payload)
            except Exception as error:  # pragma: no cover - assertion below
                errors.append(error)

        with mock.patch("llmcore.reload_mykeys", return_value=(raw, False)):
            threads = [threading.Thread(target=save, args=(payload,)) for payload in payloads]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=5)

        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        saved = runpy.run_path(str(self.mykey))
        self.assertEqual(
            saved["kb_embedding_config"]["apibase"],
            "https://embedding.concurrent/v1",
        )
        self.assertEqual(
            saved["mineru_config"]["base_url"],
            "https://mineru.concurrent/api/v4",
        )


if __name__ == "__main__":
    unittest.main()
