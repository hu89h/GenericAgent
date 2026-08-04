import json
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from aiohttp import web

from frontends import desktop_bridge
import llmcore


class DesktopMultimodalTests(unittest.TestCase):
    def test_desktop_plan_state_uses_package_import_when_loaded_as_module(self):
        self.assertEqual(
            desktop_bridge._sanitize_desktop_plan_path(
                "sess-test", "temp/plan_example/plan.md"
            ),
            "temp/plan_example/plan.md",
        )
        self.assertEqual(
            desktop_bridge._sanitize_desktop_plan_path(
                "sess-test", "temp/not-a-plan/notes.md"
            ),
            "",
        )

    def test_visual_probe_uses_pending_config_and_requires_correct_sequence(self):
        response = SimpleNamespace(content="RRRRRRRR")
        sessions = []

        class FakeSession:
            def __init__(self, cfg):
                self.cfg = cfg
                self.system = ""
                sessions.append(self)

            def ask(self, message):
                self.message = message
                if False:
                    yield None
                return response

        with mock.patch("random.SystemRandom.choice", return_value="R"), \
             mock.patch("llmcore.NativeOAISession", FakeSession):
            llmcore.probe_model_vision({
                "apikey": "pending-key",
                "apibase": "https://example.test/v1",
                "model": "pending-model",
                "vision": True,
            }, "oai")

        self.assertEqual(sessions[0].cfg["apikey"], "pending-key")
        self.assertEqual(sessions[0].cfg["model"], "pending-model")
        self.assertFalse(sessions[0].cfg["stream"])
        self.assertEqual(sessions[0].cfg["max_tokens"], 32)
        self.assertEqual(sessions[0].message["content"][1]["type"], "image_path")
        self.assertEqual(sessions[0].message["content"][1]["name"], "vision-probe.png")

        response.content = "BBBBBBBB"
        with mock.patch("random.SystemRandom.choice", return_value="R"), \
             mock.patch("llmcore.NativeOAISession", FakeSession), \
             self.assertRaisesRegex(ValueError, "vision_probe_failed"):
            llmcore.probe_model_vision({
                "apikey": "pending-key",
                "apibase": "https://example.test/v1",
                "model": "pending-model",
                "vision": True,
            }, "oai")

    def test_explicit_visual_probe_does_not_write_model_config(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        manager.ga_root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "mykey.py"
            target.write_text("# unchanged\n", encoding="utf-8")
            manager._mykey_file = lambda: target
            with mock.patch("llmcore.probe_model_vision", side_effect=ValueError("vision_probe_failed: rejected")), \
                 self.assertRaisesRegex(ValueError, "vision_probe_failed"):
                manager.probe_model_profile_vision({
                    "protocol": "oai",
                    "apikey": "pending-key",
                    "apibase": "https://example.test/v1",
                    "model": "pending-model",
                })
            self.assertEqual(target.read_text(encoding="utf-8"), "# unchanged\n")

    def test_saving_model_does_not_probe_vision_implicitly(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        manager.ga_root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "mykey.py"
            target.write_text("# unchanged\n", encoding="utf-8")
            manager._mykey_file = lambda: target
            saved = []
            manager._save_mykey_text = lambda text: (saved.append(text) or [{"id": 0, "name": "pending-model"}])
            with mock.patch("llmcore.probe_model_vision") as probe:
                manager.add_model_profile({
                    "protocol": "oai",
                    "apikey": "pending-key",
                    "apibase": "https://example.test/v1",
                    "model": "pending-model",
                    "vision_mode": "text",
                })
            probe.assert_not_called()
            self.assertEqual(len(saved), 1)
            self.assertIn("pending-model", saved[0])
            self.assertNotIn("max_retries", saved[0])
            self.assertNotIn("connect_timeout", saved[0])
            self.assertNotIn("read_timeout", saved[0])

    def test_model_profile_advanced_transport_options_are_normalized(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        cfg = manager._build_cfg({
            "protocol": "oai",
            "api_mode": "responses",
            "stream": "false",
            "max_retries": "7",
            "connect_timeout": "21",
            "read_timeout": "480",
            "apikey": "key",
            "apibase": "https://example.test/v1",
            "model": "model",
            "vision_mode": "text",
        })
        self.assertEqual(cfg["api_mode"], "responses")
        self.assertIs(cfg["stream"], False)
        self.assertEqual((cfg["max_retries"], cfg["connect_timeout"], cfg["read_timeout"]), (7, 21, 480))
        with self.assertRaisesRegex(ValueError, "api_mode_invalid_for_protocol"):
            manager._build_cfg({
                "protocol": "claude",
                "api_mode": "responses",
                "apikey": "key",
                "apibase": "https://example.test/v1",
                "model": "model",
                "vision_mode": "text",
            })

    def test_editing_an_unchanged_verified_multimodal_profile_keeps_verification(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        existing = {
            "apikey": "vision-key",
            "apibase": "https://vision.example/v1",
            "model": "vision-model",
            "vision_mode": "multimodal",
            "vision": True,
            "vision_verified": True,
        }
        cfg = manager._build_cfg({
            "protocol": "oai",
            "apibase": existing["apibase"],
            "model": existing["model"],
            "vision_mode": "multimodal",
        }, existing, require_key=False)
        self.assertTrue(cfg["vision_verified"])
        with self.assertRaisesRegex(ValueError, "vision_probe_required"):
            manager._build_cfg({
                "protocol": "oai",
                "apibase": existing["apibase"],
                "model": "changed-model",
                "vision_mode": "multimodal",
            }, existing, require_key=False)

    def test_model_profile_detail_reports_key_for_password_card(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        manager._profile_at = lambda profile_id: ("native_oai_config", {
            "apikey": "secret-that-must-not-leak",
            "apibase": "https://example.test/v1",
            "model": "vision-model",
            "vision_mode": "text",
        })
        profile = manager.get_model_profile(0)
        self.assertTrue(profile["apiKeyConfigured"])
        self.assertEqual(profile["apiKey"], "secret-that-must-not-leak")

    def test_desktop_prompt_contract_has_no_legacy_images_argument(self):
        parameters = inspect.signature(desktop_bridge.AgentManager.submit_prompt).parameters
        self.assertNotIn("images", parameters)
        self.assertFalse(hasattr(desktop_bridge, "normalize_prompt"))

    def test_unverified_profile_rejects_images_before_persistence(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        manager.config = {}
        manager.list_model_profiles = lambda: [{
            "id": 0, "vision": False, "visionVerified": False,
        }]
        session = desktop_bridge.Session(id="s", llm_no=0)
        with self.assertRaises(web.HTTPBadRequest) as error:
            manager._require_session_vision(session, [{"path": "image.png"}])
        payload = json.loads(error.exception.text)
        self.assertEqual(payload["error"], "vision_model_unverified")

    def test_desktop_image_paths_must_be_under_upload_root_and_decode(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        manager.ga_root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "good.png"
            Image.new("RGB", (4, 3), "red").save(image)
            with mock.patch.object(desktop_bridge, "_WEB_UPLOAD_DIR", root):
                value = manager._validated_prompt_images([{"path": str(image), "name": "good.png"}])
                self.assertEqual(value[0]["path"], str(image.resolve()))

                outside = root.parent / "outside.png"
                Image.new("RGB", (2, 2), "blue").save(outside)
                try:
                    with self.assertRaises(web.HTTPBadRequest):
                        manager._validated_prompt_images([{"path": str(outside)}])
                finally:
                    outside.unlink(missing_ok=True)

    def test_session_persistence_contains_image_references_not_base64(self):
        manager = desktop_bridge.AgentManager.__new__(desktop_bridge.AgentManager)
        manager.lock = __import__("threading").RLock()
        with tempfile.TemporaryDirectory() as temp:
            manager._sessions_dir = Path(temp)
            image = Path(temp) / "image.png"
            Image.new("RGB", (2, 2), "red").save(image)
            session = desktop_bridge.Session(
                id="session-test",
                messages=[{"role": "user", "content": "look", "images": [{"path": str(image), "name": "image.png"}]}],
                llm_history=[{"role": "user", "content": [{"type": "image_path", "path": str(image), "name": "image.png"}]}],
            )
            manager._persist_session(session)
            raw = manager._session_file(session.id).read_text(encoding="utf-8")
            self.assertIn('"image_path"', raw)
            self.assertNotIn("data:image", raw)
            self.assertNotIn("base64", raw)

    def test_missing_visible_session_image_becomes_explicit_text(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing" / "image.png"
            messages = [{
                "role": "user",
                "content": "look",
                "images": [{"path": str(missing), "name": "image.png"}],
            }]
            restored = desktop_bridge.AgentManager._restored_session_messages(messages)
            self.assertEqual(restored[0]["images"], [])
            self.assertIn("no longer available", restored[0]["content"])


if __name__ == "__main__":
    unittest.main()
