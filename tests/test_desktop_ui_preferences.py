import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from frontends import desktop_bridge


class _JsonRequest:
    can_read_body = True

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class DesktopUiPreferenceTests(unittest.TestCase):
    def test_fold_defaults_round_trip_through_desktop_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "desktop-settings.json"
            manager = SimpleNamespace(
                config={},
                ga_root="test-root",
                mykey_path="test-mykey.py",
                list_model_profiles=lambda: [],
            )
            request = _JsonRequest({
                "config": {
                    "expandThinking": True,
                    "expandTools": False,
                    "defaultKbEnabled": False,
                    "ignored": "not-a-ui-preference",
                }
            })

            with (
                mock.patch.object(desktop_bridge, "_SETTINGS", settings_path),
                mock.patch.object(desktop_bridge, "manager", manager),
            ):
                asyncio.run(desktop_bridge.save_config_handler(request))
                response = asyncio.run(desktop_bridge.get_config_handler(None))

            stored = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["ui"]["expandThinking"], True)
            self.assertEqual(stored["ui"]["expandTools"], False)
            self.assertEqual(stored["ui"]["defaultKbEnabled"], False)
            self.assertNotIn("ignored", stored["ui"])

            payload = json.loads(response.text)
            self.assertEqual(payload["config"]["expandThinking"], True)
            self.assertEqual(payload["config"]["expandTools"], False)
            self.assertEqual(payload["config"]["defaultKbEnabled"], False)


if __name__ == "__main__":
    unittest.main()
