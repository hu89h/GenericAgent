import os
import tempfile
import unittest
from unittest import mock

from knowledge_base import config


class KnowledgeBaseConfigTests(unittest.TestCase):
    def test_registry_persists_identity_and_derives_runtime_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = os.path.join(temp, "kb.yaml")
            data_root = os.path.join(temp, "kbs")
            source = os.path.join(temp, "source")
            os.makedirs(source)
            with mock.patch.object(config, "CONFIG_PATH", config_path), mock.patch.object(
                config, "DATA_ROOT", data_root
            ):
                kb_id = config.kb_id_for_source(source)
                config.upsert_kb(
                    kb_id,
                    name="paper",
                    source_path=source,
                    config_path=config_path,
                )
                rows = config.load_config(config_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], kb_id)
            self.assertEqual(
                rows[0]["path"],
                os.path.join(data_root, kb_id, "active", "processed"),
            )
            with open(config_path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertNotIn("    path:", text)
            self.assertIn("source_path:", text)


if __name__ == "__main__":
    unittest.main()
