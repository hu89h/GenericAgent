import os
import stat
import tempfile
import unittest
from pathlib import Path

from knowledge_base.fs import remove_tree


class KnowledgeBaseCleanupTests(unittest.TestCase):
    def test_remove_tree_handles_read_only_source_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "staging"
            target.mkdir()
            readonly = target / "source.pdf"
            readonly.write_bytes(b"source")
            os.chmod(readonly, stat.S_IREAD)

            remove_tree(target, ignore_errors=False)

            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
