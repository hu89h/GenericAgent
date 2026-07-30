import json
import os
import tempfile
import unittest
from pathlib import Path

from knowledge_base.retrieval import KnowledgeBaseRetriever


class _Registry:
    def __init__(self, kb):
        self.kb = kb

    def load_config(self):
        return [self.kb]

    def kb_by_id(self, kb_id):
        return self.kb if kb_id == self.kb["id"] else None


class _UnusedIndex:
    pass


class _UnusedAssets:
    pass


class OriginalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source_root = root / "source"
        self.active_root = root / "kb" / "active"
        self.processed_root = self.active_root / "processed"
        self.source_document = self.source_root / "docs" / "original.md"
        self.source_asset = self.source_root / "docs" / "images" / "figure.jpg"
        self.processed_document = self.processed_root / "documents" / "normalized.md"

        self.source_asset.parent.mkdir(parents=True)
        self.processed_document.parent.mkdir(parents=True)
        self.source_document.write_text(
            "# 原始文档\n\n![原图](images/figure.jpg)\n",
            encoding="utf-8",
        )
        self.source_asset.write_bytes(b"original-image")
        self.processed_document.write_text(
            "# MinerU 处理结果\n\n这不是 Desktop 阅读器应展示的内容。\n",
            encoding="utf-8",
        )
        (self.active_root / "manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "source": "docs/original.md",
                            "kind": "document",
                            "processed": ["documents/normalized.md"],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        kb = {
            "id": "kb-test",
            "name": "test",
            "path": str(self.processed_root),
            "source_path": str(self.source_root),
            "exists": True,
        }
        self.retriever = KnowledgeBaseRetriever(
            registry=_Registry(kb),
            index=_UnusedIndex(),
            assets=_UnusedAssets(),
        )
        self.data_id = "kb-test::documents/normalized.md"

    def tearDown(self):
        self.temp.cleanup()

    def test_agent_document_read_remains_processed(self):
        result = self.retriever.read_document(data_id=self.data_id)

        self.assertIn("MinerU 处理结果", result["content"])
        self.assertNotIn("原始文档", result["content"])

    def test_desktop_source_resolution_returns_original(self):
        result = self.retriever.resolve_source_document(data_id=self.data_id)

        self.assertTrue(result["is_original"])
        self.assertEqual(
            os.path.realpath(result["path"]),
            os.path.realpath(self.source_document),
        )
        self.assertEqual(result["source_file_name"], "original.md")

    def test_processed_document_resolution_returns_normalized_markdown(self):
        result = self.retriever.resolve_processed_document(data_id=self.data_id)

        self.assertTrue(result["is_processed"])
        self.assertEqual(Path(result["path"]).resolve(), self.processed_document.resolve())
        self.assertEqual(result["file_name"], "documents/normalized.md")

    def test_missing_original_never_falls_back_to_processed(self):
        self.source_document.unlink()

        result = self.retriever.resolve_source_document(data_id=self.data_id)

        self.assertEqual(result["error_code"], "source_document_not_found")
        self.assertNotIn("path", result)

    def test_source_asset_is_relative_to_original_document(self):
        result = self.retriever.resolve_source_asset(
            data_id=self.data_id,
            image_path="images/figure.jpg",
        )

        self.assertEqual(os.path.realpath(result), os.path.realpath(self.source_asset))

    def test_source_asset_rejects_escape_and_remote_paths(self):
        self.assertIsNone(
            self.retriever.resolve_source_asset(
                data_id=self.data_id,
                image_path="../../outside.jpg",
            )
        )
        self.assertIsNone(
            self.retriever.resolve_source_asset(
                data_id=self.data_id,
                image_path="https://example.com/figure.jpg",
            )
        )

    def test_selected_document_can_resolve_original_without_shared_source_root(self):
        entry = {
            "source": "files/abc-selected.md",
            "name": "selected.md",
            "source_path": str(self.source_document),
            "kind": "document",
            "processed": ["documents/normalized.md"],
        }
        (self.active_root / "manifest.json").write_text(
            json.dumps({"files": [entry]}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.retriever._registry.kb["source_path"] = ""

        result = self.retriever.resolve_source_document(data_id=self.data_id)

        self.assertTrue(result["is_original"])
        self.assertEqual(Path(result["path"]).resolve(), self.source_document.resolve())
        self.assertEqual(result["source_file_name"], "selected.md")

    def test_selected_document_listing_keeps_original_display_name(self):
        entry = {
            "source": "files/abc-selected.md",
            "name": "selected.md",
            "source_path": str(self.source_document),
            "kind": "document",
            "processed": ["documents/normalized.md"],
        }
        (self.active_root / "manifest.json").write_text(
            json.dumps({"files": [entry]}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.retriever._registry.kb["source_path"] = ""

        documents = self.retriever.list_documents(kb_id="kb-test")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["source_file_name"], "selected.md")
        self.assertTrue(documents[0]["source_exists"])


if __name__ == "__main__":
    unittest.main()
