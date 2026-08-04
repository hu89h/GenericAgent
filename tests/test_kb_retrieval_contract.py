import os
import tempfile
import unittest
from types import SimpleNamespace

from knowledge_base.retrieval import KnowledgeBaseRetriever


class _Registry:
    def __init__(self, kbs):
        self.kbs = list(kbs)

    def load_config(self):
        return self.kbs

    def kb_by_id(self, kb_id):
        return next((item for item in self.kbs if item["id"] == kb_id), None)


class _Index:
    def __init__(self, paths):
        self.paths = paths
        self.dense_calls = 0
        self.sparse_calls = 0

    def path(self, kb_path):
        return self.paths[kb_path]

    def probe(self, _path):
        return {
            "present": True,
            "openable": True,
            "schema_valid": True,
            "embedding_matches": True,
        }

    def embed_dense(self, _texts):
        self.dense_calls += 1
        return [[1.0]]

    def embed_sparse(self, _texts, *, text_type):
        self.sparse_calls += 1
        return [{1: 1.0}]

    @staticmethod
    def doc_id(data_id, chunk_index):
        return f"{data_id}:{chunk_index}"


class _Assets:
    @staticmethod
    def local_ref_key(value):
        return str(value or "").replace(" ", "")


def _text_hit(kb_id, file_name):
    data_id = f"{kb_id}::{file_name}"
    return {
        "kb_id": kb_id,
        "data_id": data_id,
        "chunk_index": 0,
        "score": 1.0,
        "score_type": "zvec",
        "kind": "text",
        "file_name": file_name,
        "title": f"{kb_id}.pdf",
        "ref": f"{kb_id}/{file_name}",
        "body": f"正文 {kb_id}",
        "snippet": f"正文 {kb_id}",
    }


class RetrievalContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.roots = {}
        kbs = []
        for kb_id in ("kb-a", "kb-b"):
            root = os.path.join(self.temp.name, kb_id)
            zvec = os.path.join(root, "zvec")
            os.makedirs(zvec)
            self.roots[root] = zvec
            kbs.append({"id": kb_id, "name": kb_id, "path": root, "exists": True})
        self.index = _Index(self.roots)
        self.retriever = KnowledgeBaseRetriever(
            registry=_Registry(kbs), index=self.index, assets=_Assets()
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_multi_kb_selection_stays_within_selected_documents_and_read_returns_body(self):
        calls = []

        def search_one(kb, _query, _top_k, _snippet_chars, **kwargs):
            calls.append((kb["id"], kwargs.get("file_name")))
            if kb["id"] == "kb-a":
                return [_text_hit("kb-a", "documents/a.md")]
            return [_text_hit("kb-b", "documents/b.md")]

        self.retriever._search_one_zvec = search_one
        result = self.retriever.search(
            "alpha", mode="vector", top_k=5,
            scope_targets=[
                {"kb_id": "kb-a", "documents": [{"file_name": "documents/a.md"}]},
                {"kb_id": "kb-b", "all_documents": True},
            ],
        )

        self.assertEqual({hit["kb_id"] for hit in result["results"]}, {"kb-a", "kb-b"})
        self.assertEqual(calls, [("kb-a", ["documents/a.md"]), ("kb-b", None)])
        self.assertTrue(all("abspath" not in hit for hit in result["results"]))

        def fetch(_kb, data_id, chunk_index, output_fields=None):
            return SimpleNamespace(fields={
                "data_id": data_id,
                "chunk_index": chunk_index,
                "title": "a.pdf",
                "header_path": "/章节/方法/",
                "body": "真实正文内容",
            })

        self.retriever._zvec_fetch_doc = fetch
        hit = next(item for item in result["results"] if item["kb_id"] == "kb-a")
        content = self.retriever.read_chunk(
            data_id=hit["data_id"], chunk_index=hit["chunk_index"], kb_id="kb-a"
        )
        self.assertIn("真实正文内容", content)
        self.assertIn("章节：方法", content)

    def test_empty_selection_does_not_call_embedding_or_return_cross_scope_hits(self):
        self.retriever._search_one_zvec = lambda *_args, **_kwargs: [
            _text_hit("kb-a", "documents/a.md")
        ]
        result = self.retriever.search(
            "nothing", mode="rrf", scope_targets=[
                {"kb_id": "kb-a", "all_documents": False, "documents": []},
            ],
        )
        self.assertEqual(result["results"], [])
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 0)

    def test_exact_image_reference_keeps_stable_image_id_without_internal_path(self):
        self.retriever._search_one_zvec_sparse = lambda *_args, **_kwargs: []
        self.retriever._query_image_rows = lambda _kb, **_kwargs: [{
            "kind": "image",
            "data_id": "kb-a::documents/a.md::image::img-1",
            "file_name": "documents/a.md",
            "title": "图1",
            "display_label": "图1 流程",
            "ref_key": "图1",
            "image_id": "img-1",
            "image_path": "documents/a.assets/img-1.png",
            "body": "图中结构说明",
            "description": "结构说明",
        }]
        result = self.retriever.search("图1", mode="sparse", top_k=1)
        self.assertEqual(self.index.sparse_calls, 1)
        self.assertEqual(len(result["results"]), 1)
        hit = result["results"][0]
        self.assertEqual(hit["kind"], "image")
        self.assertEqual(hit["data_id"], "kb-a::documents/a.md::image::img-1")
        self.assertEqual(hit["matched_by"], ["ref_exact"])
        self.assertNotIn("image_abspath", hit)

    def test_table_read_assembles_the_complete_structure_and_reports_following_content(self):
        data_id = "kb-a::documents/a.md"
        rows = {
            4: {
                "data_id": data_id,
                "chunk_index": 4,
                "kind": "text",
                "content_type": "table",
                "structure_id": "table-1",
                "structure_title": "重要财务与估值指标",
                "structure_part_index": 0,
                "structure_part_count": 2,
                "title": "a.pdf",
                "header_path": "/财务数据/",
                "body": "| 指标 | 2020E |\n| --- | --- |\n| 收入 | 100 |",
            },
            5: {
                "data_id": data_id,
                "chunk_index": 5,
                "kind": "text",
                "content_type": "table",
                "structure_id": "table-1",
                "structure_title": "重要财务与估值指标",
                "structure_part_index": 1,
                "structure_part_count": 2,
                "title": "a.pdf",
                "header_path": "/财务数据/",
                "body": "| 指标 | 2020E |\n| --- | --- |\n| 资产负债率 | 22.3% |",
            },
            6: {
                "data_id": data_id,
                "chunk_index": 6,
                "kind": "text",
                "content_type": "prose",
                "title": "a.pdf",
                "header_path": "/财务数据/",
                "body": "表后说明",
            },
        }

        self.retriever._zvec_fetch_doc = lambda _kb, _data_id, index, output_fields=None: (
            SimpleNamespace(fields=rows[index]) if index in rows else None
        )

        result = self.retriever.read_content(
            data_id=data_id,
            chunk_index=5,
            span=1,
            kb_id="kb-a",
            max_chars=4000,
        )

        self.assertEqual(result["start_chunk_index"], 4)
        self.assertEqual(result["end_chunk_index"], 5)
        self.assertIn("收入", result["content"])
        self.assertIn("资产负债率", result["content"])
        self.assertIn("22.3%", result["content"])
        self.assertFalse(result["continuation"]["has_more"])
        self.assertIsNone(result["continuation"]["next_chunk_index"])
        self.assertFalse(result["continuation"]["same_structure"])
        self.assertTrue(result["continuation"]["same_section"])

    def test_read_does_not_point_continuation_back_to_a_partially_returned_chunk(self):
        data_id = "kb-a::documents/report.md"
        fields = {
            "data_id": data_id,
            "chunk_index": 0,
            "body": "| 指标 | 2020E |\n| --- | --- |\n" + "| 很长的指标 | 22.3% |\n" * 80,
            "content_type": "table",
            "structure_id": "table-1",
            "structure_part_index": 0,
            "structure_part_count": 1,
            "title": "report.pdf",
            "header_path": "/财务数据/",
        }
        self.retriever._zvec_fetch_doc = (
            lambda _kb, _data_id, index, output_fields=None:
            SimpleNamespace(fields=fields) if index == 0 else None
        )

        result = self.retriever.read_content(
            data_id=data_id,
            chunk_index=0,
            max_chars=500,
        )

        self.assertTrue(result["continuation"]["has_more"])
        self.assertTrue(result["continuation"]["truncated_within_chunk"])
        self.assertIsNone(result["continuation"]["next_chunk_index"])
        self.assertGreater(result["continuation"]["required_max_chars"], 500)


if __name__ == "__main__":
    unittest.main()
