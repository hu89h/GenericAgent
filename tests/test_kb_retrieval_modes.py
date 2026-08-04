import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_base.retrieval import KnowledgeBaseRetriever


class _Registry:
    def __init__(self, kb):
        self.kb = kb

    def load_config(self):
        return [self.kb]

    def kb_by_id(self, kb_id):
        return self.kb if kb_id == self.kb["id"] else None


class _Index:
    def __init__(self, zvec_path):
        self.zvec_path = zvec_path
        self.dense_calls = 0
        self.sparse_calls = 0

    def path(self, _kb_path):
        return self.zvec_path

    def embed_dense(self, _texts):
        self.dense_calls += 1
        return [[1.0]]

    def embed_sparse(self, _texts, *, text_type):
        self.sparse_calls += 1
        return [{1: 1.0}]

    @staticmethod
    def require():
        return object()

    @staticmethod
    def connect(_path, **_kwargs):
        raise AssertionError("test search channels are stubbed")

    @staticmethod
    def doc_id(data_id, chunk_index):
        return f"{data_id}:{chunk_index}"

    @staticmethod
    def fetch(*_args, **_kwargs):
        return None


class _Assets:
    @staticmethod
    def local_ref_key(value):
        return str(value or "").replace(" ", "")


def _hit(score_type, data_id="kb-test::doc.md"):
    return {
        "kb_id": "kb-test",
        "data_id": data_id,
        "chunk_index": 0,
        "score": 1.0,
        "score_type": score_type,
        "kind": "text",
        "file_name": "doc.md",
        "title": "doc",
        "ref": "kb-test/doc.md",
        "body": "body",
        "snippet": "body",
    }


class RetrievalModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        zvec_path = os.path.join(self.temp.name, "zvec")
        os.makedirs(zvec_path)
        kb = {
            "id": "kb-test",
            "name": "test",
            "path": self.temp.name,
            "exists": True,
        }
        self.index = _Index(zvec_path)
        self.retriever = KnowledgeBaseRetriever(
            registry=_Registry(kb),
            index=self.index,
            assets=_Assets(),
        )
        self.retriever._search_exact_image_refs = lambda *_args, **_kwargs: [
            _hit("ref_exact")
        ]
        self.retriever._search_one_zvec = lambda *_args, **_kwargs: [_hit("zvec")]
        self.retriever._search_one_zvec_sparse = lambda *_args, **_kwargs: [
            _hit("zvec_sparse")
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_vector_mode_only_calls_dense_channel(self):
        result = self.retriever.search("图1 semantic", mode="vector")
        self.assertEqual(result["mode"], "vector")
        self.assertEqual(self.index.dense_calls, 1)
        self.assertEqual(self.index.sparse_calls, 0)
        self.assertEqual(result["results"][0]["matched_by"], ["ref_exact", "vector"])

    def test_sparse_mode_only_calls_sparse_channel(self):
        result = self.retriever.search("图1 TextGrad", mode="sparse")
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 1)
        self.assertEqual(result["results"][0]["matched_by"], ["ref_exact", "sparse"])

    def test_rrf_exposes_both_channel_ranks(self):
        result = self.retriever.search("mixed", mode="rrf")
        hit = result["results"][0]
        self.assertEqual(hit["matched_by"], ["ref_exact", "vector", "sparse"])
        self.assertEqual(hit["channel_ranks"], {
            "ref_exact": 1,
            "vector": 1,
            "sparse": 1,
        })
        self.assertEqual(hit["final_rank"], 1)

    def test_rrf_passes_expanded_candidate_pool_before_final_cut(self):
        calls = []

        def capture(_kb, _query, top_k, _snippet_chars, **_kwargs):
            calls.append(top_k)
            return []

        self.retriever._search_exact_image_refs = capture
        self.retriever._search_one_zvec = capture
        self.retriever._search_one_zvec_sparse = capture

        result = self.retriever.search("mixed", top_k=2, mode="rrf")

        self.assertEqual(result["results"], [])
        self.assertEqual(calls, [8, 8, 8])

    def test_selected_mode_failure_is_not_replaced_by_another_channel(self):
        with mock.patch.object(
            self.index,
            "embed_dense",
            side_effect=RuntimeError("dense unavailable"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "vector retrieval unavailable",
            ):
                self.retriever.search("semantic query", mode="vector")

        self.assertEqual(self.index.sparse_calls, 0)

    def test_broken_index_does_not_embed_query(self):
        self.index.probe = lambda _path: {
            "present": True,
            "openable": False,
            "schema_valid": False,
            "embedding_matches": False,
        }
        self.retriever._search_exact_image_refs = lambda *_args, **_kwargs: []
        result = self.retriever.search("query", mode="rrf")
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 0)
        self.assertFalse(result["results"])

    def test_selection_scope_passes_document_filter_to_each_channel(self):
        calls = []

        def capture(*_args, **kwargs):
            calls.append(kwargs.get("file_name"))
            return [_hit("zvec")]

        self.retriever._search_exact_image_refs = capture
        self.retriever._search_one_zvec = capture
        result = self.retriever.search(
            "scoped",
            mode="vector",
            scope_targets=[{
                "kb_id": "kb-test",
                "all_documents": False,
                "documents": [{"file_name": "documents/selected.md"}],
            }],
        )

        self.assertTrue(result["results"])
        self.assertEqual(calls, [["documents/selected.md"], ["documents/selected.md"]])

    def test_selection_scope_normalizes_ref_to_relative_document_name(self):
        calls = []

        def capture(*_args, **kwargs):
            calls.append(kwargs.get("file_name"))
            return [_hit("zvec")]

        self.retriever._search_exact_image_refs = capture
        self.retriever._search_one_zvec = capture
        self.retriever.search(
            "scoped",
            mode="vector",
            scope_targets=[{
                "kb_id": "kb-test",
                "documents": [{"ref": "kb-test/documents/selected.md"}],
            }],
        )

        self.assertEqual(calls, [["documents/selected.md"], ["documents/selected.md"]])

    def test_read_image_accepts_source_document_filter(self):
        captured = {}

        def query_rows(_kb, **kwargs):
            captured.update(kwargs)
            return [{
                "kind": "image",
                "data_id": "kb-test::documents/selected.md::image::one",
                "file_name": "documents/selected.md",
                "title": "图片",
                "source_data_id": kwargs.get("source_data_id"),
                "image_path": "documents/selected.assets/one.png",
                "image_id": "one",
                "ref_key": "图1",
            }]

        self.retriever._query_image_rows = query_rows
        result = self.retriever.read_image(
            kb_id="kb-test",
            ref_key="图1",
            source_data_id="kb-test::documents/selected.md",
        )

        self.assertEqual(captured["source_data_id"], "kb-test::documents/selected.md")
        self.assertEqual(result["data_id"], "kb-test::documents/selected.md::image::one")

    def test_read_chunk_resolves_ref_when_scope_also_supplies_kb_id(self):
        captured = {}

        def fetch(_kb, data_id, chunk_index, output_fields=None):
            captured.update(data_id=data_id, chunk_index=chunk_index)
            return SimpleNamespace(fields={
                "title": "原始文档.pdf",
                "header_path": "章节/方法",
                "body": "这是通过 ref 读取到的正文。",
            })

        self.retriever._zvec_fetch_doc = fetch
        content = self.retriever.read_chunk(
            ref="kb-test/documents/selected.md",
            kb_id="kb-test",
            chunk_index=2,
        )

        self.assertIn("这是通过 ref 读取到的正文。", content)
        self.assertEqual(captured, {
            "data_id": "kb-test::documents/selected.md",
            "chunk_index": 2,
        })


if __name__ == "__main__":
    unittest.main()
