import os
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
