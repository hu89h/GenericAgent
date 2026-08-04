import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_base.usage import UsageTracker
from knowledge_base.zvec import ZvecIndex


class _Collection:
    def __init__(self):
        self.docs = []

    def insert(self, docs):
        self.docs.extend(docs)


class _Doc:
    def __init__(self, *, id, vectors, fields):
        self.id = id
        self.vectors = vectors
        self.fields = fields


class RecordEmbeddingCacheTests(unittest.TestCase):
    def test_unchanged_records_reuse_persisted_dense_and_sparse_vectors(self):
        records = [
            {
                "data_id": "kb-test::doc.md",
                "chunk_index": 0,
                "kind": "text",
                "file_name": "doc.md",
                "title": "Document",
                "body": "stable content",
            }
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "knowledge_base.zvec.embeddings.embedding_meta",
            return_value={"provider": "test", "model": "dense", "dimension": 2},
        ), mock.patch(
            "knowledge_base.zvec.embeddings.sparse_embedding_meta",
            return_value={"provider": "test", "model": "sparse", "dimension": 2},
        ):
            index = ZvecIndex(dimension=2, batch_size=8, usage_tracker=UsageTracker())
            kb = {"id": "kb-test", "path": directory}
            with mock.patch.object(
                index,
                "require",
                return_value=SimpleNamespace(Doc=_Doc),
            ), mock.patch.object(
                index, "embed_dense", return_value=[[0.1, 0.2]]
            ) as dense, mock.patch.object(
                index, "embed_sparse", return_value=[{1: 0.3}]
            ) as sparse:
                first = _Collection()
                index.insert_records(kb, first, records)
                second = _Collection()
                index.insert_records(kb, second, records)

            self.assertEqual(dense.call_count, 1)
            self.assertEqual(sparse.call_count, 1)
            self.assertEqual(len(first.docs), 1)
            self.assertEqual(len(second.docs), 1)
            self.assertEqual(second.docs[0].vectors["embedding"], [0.1, 0.2])

    def test_rechunked_record_reuses_vector_when_embedding_input_is_unchanged(self):
        first_record = {
            "data_id": "kb-test::doc.md",
            "chunk_index": 3,
            "kind": "text",
            "file_name": "doc.md",
            "body": "rendered body",
            "search_text": "Document\nSection\nstable semantic input",
        }
        moved_record = {**first_record, "chunk_index": 9, "body": "new rendering"}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "knowledge_base.zvec.embeddings.embedding_meta",
            return_value={"provider": "test", "model": "dense", "dimension": 2},
        ), mock.patch(
            "knowledge_base.zvec.embeddings.sparse_embedding_meta",
            return_value={"provider": "test", "model": "sparse", "dimension": 2},
        ):
            index = ZvecIndex(dimension=2, batch_size=8, usage_tracker=UsageTracker())
            kb = {"id": "kb-test", "path": directory}
            with mock.patch.object(
                index, "require", return_value=SimpleNamespace(Doc=_Doc)
            ), mock.patch.object(
                index, "embed_dense", return_value=[[0.1, 0.2]]
            ) as dense, mock.patch.object(
                index, "embed_sparse", return_value=[{1: 0.3}]
            ) as sparse:
                index.insert_records(kb, _Collection(), [first_record])
                index.insert_records(kb, _Collection(), [moved_record])

            self.assertEqual(dense.call_count, 1)
            self.assertEqual(sparse.call_count, 1)


if __name__ == "__main__":
    unittest.main()
