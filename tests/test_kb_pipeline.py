import os
import hashlib
import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_base import config
from knowledge_base.cancellation import KnowledgeBaseCancelled
from knowledge_base.locking import KnowledgeBaseLockedError, KnowledgeBaseMutationLock
from knowledge_base.pipeline import IngestPipeline, Publisher


class _ProbeIndex:
    @staticmethod
    def path(kb_path):
        return os.path.join(kb_path, ".kb_index", "zvec")


class _PreparedDocuments:
    def prepare(
        self,
        source_dir,
        *,
        stage_root,
        kb_id,
        name="",
        progress=None,
        cancelled=None,
    ):
        processed = os.path.join(stage_root, "processed")
        os.makedirs(processed)
        return {
            "processed_path": processed,
            "name": name or "test",
            "manifest": {"files": [], "summary": {}},
            "summary": {"ready": 1},
            "failures": [],
            "files": [],
        }


class _OneRecord:
    def build(
        self,
        kb,
        manifest,
        *,
        progress=None,
        logfn=None,
        cancelled=None,
    ):
        return SimpleNamespace(
            records=[{
                "data_id": f"{kb['id']}::doc.md",
                "chunk_index": 0,
                "kind": "text",
                "file_name": "doc.md",
                "title": "doc",
                "body": "content",
            }],
            failures=[],
            sources={},
            stats={},
        )

    @staticmethod
    def write_records(path, records, *, kb_id):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"body":"content"}\n')

    @staticmethod
    def records_sha256(path):
        return "test-records-sha"


class _FailingIndexBuilder:
    def __init__(self, message):
        self.message = message

    def begin_build(self):
        return None

    def build(self, *args, **kwargs):
        raise RuntimeError(self.message)


class _CancellingDocuments(_PreparedDocuments):
    def __init__(self, cancel_event):
        self.cancel_event = cancel_event

    def prepare(self, *args, **kwargs):
        result = super().prepare(*args, **kwargs)
        self.cancel_event.set()
        return result


class _DeleteRecords:
    @staticmethod
    def read_records(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def write_records(path, records, *, kb_id):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def records_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
        return digest.hexdigest()


class _DeleteIndexBuilder:
    def begin_build(self):
        return None

    def build(self, kb, *, records, sources, progress=None, logfn=None, cancelled=None):
        index_path = os.path.join(kb["path"], ".kb_index", "zvec")
        os.makedirs(index_path, exist_ok=True)
        if callable(progress):
            progress({"phase": "validated", "processed": len(records), "total": len(records)})
        docs = {record.get("file_name") for record in records}
        return {
            "n_docs": len(docs),
            "n_chunks": len(records),
            "text_chunks": len(records),
            "image_chunks": 0,
            "image_assets": 0,
        }


class _DeleteIndex:
    @staticmethod
    def probe(path):
        return {
            "present": os.path.isdir(os.path.join(path, ".kb_index", "zvec")),
            "openable": True,
            "schema_valid": True,
            "embedding_matches": True,
        }


class DocumentDeletionTests(unittest.TestCase):
    def test_delete_publishes_remaining_documents_and_records(self):
        with tempfile.TemporaryDirectory() as temp:
            data_root = os.path.join(temp, "kbs")
            config_path = os.path.join(temp, "kb.yaml")
            source = os.path.join(temp, "source")
            os.makedirs(source)
            with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                config, "CONFIG_PATH", config_path
            ):
                kb_id = "kb-delete-test"
                config.upsert_kb(kb_id, name="Delete test", source_path=source)
                active = config.active_root(kb_id)
                processed = config.processed_path(kb_id)
                os.makedirs(os.path.join(processed, "documents", "one.assets-x"))
                os.makedirs(os.path.join(processed, "documents"), exist_ok=True)
                with open(os.path.join(processed, "documents", "one.md"), "w", encoding="utf-8") as handle:
                    handle.write("one")
                with open(os.path.join(processed, "documents", "two.md"), "w", encoding="utf-8") as handle:
                    handle.write("two")
                with open(os.path.join(processed, "documents", "one.assets-x", "figure.png"), "wb") as handle:
                    handle.write(b"asset")
                manifest = {
                    "kb_id": kb_id,
                    "name": "Delete test",
                    "source_path": source,
                    "files": [
                        {"kind": "document", "source": "one.pdf", "name": "one.pdf", "processed": ["documents/one.md"]},
                        {"kind": "document", "source": "two.pdf", "name": "two.pdf", "processed": ["documents/two.md"]},
                    ],
                    "summary": {"n_docs": 2},
                    "failures": [],
                    "index_sources": {},
                }
                with open(os.path.join(active, "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle)
                _DeleteRecords.write_records(
                    os.path.join(active, "records.jsonl"),
                    [
                        {"data_id": f"{kb_id}::documents/one.md", "file_name": "documents/one.md", "kind": "text", "body": "one"},
                        {"data_id": f"{kb_id}::documents/two.md", "file_name": "documents/two.md", "kind": "text", "body": "two"},
                    ],
                    kb_id=kb_id,
                )
                os.makedirs(os.path.join(processed, ".kb_index", "zvec"), exist_ok=True)
                pipeline = IngestPipeline(
                    document_processor=None,
                    record_builder=_DeleteRecords(),
                    index_builder=_DeleteIndexBuilder(),
                    publisher=Publisher(),
                    index=_DeleteIndex(),
                )

                result = pipeline.delete_document(
                    kb_id,
                    data_id=f"{kb_id}::documents/one.md",
                )

                self.assertTrue(result["ok"])
                self.assertFalse(os.path.exists(os.path.join(processed, "documents", "one.md")))
                self.assertFalse(os.path.exists(os.path.join(processed, "documents", "one.assets-x")))
                self.assertTrue(os.path.isfile(os.path.join(processed, "documents", "two.md")))
                remaining = _DeleteRecords.read_records(os.path.join(active, "records.jsonl"))
                self.assertEqual([item["file_name"] for item in remaining], ["documents/two.md"])


class PublisherTests(unittest.TestCase):
    def test_registry_failure_restores_old_active_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            data_root = os.path.join(temp, "kbs")
            kb_id = "kb-test"
            with mock.patch.object(config, "DATA_ROOT", data_root):
                active = config.active_root(kb_id)
                stage = config.staging_root(kb_id)
                os.makedirs(active)
                os.makedirs(stage)
                with open(os.path.join(active, "old.txt"), "w", encoding="utf-8") as handle:
                    handle.write("old")
                with open(os.path.join(stage, "new.txt"), "w", encoding="utf-8") as handle:
                    handle.write("new")
                with mock.patch.object(config, "upsert_kb", side_effect=OSError("write failed")):
                    with self.assertRaises(OSError):
                        Publisher().publish(
                            kb_id=kb_id,
                            name="test",
                            source_path=temp,
                        )
                self.assertTrue(os.path.isfile(os.path.join(active, "old.txt")))
                self.assertFalse(os.path.exists(os.path.join(active, "new.txt")))


class IngestRollbackTests(unittest.TestCase):
    def test_embedding_and_zvec_failures_leave_old_active_untouched(self):
        for message in ("injected embedding failure", "injected zvec failure"):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                source = os.path.join(temp, "source")
                os.makedirs(source)
                data_root = os.path.join(temp, "kbs")
                config_path = os.path.join(temp, "kb.yaml")
                with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                    config, "CONFIG_PATH", config_path
                ):
                    kb_id = config.kb_id_for_source(source)
                    active = config.active_root(kb_id)
                    os.makedirs(os.path.join(active, "processed"))
                    sentinel = os.path.join(active, "old-active.txt")
                    with open(sentinel, "w", encoding="utf-8") as handle:
                        handle.write("old")
                    config.upsert_kb(kb_id, name="old", source_path=source)
                    pipeline = IngestPipeline(
                        document_processor=_PreparedDocuments(),
                        record_builder=_OneRecord(),
                        index_builder=_FailingIndexBuilder(message),
                        publisher=Publisher(),
                        index=_ProbeIndex(),
                    )

                    with self.assertRaisesRegex(RuntimeError, message):
                        pipeline.import_kb(source, name="new")

                    self.assertTrue(os.path.isfile(sentinel))
                    self.assertFalse(os.path.exists(config.staging_root(kb_id)))
                    self.assertFalse(os.path.exists(os.path.join(config.kb_root(kb_id), "rollback")))

    def test_cancellation_cleans_staging_and_preserves_old_active(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            os.makedirs(source)
            data_root = os.path.join(temp, "kbs")
            config_path = os.path.join(temp, "kb.yaml")
            cancel_event = threading.Event()
            with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                config, "CONFIG_PATH", config_path
            ):
                kb_id = config.kb_id_for_source(source)
                active = config.active_root(kb_id)
                os.makedirs(os.path.join(active, "processed"))
                sentinel = os.path.join(active, "old-active.txt")
                with open(sentinel, "w", encoding="utf-8") as handle:
                    handle.write("old")
                config.upsert_kb(kb_id, name="old", source_path=source)
                pipeline = IngestPipeline(
                    document_processor=_CancellingDocuments(cancel_event),
                    record_builder=_OneRecord(),
                    index_builder=_FailingIndexBuilder("must not build"),
                    publisher=Publisher(),
                    index=_ProbeIndex(),
                )

                with self.assertRaises(KnowledgeBaseCancelled):
                    pipeline.import_kb(
                        source,
                        name="new",
                        cancelled=cancel_event.is_set,
                    )

                self.assertTrue(os.path.isfile(sentinel))
                self.assertFalse(os.path.exists(config.staging_root(kb_id)))


class DocumentResultTests(unittest.TestCase):
    def test_results_are_based_on_published_records_and_original_documents(self):
        manifest = {
            "files": [
                {
                    "kind": "document",
                    "source": "books/alpha.pdf",
                    "name": "alpha.pdf",
                    "status": "ready",
                    "processed": ["documents/a-alpha.md"],
                },
                {
                    "kind": "document",
                    "source": "books/beta.pdf",
                    "name": "beta.pdf",
                    "status": "failed",
                    "processed": [],
                },
            ],
        }
        records = [
            {"kind": "text", "file_name": "documents/a-alpha.md"},
            {"kind": "text", "file_name": "documents/a-alpha.md"},
            {"kind": "image", "file_name": "documents/a-alpha.md"},
        ]
        results, failures = IngestPipeline._document_results(
            manifest,
            records,
            [
                {
                    "source": "documents/a-alpha.md:assets/figure.png",
                    "document": "documents/a-alpha.md",
                    "stage": "image_analysis",
                    "error_type": "ImageAnalysisError",
                    "error": "image failed",
                },
                {
                    "source": "books/beta.pdf",
                    "stage": "mineru",
                    "error_type": "MinerUError",
                    "error": "parse failed",
                },
            ],
        )

        self.assertEqual(results[0]["name"], "alpha.pdf")
        self.assertEqual(results[0]["status"], "succeeded_with_warnings")
        self.assertEqual(results[0]["text_chunks"], 2)
        self.assertEqual(results[0]["images_indexed"], 1)
        self.assertEqual(results[0]["images_total"], 2)
        self.assertEqual(results[1]["status"], "failed")
        self.assertEqual(failures[0]["source_document"], "books/alpha.pdf")
        self.assertEqual(failures[1]["source_document"], "books/beta.pdf")

    def test_selected_document_result_uses_original_filename(self):
        results, _failures = IngestPipeline._document_results(
            {
                "files": [{
                    "kind": "document",
                    "source": "files/abc-selected.md",
                    "source_path": "C:/documents/selected.md",
                    "name": "selected.md",
                    "processed": ["documents/selected.md"],
                }],
            },
            [{"kind": "text", "file_name": "documents/selected.md"}],
            [],
        )

        self.assertEqual(results[0]["name"], "selected.md")


class MutationLockTests(unittest.TestCase):
    def test_lock_is_reentrant_and_rejects_competing_thread(self):
        lock = KnowledgeBaseMutationLock(port=47891)
        errors = []
        with lock:
            with lock:
                def compete():
                    try:
                        with lock:
                            pass
                    except Exception as error:
                        errors.append(error)

                thread = threading.Thread(target=compete)
                thread.start()
                thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], KnowledgeBaseLockedError)


if __name__ == "__main__":
    unittest.main()
