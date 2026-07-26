import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_base import config
from knowledge_base.locking import KnowledgeBaseLockedError, KnowledgeBaseMutationLock
from knowledge_base.pipeline import IngestPipeline, Publisher


class _ProbeIndex:
    @staticmethod
    def path(kb_path):
        return os.path.join(kb_path, ".kb_index", "zvec")


class _PreparedDocuments:
    def prepare(self, source_dir, *, stage_root, kb_id, name="", progress=None):
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
    def build(self, kb, manifest, *, progress=None, logfn=None):
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
