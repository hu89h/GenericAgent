import os
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from knowledge_base import config
from knowledge_base.assets import ImageAssetProcessor
from knowledge_base.build import RecordBuilder
from knowledge_base.cancellation import KnowledgeBaseCancelled
from knowledge_base.locking import KnowledgeBaseLockedError, KnowledgeBaseMutationLock
from knowledge_base.pipeline import IngestPipeline, Publisher
from knowledge_base.usage import UsageTracker


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
        resume_manifest=None,
        retain_on_cancel=None,
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

    def test_cancellation_can_retain_a_checkpoint_without_publishing(self):
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
                        retain_partial=lambda: True,
                    )

                checkpoint = Path(config.staging_root(kb_id)) / "manifest.json"
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(json.loads(checkpoint.read_text(encoding="utf-8"))["state"], "checkpoint")

    def test_startup_cleanup_removes_invalid_checkpoint_but_keeps_valid_image_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            data_root = os.path.join(temp, "kbs")
            config_path = os.path.join(temp, "kb.yaml")
            source = os.path.join(temp, "source")
            os.makedirs(source)
            with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                config, "CONFIG_PATH", config_path
            ):
                kb_id = config.kb_id_for_source(source)
                root = Path(config.kb_root(kb_id))
                invalid_stage = root / "staging"
                invalid_stage.mkdir(parents=True)
                (invalid_stage / "manifest.json").write_text(
                    json.dumps({
                        "state": "checkpoint",
                        "files": [],
                        "checkpoint": {
                            "mode": "unknown",
                            "created_at": int(time.time()),
                        },
                    }),
                    encoding="utf-8",
                )
                pipeline = IngestPipeline(
                    document_processor=_PreparedDocuments(),
                    record_builder=_OneRecord(),
                    index_builder=_FailingIndexBuilder("unused"),
                    publisher=Publisher(),
                    index=_ProbeIndex(),
                )
                pipeline.cleanup_orphans()
                self.assertFalse(invalid_stage.exists())

                valid_stage = root / "staging"
                valid_stage.mkdir(parents=True)
                (valid_stage / "manifest.json").write_text(
                    json.dumps({
                        "state": "checkpoint",
                        "files": [],
                        "checkpoint": {
                            "mode": "retry_image_analysis",
                            "created_at": int(time.time()),
                        },
                    }),
                    encoding="utf-8",
                )
                pipeline.cleanup_orphans()
                self.assertTrue(valid_stage.exists())


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


class ProcessedContentRepairTests(unittest.TestCase):
    class _Assets(ImageAssetProcessor):
        def __init__(self):
            super().__init__(usage_tracker=UsageTracker())

        def analyze_image_jobs(self, kb, image_jobs, log, progress=None, cancelled=None):
            results = {}
            jobs = list(image_jobs.values())
            for index, job in enumerate(jobs, 1):
                results[job.image_sha] = {
                    "description": "repaired image description",
                    "table_markdown": "",
                    "ref_key": "",
                    "uncertain": [],
                }
                if callable(progress):
                    progress({
                        "phase": "image_analysis",
                        "analysis_completed": index,
                        "analysis_total": len(jobs),
                        "image_documents": [{
                            "key": job.origins[0]["key"],
                            "name": job.origins[0]["name"],
                            "completed": index,
                            "total": len(jobs),
                        }],
                    })
            return results

    class _Index:
        @staticmethod
        def path(kb_path):
            return os.path.join(kb_path, ".kb_index", "zvec")

        def probe(self, kb_path):
            present = os.path.isdir(self.path(kb_path))
            return {
                "present": present,
                "openable": present,
                "schema_valid": present,
                "embedding_matches": present,
                "error": "",
                "meta": {},
            }

    class _IndexBuilder:
        def __init__(self, index):
            self.index = index

        def begin_build(self):
            return None

        def build(self, kb, *, records, sources, progress=None, logfn=None, cancelled=None):
            os.makedirs(self.index.path(kb["path"]), exist_ok=True)
            stats = {
                "n_docs": len({record.get("file_name") for record in records}),
                "n_chunks": len(records),
                "text_chunks": sum(record.get("kind") == "text" for record in records),
                "image_chunks": sum(record.get("kind") == "image" for record in records),
                "image_assets": sum(record.get("kind") == "image" for record in records),
            }
            if callable(progress):
                progress({"phase": "validated", "processed": len(records), "total": len(records), **stats})
            return stats

    def test_repair_retries_processed_images_without_mineru(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            data_root = os.path.join(temp, "data")
            config_path = os.path.join(temp, "kb.yaml")
            os.makedirs(source)
            with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                config, "CONFIG_PATH", config_path
            ), mock.patch("knowledge_base.providers.vision.enabled", return_value=True):
                kb_id = config.kb_id_for_source(source)
                config.upsert_kb(kb_id, name="Repair test", source_path=source)
                active = config.active_root(kb_id)
                processed = config.processed_path(kb_id)
                image_dir = os.path.join(processed, "documents", "doc.assets-test")
                os.makedirs(image_dir)
                with open(os.path.join(processed, "documents", "doc.md"), "w", encoding="utf-8") as handle:
                    handle.write("# Test\n\n![图1](doc.assets-test/figure.png)\n")
                with open(os.path.join(image_dir, "figure.png"), "wb") as handle:
                    handle.write(b"test image")
                manifest = {
                    "kb_id": kb_id,
                    "name": "Repair test",
                    "source_path": source,
                    "files": [{
                        "kind": "document",
                        "source": "doc.pdf",
                        "name": "doc.pdf",
                        "processed": ["documents/doc.md"],
                    }],
                    "summary": {"documents_total": 1},
                    "failures": [{
                        "source": "documents/doc.md:doc.assets-test/figure.png",
                        "stage": "image_analysis",
                        "error": "timed out",
                    }],
                    "index_sources": {},
                }
                os.makedirs(active, exist_ok=True)
                with open(os.path.join(active, "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle)
                os.makedirs(os.path.join(processed, ".kb_index", "image_cache"), exist_ok=True)
                index = self._Index()
                pipeline = IngestPipeline(
                    document_processor=None,
                    record_builder=RecordBuilder(assets=self._Assets()),
                    index_builder=self._IndexBuilder(index),
                    publisher=Publisher(),
                    index=index,
                )

                result = pipeline.retry_image_analysis(kb_id)

                self.assertEqual(result["state"], "ready")
                self.assertEqual(result["summary"]["image_chunks"], 1)
                self.assertFalse(result["failures"])
                records = RecordBuilder.read_records(os.path.join(active, "records.jsonl"))
                image_records = [record for record in records if record.get("kind") == "image"]
                self.assertEqual(len(image_records), 1)
                self.assertEqual(image_records[0]["description"], "repaired image description")
                self.assertFalse(os.path.exists(config.staging_root(kb_id)))

    def test_reindex_rebuilds_only_the_published_records(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            data_root = os.path.join(temp, "data")
            config_path = os.path.join(temp, "kb.yaml")
            os.makedirs(source)
            with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                config, "CONFIG_PATH", config_path
            ), mock.patch("knowledge_base.providers.vision.enabled", side_effect=AssertionError("reindex must not probe vision")):
                kb_id = config.kb_id_for_source(source)
                config.upsert_kb(kb_id, name="Index test", source_path=source)
                active = config.active_root(kb_id)
                processed = config.processed_path(kb_id)
                os.makedirs(processed, exist_ok=True)
                records = [{
                    "data_id": f"{kb_id}::doc.md::0",
                    "chunk_index": 0,
                    "kind": "text",
                    "file_name": "doc.md",
                    "title": "doc",
                    "body": "content",
                }]
                records_path = os.path.join(active, "records.jsonl")
                with open(records_path, "w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                with open(os.path.join(active, "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump({"index_sources": {}, "files": []}, handle)

                class _PublishedRecords:
                    @staticmethod
                    def read_records(path):
                        with open(path, encoding="utf-8") as handle:
                            return [json.loads(line) for line in handle if line.strip()]

                    @staticmethod
                    def records_sha256(path):
                        return "published-records"

                index = self._Index()
                pipeline = IngestPipeline(
                    document_processor=None,
                    record_builder=_PublishedRecords(),
                    index_builder=self._IndexBuilder(index),
                    publisher=Publisher(),
                    index=index,
                )

                result = pipeline.reindex(kb_id)

                self.assertTrue(result["ok"])
                self.assertEqual(result["stats"]["n_chunks"], 1)
                self.assertEqual(result["stats"]["text_chunks"], 1)

    def test_reindex_failure_keeps_published_active_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            data_root = os.path.join(temp, "data")
            config_path = os.path.join(temp, "kb.yaml")
            os.makedirs(source)
            with mock.patch.object(config, "DATA_ROOT", data_root), mock.patch.object(
                config, "CONFIG_PATH", config_path
            ):
                kb_id = config.kb_id_for_source(source)
                config.upsert_kb(kb_id, name="Transactional", source_path=source)
                active = config.active_root(kb_id)
                processed = config.processed_path(kb_id)
                os.makedirs(os.path.join(processed, ".kb_index", "zvec"), exist_ok=True)
                records_path = os.path.join(active, "records.jsonl")
                with open(records_path, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"body": "published"}) + "\n")
                manifest_path = os.path.join(active, "manifest.json")
                original_manifest = {"state": "ready", "marker": "old"}
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(original_manifest, handle)

                class _Records:
                    @staticmethod
                    def read_records(path):
                        with open(path, encoding="utf-8") as handle:
                            return [json.loads(line) for line in handle if line.strip()]

                    @staticmethod
                    def records_sha256(path):
                        return "published-records"

                pipeline = IngestPipeline(
                    document_processor=None,
                    record_builder=_Records(),
                    index_builder=_FailingIndexBuilder("index failure"),
                    publisher=Publisher(),
                    index=self._Index(),
                )
                with self.assertRaisesRegex(RuntimeError, "index failure"):
                    pipeline.reindex(kb_id)

                with open(manifest_path, encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), original_manifest)
                self.assertTrue(os.path.isdir(active))
                self.assertFalse(os.path.exists(config.staging_root(kb_id)))


if __name__ == "__main__":
    unittest.main()
