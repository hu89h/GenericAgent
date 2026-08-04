import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from knowledge_base import backend, config
from knowledge_base.pipeline import IngestPipeline, Publisher


class _SelectionDocuments:
    def __init__(self):
        self.prepare_calls = 0

    @staticmethod
    def _selection_label(path):
        return f"files/{Path(path).name}"

    def prepare_files(self, source_files, *, stage_root, kb_id, name="", **_kwargs):
        self.prepare_calls += 1
        processed_root = Path(stage_root) / "processed" / "documents"
        processed_root.mkdir(parents=True, exist_ok=True)
        entries = []
        fingerprints = []
        for raw in source_files:
            source = Path(raw).resolve()
            relative = f"documents/{source.stem}.md"
            target = Path(stage_root) / "processed" / relative
            target.write_text(f"# {source.stem}\n\n{source.read_text(encoding='utf-8')}", encoding="utf-8")
            stat = source.stat()
            entries.append({
                "kind": "document",
                "status": "ready",
                "source": f"files/{source.name}",
                "source_path": str(source),
                "name": source.name,
                "processed": [relative],
            })
            fingerprints.append({
                "source": f"files/{source.name}",
                "path": str(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        manifest = {
            "schema_version": 1,
            "kb_id": kb_id,
            "name": name,
            "source_path": "",
            "files": entries,
            "source_fingerprint": fingerprints,
            "failures": [],
            "summary": {"ready": len(entries), "total": len(entries)},
        }
        (Path(stage_root) / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "manifest": manifest,
            "summary": manifest["summary"],
            "failures": [],
            "files": entries,
            "name": name,
            "processed_path": str(Path(stage_root) / "processed"),
        }


class _SelectionRecords:
    def __init__(self):
        self.build_calls = 0

    def build(self, kb, manifest, *, include_files=None, **_kwargs):
        self.build_calls += 1
        wanted = {
            str(value).replace("\\", "/").lstrip("/")
            for value in (include_files or [])
        }
        records = []
        sources = {}
        for entry in manifest.get("files") or []:
            for relative in entry.get("processed") or []:
                relative = str(relative).replace("\\", "/").lstrip("/")
                if include_files is not None and relative not in wanted:
                    continue
                records.append({
                    "data_id": f"{kb['id']}::{relative}",
                    "chunk_index": 0,
                    "kind": "text",
                    "file_name": relative,
                    "title": entry.get("name") or relative,
                    "body": "test body",
                })
                sources[relative] = {"size": 1, "mtime": 1}
        return SimpleNamespace(
            records=records,
            failures=[],
            sources={
                "documents": sources,
                "images": {},
                "chunking": {},
                "image_analysis": {},
            },
            stats={
                "n_docs": len(records),
                "n_chunks": len(records),
                "text_chunks": len(records),
                "image_chunks": 0,
                "image_assets": 0,
            },
        )

    @staticmethod
    def write_records(path, records, *, kb_id):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def records_sha256(path):
        return "records-sha"

    @staticmethod
    def read_records(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class _SelectionIndexBuilder:
    def __init__(self):
        self.build_calls = 0

    def begin_build(self):
        return None

    def build(self, kb, *, records, **_kwargs):
        self.build_calls += 1
        Path(kb["path"], ".kb_index", "zvec").mkdir(parents=True, exist_ok=True)
        return {
            "n_docs": len({record.get("file_name") for record in records}),
            "n_chunks": len(records),
            "text_chunks": len(records),
            "image_chunks": 0,
            "image_assets": 0,
            "usage": {},
        }


class _SelectionIndex:
    @staticmethod
    def probe(path):
        present = Path(path, ".kb_index", "zvec").is_dir()
        return {
            "present": present,
            "openable": present,
            "schema_valid": present,
            "embedding_matches": present,
            "error": "",
            "meta": {"stats": {"n_chunks": sum(
                1 for line in Path(path).parent.joinpath("records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )}} if present else {},
        }


class DuplicatePolicyTests(unittest.TestCase):
    def test_backend_image_retry_uses_the_non_resumable_maintenance_contract(self):
        calls = []

        class Pipeline:
            @staticmethod
            def retry_image_analysis(kb_id, *, progress=None, logfn=None, cancelled=None):
                calls.append((kb_id, progress, logfn, cancelled))
                return {"ok": True}

        runtime = SimpleNamespace(pipeline=Pipeline())
        with mock.patch.object(backend, "_runtime", return_value=runtime):
            result = backend.retry_image_analysis("kb-images")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls[0][0], "kb-images")
        self.assertFalse(backend._is_processing("kb-images"))

    def test_status_marks_an_old_chunking_contract_as_structurally_updatable(self):
        class OpenCollection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class Index:
            @staticmethod
            def probe(_path):
                return {
                    "present": True,
                    "openable": True,
                    "schema_valid": True,
                    "embedding_matches": True,
                    "error": "",
                    "meta": {"schema_version": 2},
                }

            @staticmethod
            def path(path):
                return str(Path(path) / ".kb_index" / "zvec")

            @staticmethod
            def open_collection(_path):
                return OpenCollection()

            @staticmethod
            def embedding_config_matches(_meta):
                return True

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            config, "DATA_ROOT", str(Path(temp) / "data")
        ), mock.patch.object(
            config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
        ):
            registered = config.create_kb("Old chunks")
            active = Path(config.active_root(registered["id"]))
            (active / "processed").mkdir(parents=True)
            (active / "manifest.json").write_text(json.dumps({
                "kb_id": registered["id"],
                "name": "Old chunks",
                "state": "ready",
                "summary": {"n_docs": 1},
                "failures": [],
                "index_sources": {
                    "chunking": {"markdown_parser": "older-contract"},
                },
            }), encoding="utf-8")
            runtime = SimpleNamespace(
                index=Index(),
                pipeline=SimpleNamespace(
                    checkpoint_status=lambda _kb_id: {"available": False}
                ),
                usage=SimpleNamespace(load=lambda _path: {}),
            )
            with mock.patch.object(backend, "_runtime", return_value=runtime):
                status = backend.kb_status(config.kb_by_id(registered["id"]))

        self.assertTrue(status["structure_update_available"])
        self.assertEqual(status["state"], "ready")

    def test_skip_is_a_successful_noop_and_replace_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "paper.md"
            source.write_text("first", encoding="utf-8")
            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ):
                kb = config.create_kb("Lifecycle")
                documents = _SelectionDocuments()
                records = _SelectionRecords()
                index_builder = _SelectionIndexBuilder()
                pipeline = IngestPipeline(
                    document_processor=documents,
                    record_builder=records,
                    index_builder=index_builder,
                    publisher=Publisher(),
                    index=_SelectionIndex(),
                )

                first = pipeline.add_documents(kb["id"], [str(source)])
                self.assertEqual(first["state"], "ready")
                self.assertEqual(documents.prepare_calls, 1)
                self.assertEqual(index_builder.build_calls, 1)

                skipped = pipeline.add_documents(kb["id"], [str(source)])
                self.assertTrue(skipped["noop"])
                self.assertEqual(skipped["notice"], "all_documents_skipped")
                self.assertEqual(skipped["skipped_files"], [source.name])
                self.assertEqual(documents.prepare_calls, 1)
                self.assertEqual(index_builder.build_calls, 1)

                replaced = pipeline.add_documents(
                    kb["id"], [str(source)], duplicate_policy="replace"
                )
                self.assertEqual(replaced["replaced_files"], [source.name])
                self.assertEqual(documents.prepare_calls, 2)
                self.assertEqual(index_builder.build_calls, 2)

    def test_incremental_sources_are_merged_as_a_mapping_and_replacement_clears_stale_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.md"
            second = Path(temp) / "second.md"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ):
                kb = config.create_kb("Lifecycle")
                pipeline = IngestPipeline(
                    document_processor=_SelectionDocuments(),
                    record_builder=_SelectionRecords(),
                    index_builder=_SelectionIndexBuilder(),
                    publisher=Publisher(),
                    index=_SelectionIndex(),
                )
                pipeline.add_documents(kb["id"], [str(first)])
                manifest_path = Path(config.active_root(kb["id"])) / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["failures"] = [{
                    "source": f"files/{first.name}",
                    "stage": "mineru",
                    "error_type": "MinerUError",
                    "error": "old failure",
                }]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                pipeline.add_documents(kb["id"], [str(second)])
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    set(manifest["index_sources"]["documents"]),
                    {"documents/first.md", "documents/second.md"},
                )
                self.assertEqual(len(manifest["failures"]), 1)

                pipeline.add_documents(
                    kb["id"], [str(first)], duplicate_policy="replace"
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertFalse(manifest["failures"])
                self.assertEqual(
                    set(manifest["index_sources"]["documents"]),
                    {"documents/first.md", "documents/second.md"},
                )

    def test_mismatched_checkpoint_blocks_new_mutations_without_deleting_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.md"
            second = Path(temp) / "second.md"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ):
                kb = config.create_kb("Lifecycle")
                pipeline = IngestPipeline(
                    document_processor=_SelectionDocuments(),
                    record_builder=_SelectionRecords(),
                    index_builder=_SelectionIndexBuilder(),
                    publisher=Publisher(),
                    index=_SelectionIndex(),
                )
                stage = Path(config.staging_root(kb["id"]))
                stage.mkdir(parents=True)
                marker = {
                    "state": "checkpoint",
                    "kb_id": kb["id"],
                    "source_path": "",
                    "files": [],
                    "checkpoint": {
                        "mode": "add_documents",
                        "created_at": 1,
                        "source_files": [str(first)],
                    },
                }
                manifest_path = stage / "manifest.json"
                manifest_path.write_text(json.dumps(marker), encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "kb_checkpoint_conflict"):
                    pipeline.add_documents(kb["id"], [str(second)])
                self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), marker)

                with self.assertRaisesRegex(RuntimeError, "kb_checkpoint_conflict"):
                    pipeline.delete(kb["id"])
                self.assertTrue(stage.exists())

    def test_image_retry_with_no_pending_images_is_a_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ), mock.patch(
                "knowledge_base.providers.vision.build_analysis_meta",
                return_value={"version": 1},
            ), mock.patch(
                "knowledge_base.providers.vision.enabled",
                side_effect=AssertionError("a no-op retry must not probe the model"),
            ):
                kb = config.create_kb("Images")
                active = Path(config.active_root(kb["id"]))
                active.mkdir(parents=True, exist_ok=True)
                (active / "processed").mkdir(parents=True, exist_ok=True)
                record = {
                    "data_id": f"{kb['id']}::image",
                    "chunk_index": 0,
                    "kind": "image",
                    "file_name": "documents/report.md",
                    "image_path": "documents/report.assets-x/figure.png",
                    "description": "already analyzed",
                    "table_markdown": "",
                    "body": "already analyzed",
                }
                (active / "records.jsonl").write_text(
                    json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                manifest = {
                    "kb_id": kb["id"],
                    "name": "Images",
                    "source_path": "",
                    "state": "ready",
                    "index_sources": {"image_analysis": {"version": 1}},
                    "files": [],
                    "failures": [],
                    "summary": {"documents_total": 1, "image_chunks": 1},
                }
                (active / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
                )
                documents = _SelectionDocuments()
                records = _SelectionRecords()
                index_builder = _SelectionIndexBuilder()
                pipeline = IngestPipeline(
                    document_processor=documents,
                    record_builder=records,
                    index_builder=index_builder,
                    publisher=Publisher(),
                    index=_SelectionIndex(),
                )
                result = pipeline.retry_image_analysis(kb["id"])
                self.assertTrue(result["noop"])
                self.assertEqual(result["notice"], "no_pending_image_analysis")
                self.assertEqual(index_builder.build_calls, 0)


if __name__ == "__main__":
    unittest.main()
