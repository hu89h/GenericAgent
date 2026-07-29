import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledge_base import config, importer
from knowledge_base.assets import ImageAssetProcessor
from knowledge_base.build import RecordBuilder
from knowledge_base.importer import DocumentProcessor
from knowledge_base.pipeline import IngestPipeline, Publisher
from knowledge_base.usage import UsageTracker


_PAPER_SOURCE = os.environ.get("GA_KB_PAPER_TEST_SOURCE", "").strip()
PAPER_SOURCE = Path(_PAPER_SOURCE).expanduser() if _PAPER_SOURCE else None


class _DeterministicAssets(ImageAssetProcessor):
    def __init__(self, *, failed_images=0, skip_vision=False):
        super().__init__(usage_tracker=UsageTracker())
        self.failed_images = failed_images
        self.skip_vision = skip_vision

    def analyze_image_jobs(
        self,
        kb,
        image_jobs,
        log,
        progress=None,
        cancelled=None,
    ):
        results = {}
        jobs = list(image_jobs.values())
        for index, job in enumerate(jobs):
            if index < self.failed_images:
                result = {"error": "injected image failure"}
            elif self.skip_vision:
                result = {
                    "vision_skipped": True,
                    "analysis_warning": "当前模型不支持图片输入，已保留基础图片引用",
                }
            else:
                result = {
                    "description": f"deterministic description for {job.title}",
                    "table_markdown": "",
                    "ref_key": "",
                    "uncertain": [],
                }
            results[job.image_sha] = result
            if callable(progress):
                progress({
                    "phase": "image_analysis",
                    "analysis_completed": index + 1,
                    "analysis_total": len(jobs),
                })
        return results


class _FakeIndex:
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


class _FakeIndexBuilder:
    def __init__(self, index):
        self.index = index

    def begin_build(self):
        return None

    def build(
        self,
        kb,
        *,
        records,
        sources,
        progress=None,
        logfn=None,
        cancelled=None,
    ):
        os.makedirs(self.index.path(kb["path"]), exist_ok=True)
        kinds = [record.get("kind") for record in records]
        stats = {
            "n_docs": len({
                record["data_id"]
                for record in records
                if record.get("kind") != "image"
            }),
            "n_chunks": len(records),
            "text_chunks": kinds.count("text"),
            "image_chunks": kinds.count("image"),
            "image_assets": kinds.count("image"),
        }
        if callable(progress):
            progress({"phase": "validated", **stats})
        return stats


@unittest.skipUnless(
    PAPER_SOURCE is not None and PAPER_SOURCE.is_dir(),
    "set GA_KB_PAPER_TEST_SOURCE to a real paper integration fixture",
)
class PaperPartialFailureTests(unittest.TestCase):
    def _copy_paper(self, target):
        shutil.copytree(PAPER_SOURCE, target, copy_function=shutil.copy2)

    def _pipeline(self, failed_images=0, skip_vision=False):
        assets = _DeterministicAssets(
            failed_images=failed_images,
            skip_vision=skip_vision,
        )
        index = _FakeIndex()
        return IngestPipeline(
            document_processor=DocumentProcessor(),
            record_builder=RecordBuilder(assets=assets),
            index_builder=_FakeIndexBuilder(index),
            publisher=Publisher(),
            index=index,
        )

    def test_one_document_failure_publishes_remaining_real_paper_content(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "paper-doc-failure"
            self._copy_paper(source)
            markdowns = sorted(source.rglob("*.md"))
            self.assertGreaterEqual(len(markdowns), 2)
            rejected_name = markdowns[0].name
            original = importer._write_markdown

            def fail_one(path, *args, **kwargs):
                if Path(path).name == rejected_name:
                    raise OSError("injected markdown failure")
                return original(path, *args, **kwargs)

            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ), mock.patch.object(importer, "_write_markdown", side_effect=fail_one):
                result = self._pipeline().import_kb(str(source))
                kb_id = result["kb"]["id"]
                active = Path(config.active_root(kb_id))

                self.assertEqual(result["state"], "ready_with_warnings")
                self.assertEqual(result["summary"]["n_docs"], len(markdowns) - 1)
                self.assertGreater(result["summary"]["text_chunks"], 0)
                self.assertEqual(result["failures"][0]["stage"], "markdown")
                self.assertEqual(
                    sorted(item["status"] for item in result["documents"]),
                    ["failed", *(["succeeded"] * (len(markdowns) - 1))],
                )
                self.assertTrue(active.is_dir())
                self.assertFalse(Path(config.staging_root(kb_id)).exists())

    def test_one_image_failure_skips_only_that_real_paper_image(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "paper-image-failure"
            self._copy_paper(source)
            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ):
                baseline = self._pipeline().import_kb(str(source))
                result = self._pipeline(failed_images=1).import_kb(str(source))
                kb_id = result["kb"]["id"]

                self.assertEqual(result["state"], "ready_with_warnings")
                self.assertEqual(result["summary"]["n_docs"], baseline["summary"]["n_docs"])
                self.assertEqual(result["summary"]["text_chunks"], baseline["summary"]["text_chunks"])
                self.assertEqual(
                    result["summary"]["image_chunks"],
                    baseline["summary"]["image_chunks"] - 1,
                )
                self.assertEqual(len(result["failures"]), 1)
                self.assertEqual(result["failures"][0]["stage"], "image_analysis")
                self.assertEqual(
                    sorted(item["status"] for item in result["documents"]),
                    ["succeeded", "succeeded_with_warnings"],
                )
                self.assertTrue(Path(config.active_root(kb_id)).is_dir())
                self.assertFalse(Path(config.staging_root(kb_id)).exists())

    def test_text_only_model_keeps_all_basic_image_references(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "paper-text-only-model"
            self._copy_paper(source)
            with mock.patch.object(config, "DATA_ROOT", str(Path(temp) / "data")), mock.patch.object(
                config, "CONFIG_PATH", str(Path(temp) / "kb.yaml")
            ):
                baseline = self._pipeline().import_kb(str(source))
                result = self._pipeline(skip_vision=True).import_kb(str(source))

                self.assertEqual(result["state"], "ready_with_warnings")
                self.assertEqual(
                    result["summary"]["text_chunks"],
                    baseline["summary"]["text_chunks"],
                )
                self.assertEqual(
                    result["summary"]["image_chunks"],
                    baseline["summary"]["image_chunks"],
                )
                self.assertTrue(result["failures"])
                self.assertTrue(all(
                    item["stage"] == "image_capability"
                    for item in result["failures"]
                ))


if __name__ == "__main__":
    unittest.main()
