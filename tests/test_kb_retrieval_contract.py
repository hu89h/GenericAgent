import json
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

    def _write_find_records(self):
        records = {
            "kb-a": [
                {
                    "kind": "text", "content_type": "prose",
                    "data_id": "kb-a::documents/a.md", "file_name": "documents/a.md",
                    "source_file_name": "a.pdf", "chunk_index": 1,
                    "body": "Dividend policy is disclosed here.",
                },
                {
                    "kind": "text", "content_type": "table",
                    "data_id": "kb-a::documents/a.md", "file_name": "documents/a.md",
                    "source_file_name": "a.pdf", "chunk_index": 2,
                    "body": "| 股利支付率 | 70% |", "structure_title": "分红表",
                },
                {
                    "kind": "image", "content_type": "figure",
                    "data_id": "kb-a::documents/a.md::image::1",
                    "source_data_id": "kb-a::documents/a.md",
                    "file_name": "documents/a.md", "source_file_name": "a.pdf",
                    "chunk_index": 0, "ref_key": "图1", "caption": "二维码下载入口",
                },
                {
                    "kind": "text", "content_type": "prose",
                    "data_id": "kb-a::documents/other.md", "file_name": "documents/other.md",
                    "source_file_name": "other.pdf", "chunk_index": 1,
                    "body": "No matching disclosure.",
                },
            ],
            "kb-b": [{
                "kind": "text", "content_type": "table",
                "data_id": "kb-b::documents/b.md", "file_name": "documents/b.md",
                "source_file_name": "b.pdf", "chunk_index": 4,
                "body": "股利支付率为 48%。",
            }],
        }
        for kb in self.retriever._registry.kbs:
            path = os.path.join(kb["path"], "records.jsonl")
            kb["records_path"] = path
            with open(path, "w", encoding="utf-8") as handle:
                for record in records[kb["id"]]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_literal_find_scans_every_scoped_record_without_embedding(self):
        self._write_find_records()

        result = self.retriever.find_terms(
            ["dividend", "股利支付率"], match="all", case_sensitive=False,
            scope_targets=[{
                "kb_id": "kb-a", "all_documents": False,
                "documents": [{"data_id": "kb-a::documents/a.md"}],
            }],
        )

        self.assertTrue(result["complete"])
        self.assertEqual(result["searched_documents"], 1)
        self.assertEqual([item["data_id"] for item in result["documents"]], [
            "kb-a::documents/a.md",
        ])
        self.assertEqual(result["documents"][0]["matched_terms"], [
            "dividend", "股利支付率",
        ])
        self.assertEqual(set(result["documents"][0]), {
            "data_id", "source_file_name", "matched_terms",
        })
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 0)

    def test_literal_find_matches_image_text_without_returning_locations(self):
        self._write_find_records()

        image_result = self.retriever.find_terms(["二维码"], kb_id="kb-a")
        self.assertTrue(image_result["complete"])
        self.assertEqual(image_result["documents"][0]["matched_terms"], ["二维码"])
        self.assertNotIn("occurrences", image_result["documents"][0])
        self.assertEqual(
            {item["data_id"] for item in image_result["unmatched_documents"]},
            {"kb-a::documents/other.md"},
        )

        os.remove(self.retriever._registry.kbs[1]["records_path"])
        broken = self.retriever.find_terms(["股利支付率"], kb_id="kb-b")
        self.assertFalse(broken["complete"])
        self.assertEqual(broken["warnings"], [{"error_code": "records_unavailable"}])

    def test_literal_find_rejects_more_than_fifty_documents_as_one_scope(self):
        kb = self.retriever._registry.kbs[0]
        path = os.path.join(kb["path"], "records.jsonl")
        kb["records_path"] = path
        with open(path, "w", encoding="utf-8") as handle:
            for index in range(51):
                handle.write(json.dumps({
                    "kind": "text", "content_type": "prose",
                    "data_id": f"kb-a::documents/{index}.md",
                    "file_name": f"documents/{index}.md",
                    "source_file_name": f"{index}.pdf",
                    "chunk_index": 0, "body": "needle",
                }) + "\n")

        result = self.retriever.find_terms(["needle"], kb_id="kb-a")
        self.assertEqual(result["error_code"], "scope_too_broad")
        self.assertEqual(result["searched_documents"], 51)

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
                {"kb_id": "kb-a", "documents": [{"data_id": "kb-a::documents/a.md"}]},
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
        content = self.retriever.read_content(
            data_id=hit["data_id"], chunk_index=hit["chunk_index"], kb_id="kb-a"
        )
        self.assertIn("真实正文内容", content["content"])

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

    def test_selection_does_not_report_missing_unselected_knowledge_base(self):
        missing_root = os.path.join(self.temp.name, "kb-missing")
        self.retriever._registry.kbs.append({
            "id": "kb-missing",
            "name": "kb-missing",
            "path": missing_root,
            "exists": False,
        })
        self.retriever._search_one_zvec = lambda *_args, **_kwargs: [
            _text_hit("kb-a", "documents/a.md")
        ]

        result = self.retriever.search(
            "scoped",
            mode="vector",
            scope_targets=[{
                "kb_id": "kb-a",
                "all_documents": False,
                "documents": [{"data_id": "kb-a::documents/a.md"}],
            }],
        )

        self.assertTrue(result["results"])
        self.assertEqual(result["diagnostics"], [])

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

    def test_public_evidence_types_strictly_separate_text_table_and_images(self):
        records = [
            {
                "kind": "text", "content_type": "prose",
                "data_id": "kb-a::documents/a.md", "chunk_index": 0,
                "file_name": "documents/a.md", "title": "a.pdf", "body": "正文",
            },
            {
                "kind": "text", "content_type": "figure",
                "data_id": "kb-a::documents/a.md", "chunk_index": 1,
                "file_name": "documents/a.md", "title": "a.pdf", "body": "文本 figure",
            },
            {
                "kind": "text", "content_type": "table",
                "data_id": "kb-a::documents/a.md", "chunk_index": 2,
                "file_name": "documents/a.md", "title": "a.pdf", "body": "| 表格 |",
            },
            {
                "kind": "image", "content_type": "figure",
                "data_id": "kb-a::documents/a.md::image::locator", "chunk_index": 0,
                "file_name": "documents/a.md", "title": "图1", "body": "图题和邻近正文",
                "ref_key": "图1", "image_id": "locator",
            },
            {
                "kind": "image", "content_type": "table",
                "data_id": "kb-a::documents/a.md::image::table", "chunk_index": 0,
                "file_name": "documents/a.md", "title": "表1", "body": "图片表格",
                "ref_key": "表1", "image_id": "table", "table_markdown": "| 指标 | 值 |",
            },
        ]
        captured_filters = []

        class Collection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def query(*_args, **kwargs):
                captured_filters.append(kwargs.get("filter"))
                return [SimpleNamespace(fields=item, score=index / 10) for index, item in enumerate(records)]

        kb = self.retriever._registry.kbs[0]
        self.retriever._zvec_open = lambda _path: Collection()
        self.retriever._require_zvec = lambda: SimpleNamespace(
            Query=lambda *_args, **_kwargs: object()
        )

        def kinds(evidence_types):
            hits = self.retriever._search_one_zvec(
                kb, "query", 10, 120, file_name=None, title=None,
                query_vector=[1.0], evidence_types=evidence_types,
            )
            return {hit["data_id"] for hit in hits}

        self.assertEqual(kinds(["text"]), {
            "kb-a::documents/a.md",
        })
        # The two text structures share a document data_id but remain text;
        # most importantly, neither image record crosses this boundary.
        self.assertNotIn("kind = 'image'", captured_filters[-1])
        self.assertEqual(kinds(["table"]), {
            "kb-a::documents/a.md",
            "kb-a::documents/a.md::image::table",
        })
        self.assertIn("table_markdown != ''", captured_filters[-1])
        self.assertEqual(kinds(["image"]), {
            "kb-a::documents/a.md::image::locator",
            "kb-a::documents/a.md::image::table",
        })
        self.assertEqual(kinds(None), {
            "kb-a::documents/a.md",
            "kb-a::documents/a.md::image::locator",
            "kb-a::documents/a.md::image::table",
        })

    def test_exact_image_only_is_a_hard_constraint_without_embedding(self):
        def exact(kb, query, *_args, **_kwargs):
            return [{
                **_text_hit(kb["id"], f"documents/{kb['id']}.md"),
                "kind": "image",
                "data_id": f"{kb['id']}::documents/{kb['id']}.md::image::81",
                "ref_key": "图81",
                "score_type": "ref_exact",
            }]

        self.retriever._search_exact_image_refs = exact
        self.retriever._search_one_zvec = lambda *_args, **_kwargs: self.fail(
            "exact image lookup must not run dense retrieval"
        )
        self.retriever._search_one_zvec_sparse = lambda *_args, **_kwargs: self.fail(
            "exact image lookup must not run sparse retrieval"
        )

        result = self.retriever.search(
            "讲讲图81", mode="rrf", top_k=1, evidence_types=["image"]
        )

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual({item["ref_key"] for item in result["results"]}, {"图81"})
        self.assertEqual(result["exact_image_references"], {
            "requested": ["图81"], "matched": ["图81"], "missing": [],
            "ambiguous": ["图81"],
        })
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 0)

    def test_missing_exact_image_returns_empty_without_semantic_fill(self):
        self.retriever._search_exact_image_refs = lambda *_args, **_kwargs: []

        result = self.retriever.search(
            "查看图999", mode="vector", evidence_types=["image"]
        )

        self.assertEqual(result["results"], [])
        self.assertEqual(result["exact_image_references"], {
            "requested": ["图999"], "matched": [], "missing": ["图999"],
            "ambiguous": [],
        })
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 0)

    def test_multiple_exact_image_numbers_are_all_enforced(self):
        captured = {}

        def image_rows(_kb, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "kind": "image", "content_type": "figure",
                    "data_id": f"kb-a::documents/a.md::image::{number}",
                    "file_name": "documents/a.md", "title": number,
                    "ref_key": number, "image_id": number, "body": number,
                }
                for number in ("图79", "图81", "图82")
            ]

        self.retriever._query_image_rows = image_rows
        result = self.retriever.search(
            "对比图79和图81", kb_id="kb-a", mode="sparse",
            evidence_types=["image"],
        )

        self.assertEqual(captured["ref_keys"], ["图79", "图81"])
        self.assertEqual(
            {item["ref_key"] for item in result["results"]},
            {"图79", "图81"},
        )
        self.assertEqual(result["exact_image_references"], {
            "requested": ["图79", "图81"],
            "matched": ["图79", "图81"],
            "missing": [],
            "ambiguous": [],
        })
        self.assertEqual(self.index.dense_calls, 0)
        self.assertEqual(self.index.sparse_calls, 0)

    def test_mixed_exact_reference_keeps_text_but_filters_other_images(self):
        exact = {
            **_text_hit("kb-a", "documents/a.md"),
            "kind": "image",
            "data_id": "kb-a::documents/a.md::image::81",
            "ref_key": "图81",
            "score_type": "ref_exact",
        }
        text = _text_hit("kb-a", "documents/a.md")
        image_81 = {**exact, "score_type": "zvec", "score": 0.1}
        image_79 = {
            **exact,
            "data_id": "kb-a::documents/a.md::image::79",
            "ref_key": "图79",
            "score_type": "zvec",
            "score": 0.05,
        }
        self.retriever._search_exact_image_refs = lambda *_args, **_kwargs: [exact]
        self.retriever._search_one_zvec = lambda *_args, **_kwargs: [image_79, text, image_81]

        result = self.retriever.search(
            "结合正文讲讲图81", mode="vector", top_k=5,
            evidence_types=["text", "image"],
        )

        images = [item for item in result["results"] if item.get("kind") == "image"]
        self.assertEqual({item["ref_key"] for item in images}, {"图81"})
        self.assertTrue(any(item.get("kind") != "image" for item in result["results"]))
        self.assertEqual(self.index.dense_calls, 1)

    def test_table_read_continuation_starts_at_requested_part_without_rewinding(self):
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

        self.assertEqual(result["start_chunk_index"], 5)
        self.assertEqual(result["end_chunk_index"], 5)
        self.assertNotIn("收入", result["content"])
        self.assertIn("资产负债率", result["content"])
        self.assertIn("22.3%", result["content"])
        self.assertFalse(result["continuation"]["has_more"])
        self.assertIsNone(result["continuation"]["next_chunk_index"])
        self.assertEqual(
            set(result["continuation"]), {"has_more", "next_chunk_index"}
        )

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
        self.assertIsNone(result["continuation"]["next_chunk_index"])
        self.assertGreater(result["continuation"]["required_max_chars"], 500)
        self.assertEqual(
            set(result["continuation"]),
            {"has_more", "next_chunk_index", "required_max_chars"},
        )

    def test_dense_channel_is_ranked_globally_across_knowledge_bases(self):
        self.retriever._search_exact_image_refs = lambda *_args, **_kwargs: []

        def dense(kb, *_args, **_kwargs):
            hit = _text_hit(kb["id"], f"documents/{kb['id']}.md")
            # Dense Zvec scores are cosine distances: lower is better.
            hit["score"] = 0.8 if kb["id"] == "kb-a" else 0.2
            return [hit]

        self.retriever._search_one_zvec = dense
        first = self.retriever.search("alpha", mode="vector", top_k=2)["results"]
        self.retriever._registry.kbs.reverse()
        second = self.retriever.search("alpha", mode="vector", top_k=2)["results"]

        self.assertEqual([item["kb_id"] for item in first], ["kb-b", "kb-a"])
        self.assertEqual(
            [item["data_id"] for item in first],
            [item["data_id"] for item in second],
        )
        self.assertEqual(
            [item["channel_ranks"]["vector"] for item in first], [1, 2]
        )

    def test_equal_exact_references_receive_the_same_global_rank(self):
        self.retriever._search_one_zvec = lambda *_args, **_kwargs: []

        def exact(kb, *_args, **_kwargs):
            hit = _text_hit(kb["id"], f"documents/{kb['id']}.md")
            hit.update({
                "kind": "image",
                "score_type": "ref_exact",
                "data_id": f"{kb['id']}::documents/{kb['id']}.md::image::1",
            })
            return [hit]

        self.retriever._search_exact_image_refs = exact
        results = self.retriever.search("图1", mode="vector", top_k=2)["results"]

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [item["channel_ranks"]["ref_exact"] for item in results], [1, 1]
        )
        self.assertEqual(results[0]["score"], results[1]["score"])

    def test_structured_search_hit_exposes_the_first_part_as_read_anchor(self):
        fields = {
            "data_id": "kb-a::documents/a.md",
            "chunk_index": 10,
            "kind": "text",
            "file_name": "documents/a.md",
            "title": "a.pdf",
            "body": "| 指标 | 数值 |",
            "content_type": "table",
            "structure_id": "table-1",
            "structure_part_index": 2,
            "structure_part_count": 4,
        }
        other_part = {
            **fields,
            "chunk_index": 9,
            "structure_part_index": 1,
            "body": "| 其他指标 | 数值 |",
        }

        class Collection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def query(*_args, **_kwargs):
                return [
                    SimpleNamespace(fields=fields, score=0.1),
                    SimpleNamespace(fields=other_part, score=0.2),
                ]

        kb = self.retriever._registry.kbs[0]
        self.retriever._zvec_open = lambda _path: Collection()
        self.retriever._require_zvec = lambda: SimpleNamespace(
            Query=lambda *_args, **_kwargs: object()
        )
        hits = self.retriever._search_one_zvec(
            kb, "指标", 5, 120,
            file_name=None, title=None, query_vector=[1.0],
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk_index"], 8)
        self.assertEqual(hits[0]["structure_part_index"], 0)

    def test_chunk_page_fetches_only_requested_window_plus_one(self):
        requested = []
        data_id = "kb-a::documents/a.md"

        class Collection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def fetch(ids, **_kwargs):
                requested.extend(ids)
                return {
                    doc_id: SimpleNamespace(fields={
                        "data_id": data_id,
                        "chunk_index": index,
                        "title": "a.pdf",
                        "file_name": "documents/a.md",
                        "body": f"第 {index} 段",
                    })
                    for index, doc_id in zip(range(500, 504), ids)
                }

        self.retriever._zvec_open = lambda _path: Collection()
        page = self.retriever.list_chunks(
            data_id=data_id, kb_id="kb-a", offset=500, limit=3
        )

        self.assertEqual(len(requested), 4)
        self.assertTrue(all(":50" in doc_id for doc_id in requested))
        self.assertEqual(page["offset"], 500)
        self.assertEqual(page["returned"], 3)
        self.assertTrue(page["has_more"])
        self.assertEqual(page["next_offset"], 503)
        self.assertNotIn("n_chunks", page)

    def test_ambiguous_image_reference_returns_only_public_candidate_fields(self):
        def image_rows(kb, **_kwargs):
            return [{
                "kind": "image",
                "data_id": f"{kb['id']}::documents/{kb['id']}.md::image::1",
                "file_name": f"documents/{kb['id']}.md",
                "source_file_name": f"{kb['id']}.pdf",
                "image_id": "image-1",
                "ref_key": "图1",
                "display_label": f"图1 {kb['id']}",
                "image_path": f"processed/{kb['id']}/image.png",
            }]

        self.retriever._query_image_rows = image_rows
        result = self.retriever.read_image(ref_key="图1")

        self.assertEqual(result["error_code"], "image_ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(
            set(result["candidates"][0]),
            {"data_id", "ref_key", "display_label", "source_hint"},
        )
        self.assertNotIn("processed/", str(result["candidates"]))
        self.assertNotIn("kb_id", str(result["candidates"]))

    def test_image_id_and_reference_must_resolve_to_the_same_asset(self):
        data_id = "kb-a::documents/a.md::image::1"
        self.retriever._fetch_doc = lambda _kb, _data_id, _chunk_index: SimpleNamespace(
            fields={"kind": "image", "data_id": data_id, "ref_key": "图1"}
        )

        result = self.retriever.read_image(
            data_id=data_id, ref_key="图2", kb_id="kb-a"
        )

        self.assertEqual(result["error_code"], "image_target_conflict")


if __name__ == "__main__":
    unittest.main()
