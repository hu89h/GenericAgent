import os
import tempfile
import unittest
from unittest import mock

from knowledge_base import documents
from knowledge_base.build import RecordBuilder


class StructuredDocumentChunkingTests(unittest.TestCase):
    """MinerU emits semantic blocks that must remain usable after chunking."""

    def _chunks(self, markdown, target=220):
        with mock.patch.object(documents, "MD_CHUNK_SIZES", (target,)):
            return documents.chunk_document_records(
                markdown,
                ext="md",
                file_name="report.md",
            )

    def test_single_line_html_table_repeats_headers_for_every_part(self):
        rows = "".join(
            f"<tr><td>指标{i}</td><td>{i}.1</td><td>{i}.2</td></tr>"
            for i in range(12)
        )
        markdown = (
            "# 财务数据\n"
            "<table><tr><td>重要财务指标</td><td>2020E</td><td>2021E</td></tr>"
            f"{rows}</table>"
        )

        chunks = [item for item in self._chunks(markdown) if item["content_type"] == "table"]

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("2020E" in item["body"] for item in chunks))
        self.assertTrue(all("2021E" in item["body"] for item in chunks))
        self.assertEqual({item["structure_id"] for item in chunks}.__len__(), 1)
        self.assertEqual(
            [item["structure_part_index"] for item in chunks],
            list(range(len(chunks))),
        )
        self.assertTrue(all(item["structure_part_count"] == len(chunks) for item in chunks))
        self.assertFalse(any("<td>" in item["body"] for item in chunks))

    def test_html_rowspan_and_colspan_keep_year_metric_value_relationship(self):
        markdown = """# 预测
<table>
<tr><th rowspan="2">指标</th><th colspan="2">预测值</th></tr>
<tr><th>2020E</th><th>2021E</th></tr>
<tr><td>资产负债率</td><td>22.3%</td><td>23.4%</td></tr>
</table>
"""

        table = next(item for item in self._chunks(markdown) if item["content_type"] == "table")

        self.assertIn("指标", table["body"])
        self.assertIn("预测值 / 2020E", table["body"])
        self.assertIn("资产负债率", table["body"])
        self.assertIn("22.3%", table["body"])

    def test_markdown_and_html_tables_share_the_same_structure_contract(self):
        html_chunks = self._chunks(
            "<table><tr><td>指标</td><td>2020E</td></tr>"
            "<tr><td>收入</td><td>100</td></tr></table>"
        )
        markdown_chunks = self._chunks(
            "| 指标 | 2020E |\n| --- | --- |\n| 收入 | 100 |"
        )

        html_table = next(item for item in html_chunks if item["content_type"] == "table")
        markdown_table = next(item for item in markdown_chunks if item["content_type"] == "table")
        for field in ("content_type", "structure_title", "structure_part_count"):
            self.assertEqual(html_table[field], markdown_table[field])
        self.assertIn("| 收入 | 100 |", html_table["body"])
        self.assertIn("| 收入 | 100 |", markdown_table["body"])

    def test_adjacent_matching_tables_remain_independent_structures(self):
        markdown = (
            "<table><tr><td>指标</td><td>2020E</td></tr>"
            "<tr><td>收入</td><td>100</td></tr></table>\n\n"
            "<table><tr><td>指标</td><td>2020E</td></tr>"
            "<tr><td>利润</td><td>20</td></tr></table>"
        )

        tables = [item for item in self._chunks(markdown) if item["content_type"] == "table"]

        self.assertEqual(len(tables), 2)
        self.assertEqual(len({item["structure_id"] for item in tables}), 2)
        combined = "\n".join(item["body"] for item in tables)
        self.assertIn("收入", combined)
        self.assertIn("利润", combined)

    def test_adjacent_markdown_tables_remain_independent_structures(self):
        markdown = """| 指标 | 2020E |
| --- | --- |
| 收入 | 100 |

| 指标 | 2020E |
| --- | --- |
| 利润 | 20 |
"""

        tables = [item for item in self._chunks(markdown)
                  if item["content_type"] == "table"]

        self.assertEqual(len(tables), 2)
        self.assertEqual(len({item["structure_id"] for item in tables}), 2)

    def test_nested_html_and_markdown_lists_keep_children_with_their_parent(self):
        markdown = """<ul><li>父项 <strong>说明</strong><ul><li>子项</li></ul></li><li>第二项</li></ul>

- Markdown 父项
    - Markdown 子项
- Markdown 第二项
"""

        lists = [item for item in self._chunks(markdown, target=128)
                 if item["content_type"] == "list"]

        self.assertEqual(len(lists), 2)
        self.assertIn("- 父项 说明", lists[0]["body"])
        self.assertIn("  - 子项", lists[0]["body"])
        self.assertNotIn("</li>", lists[0]["body"])
        self.assertIn("Markdown 父项\n    - Markdown 子项", lists[1]["body"])

    def test_ordinary_prose_starting_with_biao_is_not_used_as_table_title(self):
        markdown = """表明公司收入持续增长。
| 指标 | 2020E |
| --- | --- |
| 收入 | 100 |
"""

        records = self._chunks(markdown)
        prose = next(item for item in records if item["content_type"] == "prose")
        table = next(item for item in records if item["content_type"] == "table")

        self.assertIn("表明公司收入持续增长", prose["body"])
        self.assertNotEqual(table["structure_title"], "表明公司收入持续增长。")

    def test_markdown_table_reescapes_literal_pipes_and_backslashes(self):
        markdown = (
            "| 名称 | 值 |\n| --- | --- |\n"
            r"| A\|B | C\\D |"
        )

        table = next(item for item in self._chunks(markdown)
                     if item["content_type"] == "table")

        self.assertIn(r"A\|B", table["body"])
        self.assertIn(r"C\\\\D", table["body"])

    def test_balanced_html_mask_keeps_following_image_source_offsets(self):
        markdown = "<ul><li>短项</li></ul> ![图1](assets/figure.png)"
        image_start = markdown.index("![")
        image_end = len(markdown)

        class RecordingImageIndex:
            def __init__(self):
                self.calls = []

            def occurrence_ids_between(self, start, end):
                self.calls.append((start, end))
                return [7] if (start, end) == (image_start, image_end) else []

        image_index = RecordingImageIndex()
        records = documents.chunk_document_records(
            markdown,
            ext="md",
            file_name="offsets.md",
            image_index=image_index,
        )
        figure = next(item for item in records if item["content_type"] == "figure")

        self.assertIn((image_start, image_end), image_index.calls)
        self.assertEqual(figure["_image_occurrence_ids"], [7])

    def test_code_list_equation_quote_note_and_figure_have_semantic_types(self):
        markdown = r"""# 结构
```python
print("first")
print("second")
```

- 第一项
  - 子项
- 第二项

$$
x = y + z
$$

> 引用第一段
> 引用第二段

[^1]: 注释正文

![图1](assets/figure.png)

图1 处理流程
"""

        chunks = self._chunks(markdown, target=160)
        types = {item["content_type"] for item in chunks}

        self.assertTrue({"code", "list", "equation", "quote", "note", "figure"} <= types)
        code = next(item for item in chunks if item["content_type"] == "code")
        self.assertTrue(code["body"].rstrip().endswith("```"))
        figure = next(item for item in chunks if item["content_type"] == "figure")
        self.assertIn("图1 处理流程", figure["body"])

    def test_multiple_images_on_one_line_remain_separate_figures(self):
        records = documents.chunk_document_records(
            "![](assets/one.png) ![第二张](assets/two.webp)",
            ext="md",
            file_name="multi-image.md",
        )

        figures = [item for item in records if item["content_type"] == "figure"]
        self.assertEqual(len(figures), 2)
        self.assertIn("one.png", figures[0]["body"])
        self.assertIn("two.webp", figures[1]["body"])

    def test_adjacent_table_title_and_unit_are_bound_to_the_table(self):
        markdown = """表 2 重要财务指标
单位：%
| 指标 | 2020E |
| --- | --- |
| 资产负债率 | 22.3 |
"""
        records = documents.chunk_document_records(
            markdown,
            ext="md",
            file_name="captioned-table.md",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["content_type"], "table")
        self.assertEqual(records[0]["structure_title"], "表 2 重要财务指标")
        self.assertIn("单位：%", records[0]["body"])
        self.assertIn("资产负债率", records[0]["search_text"])

    def test_table_keeps_adjacent_prose_without_promoting_it_to_metadata(self):
        markdown = """Performance results (ms)

<table><tr><td>Case</td><td>Value</td></tr><tr><td>A</td><td>12</td></tr></table>

Source: benchmark laboratory
"""

        records = self._chunks(markdown)
        table = next(item for item in records if item["content_type"] == "table")
        prose = [item["body"] for item in records if item["content_type"] == "prose"]

        self.assertEqual(table["structure_title"], "Case")
        self.assertIn("[相邻上文]\nPerformance results (ms)", table["body"])
        self.assertIn("[表格正文]\n| Case | Value |", table["body"])
        self.assertIn("[相邻下文]\nSource: benchmark laboratory", table["body"])
        self.assertIn("Performance results (ms)", table["search_text"])
        self.assertTrue(any("Performance results (ms)" in item for item in prose))
        self.assertTrue(any("Source: benchmark laboratory" in item for item in prose))

    def test_table_context_does_not_cross_an_adjacent_structure(self):
        markdown = (
            "before first\n\n"
            "<table><tr><td>A</td></tr><tr><td>1</td></tr></table>\n\n"
            "<table><tr><td>B</td></tr><tr><td>2</td></tr></table>\n\n"
            "after second"
        )

        tables = [
            item for item in self._chunks(markdown)
            if item["content_type"] == "table"
        ]

        self.assertEqual(len(tables), 2)
        self.assertIn("before first", tables[0]["body"])
        self.assertNotIn("after second", tables[0]["body"])
        self.assertNotIn("before first", tables[1]["body"])
        self.assertIn("after second", tables[1]["body"])

    def test_table_context_is_bounded_and_repeated_for_every_part(self):
        rows = "".join(
            f"<tr><td>row-{index}</td><td>{index}</td></tr>"
            for index in range(20)
        )
        markdown = (
            "BEGIN-" + "a" * 500 + "-END\n\n"
            "<table><tr><td>Item</td><td>Value</td></tr>"
            + rows + "</table>\n\n"
            "START-" + "b" * 500 + "-FINISH"
        )

        tables = [
            item for item in self._chunks(markdown, target=180)
            if item["content_type"] == "table"
        ]

        self.assertGreater(len(tables), 1)
        for table in tables:
            self.assertIn("[相邻上文]", table["body"])
            self.assertIn("…[相邻上文已截断]", table["body"])
            self.assertNotIn("BEGIN-", table["body"])
            self.assertIn("-END", table["body"])
            self.assertIn("[相邻下文]", table["body"])
            self.assertIn("…[相邻下文已截断]", table["body"])
            self.assertIn("START-", table["body"])
            self.assertNotIn("-FINISH", table["body"])

    def test_malformed_html_table_degrades_without_exposing_half_tags(self):
        markdown = """# 表格
<table>
<tr><td>指标</td><td>2020E</td></tr>
<tr><td>收入</td><td>100</td></tr>
"""

        table = next(item for item in self._chunks(markdown) if item["content_type"] == "table")

        self.assertNotIn("<td>", table["body"])
        self.assertNotIn("<table>", table["body"])
        self.assertIn("指标", table["body"])
        self.assertIn("收入", table["body"])
        self.assertIn("_structure_warning", table)

    def test_safe_structure_recovery_is_not_reported_as_failed_import(self):
        class Assets:
            @staticmethod
            def build_document_index(_text):
                return None

            @staticmethod
            def analyze_image_jobs(_kb, _jobs, _log, **_kwargs):
                return {}

            @staticmethod
            def image_source_fingerprint(_path, _scanned, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "report.md")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("<table><tr><td>指标</td><td>2020E</td></tr>")
            messages = []

            result = RecordBuilder(assets=Assets()).build(
                {"id": "kb-test", "path": temp},
                {"files": []},
                logfn=messages.append,
            )

        self.assertFalse(result.failures)
        self.assertTrue(result.records)
        self.assertTrue(any("结构已安全恢复" in item for item in messages))


if __name__ == "__main__":
    unittest.main()
