import unittest
from unittest import mock

from knowledge_base import documents


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

    def test_adjacent_matching_tables_are_one_continuing_structure(self):
        markdown = (
            "<table><tr><td>指标</td><td>2020E</td></tr>"
            "<tr><td>收入</td><td>100</td></tr></table>\n\n"
            "<table><tr><td>指标</td><td>2020E</td></tr>"
            "<tr><td>利润</td><td>20</td></tr></table>"
        )

        tables = [item for item in self._chunks(markdown) if item["content_type"] == "table"]

        self.assertEqual({item["structure_id"] for item in tables}.__len__(), 1)
        combined = "\n".join(item["body"] for item in tables)
        self.assertIn("收入", combined)
        self.assertIn("利润", combined)

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

    def test_malformed_html_table_degrades_without_exposing_half_tags(self):
        markdown = "# 表格\n<table><tr><td>指标</td><td>2020E</td></tr>"

        table = next(item for item in self._chunks(markdown) if item["content_type"] == "table")

        self.assertNotIn("<td>", table["body"])
        self.assertNotIn("<table>", table["body"])
        self.assertIn("指标", table["body"])
        self.assertIn("_structure_warning", table)


if __name__ == "__main__":
    unittest.main()
