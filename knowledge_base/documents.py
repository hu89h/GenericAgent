"""Document discovery, text extraction, and Markdown chunking for the KB.

This module deliberately has no knowledge of KB configuration, embeddings, or
Zvec.  Keeping document preparation pure makes it reusable by build and
read-only inspection paths without importing the indexing stack.
"""
from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass, field
from html.parser import HTMLParser


INDEX_SUBDIR = ".kb_index"
SUPPORTED_EXTS = {".md", ".markdown"}
MD_CHUNK_SIZES = tuple(
    int(x)
    for x in re.split(
        r"[,/ ]+", os.environ.get("GA_KB_MD_CHUNK_SIZES", "3072,768").strip()
    )
    if x.strip().isdigit()
) or (3072, 768)


def scan_documents(kb_path):
    """Return ``(relative path, absolute path, mtime, size)`` rows."""
    out = []
    for dirpath, dirnames, filenames in os.walk(kb_path):
        dirnames[:] = [d for d in dirnames if d != INDEX_SUBDIR and not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            absolute_path = os.path.join(dirpath, filename)
            try:
                stat = os.stat(absolute_path)
            except OSError:
                continue
            relative_path = os.path.relpath(absolute_path, kb_path).replace(os.sep, "/")
            out.append((relative_path, absolute_path, int(stat.st_mtime), stat.st_size))
    out.sort(key=lambda row: row[0])
    return out


def fingerprint(scanned):
    return {relative_path: {"mtime": mtime, "size": size}
            for relative_path, _absolute_path, mtime, size in scanned}


def chunking_meta():
    # Only the effective target size drives chunking (see chunk_document_records:
    # target_size = max(128, MD_CHUNK_SIZES[-1])). The leading sizes are legacy
    # from an abandoned two-level hierarchy and never affect output, so we
    # fingerprint the effective size alone -- configs that yield the same
    # target (e.g. "768" vs "1024,768") share the index instead of forcing a
    # spurious rebuild.
    return {
        "chunker": "markdown_sections_packer",
        "markdown_chunk_target_size": max(128, int(MD_CHUNK_SIZES[-1])),
        "markdown_parser": "ga_semantic_markdown_v3_balanced",
        "markdown_packer": "ga_structural_blocks_v4_correctness",
        "image_caption_binding": "ga_strict_adjacent_v3_explicit",
    }


def read_textfile(path):
    # latin-1 is deliberately absent: with errors="strict" it decodes any byte
    # sequence without raising, so it would mask real encoding problems as
    # mojibake and make the controlled utf-8/replace fallback below dead code.
    # KB sources are UTF-8 markdown from the document-conversion pipeline;
    # gb18030 covers legacy Chinese files. Anything else falls to utf-8/replace,
    # which surfaces corruption visibly (U+FFFD) instead of silently.
    for encoding in ("utf-8", "gb18030"):
        try:
            with open(path, encoding=encoding, errors="strict") as handle:
                return handle.read()
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception:
            break
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except Exception:
        return ""


def extract_text(path):
    """Read Markdown source text; other formats are outside this processing path."""
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_EXTS:
        return read_textfile(path)
    return ""


def _clean(text):
    """Normalize transport noise without changing Markdown semantics."""
    text = str(text or "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _chunk_prefix(header_path):
    header_path = re.sub(r"\s+", " ", str(header_path or "")).strip()
    if not header_path or header_path == "/":
        return ""
    return f"章节路径：{header_path}\n\n"


def _chunk_record(
    body,
    *,
    header_path="",
    image_occurrence_ids=None,
    content_type="prose",
    structure_id="",
    structure_title="",
    structure_part_index=0,
    structure_part_count=1,
    search_text="",
    structure_warning="",
):
    result = {
        "body": body,
        "header_path": str(header_path or ""),
        "_image_occurrence_ids": list(image_occurrence_ids or []),
        "content_type": str(content_type or "prose"),
        "structure_id": str(structure_id or ""),
        "structure_title": str(structure_title or ""),
        "structure_part_index": int(structure_part_index or 0),
        "structure_part_count": max(1, int(structure_part_count or 1)),
        "search_text": str(search_text or body or ""),
    }
    if structure_warning:
        result["_structure_warning"] = str(structure_warning)
    return result


_MD_HEADING_LINE_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def _markdown_sections(text):
    """Return ``(header path, section text)`` rows using Markdown headings."""
    lines = (text or "").splitlines()
    stack = []
    current_header = "/"
    buffer = []
    sections = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_header, body))

    for line in lines:
        match = _MD_HEADING_LINE_RE.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = re.sub(r"<[^>]+>", "", match.group(2))
            title = re.sub(
                r"^\s*\[Table\\?_?MaiN\]\s*",
                "",
                title,
                flags=re.IGNORECASE,
            )
            title = re.sub(r"\s+", " ", title.strip().strip("#").strip())
            stack[:] = stack[: level - 1]
            stack.append(title)
            current_header = "/" + "/".join(stack) + "/" if stack else "/"
            continue
        buffer.append(line)
    flush()
    if not sections and text.strip():
        return [("/", text.strip())]
    return sections


_MD_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\((?:[^()\\\s\r\n]|\\.|\([^()\r\n]*\))+"
    r"(?:\s+\"[^\"\r\n]*\")?\)"
)
_FENCE_RE = re.compile(r"^\s{0,3}(\`{3,}|~{3,})(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_LIST_ITEM_DETAIL_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-+*]|\d+[.)])\s+"
)
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$"
)
_FOOTNOTE_RE = re.compile(r"^\s*\[\^[^\]]+\]:\s*")
_HTML_STRUCTURAL_TAG_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<tag>table|pre|blockquote|ul|ol)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_UNCLOSED_START_RE = re.compile(
    r"<\s*(?P<tag>table|pre|blockquote|ul|ol)\b[^>]*>",
    re.IGNORECASE,
)
_TABLE_TITLE_RE = re.compile(
    r"^(?:(?:附?表)\s*(?:\d+(?:[.\-]\d+)*|[一二三四五六七八九十百]+|[A-Za-z]\d*)"
    r"(?:\s*[:：.\-]?\s*\S.*)?|Table\s+\d+(?:[.\-]\d+)*"
    r"(?:\s*[:：.\-]?\s*\S.*)?)$",
    re.IGNORECASE,
)
_FIGURE_CAPTION_RE = re.compile(
    r"^(?:(?:图|Figure|Fig\.?)\s*\d+(?:[.\-]\d+)*\b.*|注\s*[:：].*)$",
    re.IGNORECASE,
)
_MATH_BEGIN_RE = re.compile(
    r"^\s*\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?|cases|matrix|array)\}"
)


@dataclass
class _ProtectedBlock:
    kind: str
    raw: str
    occurrence_ids: list[int] = field(default_factory=list)
    warning: str = ""
    parsed: dict | None = None


@dataclass
class _TableCell:
    text: str
    rowspan: int = 1
    colspan: int = 1
    header: bool = False


class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_TableCell]] = []
        self.caption = ""
        self._row: list[_TableCell] | None = None
        self._cell: _TableCell | None = None
        self._cell_parts: list[str] = []
        self._in_caption = False
        self._caption_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = {str(key).lower(): value for key, value in attrs}
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            try:
                rowspan = max(1, int(attrs.get("rowspan") or 1))
            except (TypeError, ValueError):
                rowspan = 1
            try:
                colspan = max(1, int(attrs.get("colspan") or 1))
            except (TypeError, ValueError):
                colspan = 1
            self._cell = _TableCell("", rowspan, colspan, tag == "th")
            self._cell_parts = []
        elif tag == "caption":
            self._in_caption = True
            self._caption_parts = []
        elif tag == "br" and self._cell is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None:
            self._cell.text = _normalize_cell_text("".join(self._cell_parts))
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
            self._cell_parts = []
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag == "caption":
            self.caption = _normalize_cell_text("".join(self._caption_parts))
            self._in_caption = False

    def handle_data(self, data):
        if self._cell is not None:
            self._cell_parts.append(data)
        if self._in_caption:
            self._caption_parts.append(data)


class _HTMLTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, _attrs):
        if tag.lower() in {"br", "p", "div", "li", "tr"}:
            self.parts.append("\n")
        if tag.lower() == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag.lower() in {"p", "div", "li", "tr", "pre", "blockquote"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


class _HTMLListParser(HTMLParser):
    """Convert nested HTML lists to Markdown without separating child items."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.list_stack: list[dict] = []
        self.in_item = 0

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        if tag in {"ul", "ol"}:
            self.list_stack.append({"tag": tag, "counter": 0})
            return
        if tag == "li":
            depth = max(0, len(self.list_stack) - 1)
            if self.parts and not self.parts[-1].endswith("\n"):
                self.parts.append("\n")
            marker = "-"
            if self.list_stack and self.list_stack[-1]["tag"] == "ol":
                self.list_stack[-1]["counter"] += 1
                marker = f"{self.list_stack[-1]['counter']}."
            self.parts.append("  " * depth + marker + " ")
            self.in_item += 1
        elif tag == "br" and self.in_item:
            self.parts.append("\n" + "  " * max(0, len(self.list_stack)))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "li":
            self.in_item = max(0, self.in_item - 1)
            if self.parts and not self.parts[-1].endswith("\n"):
                self.parts.append("\n")
        elif tag in {"ul", "ol"} and self.list_stack:
            self.list_stack.pop()

    def handle_data(self, data):
        if self.in_item:
            value = re.sub(r"\s+", " ", data)
            content = value.strip()
            if not content:
                if (
                    value
                    and self.parts
                    and not self.parts[-1].endswith((" ", "\n"))
                ):
                    self.parts.append(" ")
                return
            if (
                value.startswith(" ")
                and self.parts
                and not self.parts[-1].endswith((" ", "\n"))
            ):
                self.parts.append(" ")
            self.parts.append(content)
            if value.endswith(" "):
                self.parts.append(" ")


class _ReplacementMap(dict):
    """Allocate collision-free, one-character private-use markers."""

    def __init__(self, source):
        super().__init__()
        self._source_chars = set(str(source or ""))
        self._next_codepoint = 0xE000

    def next_marker(self):
        while self._next_codepoint <= 0xF8FF:
            marker = chr(self._next_codepoint)
            self._next_codepoint += 1
            if marker not in self._source_chars and marker not in self:
                return marker
        raise ValueError("文档包含过多结构块，无法分配内部标记")


def _normalize_cell_text(value):
    value = re.sub(r"<br\s*/?>", " ", str(value or ""), flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _html_plain_text(value):
    parser = _HTMLTextParser()
    try:
        parser.feed(str(value or ""))
        parser.close()
        return re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()
    except Exception:
        return _normalize_cell_text(value)


def _html_list_markdown(value):
    parser = _HTMLListParser()
    try:
        parser.feed(str(value or ""))
        parser.close()
        return re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()
    except Exception:
        return _html_plain_text(value)


def _expand_table_rows(rows):
    expanded = []
    pending: dict[int, tuple[str, int, bool]] = {}
    for raw_row in rows:
        row: list[str] = []
        headers: list[bool] = []
        column = 0

        def fill_pending():
            nonlocal column
            while column in pending:
                text, remaining, is_header = pending[column]
                row.append(text)
                headers.append(is_header)
                if remaining <= 1:
                    pending.pop(column, None)
                else:
                    pending[column] = (text, remaining - 1, is_header)
                column += 1

        for cell in raw_row:
            fill_pending()
            for _ in range(cell.colspan):
                row.append(cell.text)
                headers.append(cell.header)
                if cell.rowspan > 1:
                    pending[column] = (cell.text, cell.rowspan - 1, cell.header)
                column += 1
        fill_pending()
        expanded.append((row, headers))
    width = max((len(row) for row, _headers in expanded), default=0)
    return [
        (row + [""] * (width - len(row)), headers + [False] * (width - len(headers)))
        for row, headers in expanded
    ]


def _parse_html_table(raw):
    parser = _HTMLTableParser()
    warning = ""
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        warning = "HTML 表格结构不完整，已使用安全文本恢复"
    expanded = _expand_table_rows(parser.rows)
    if not expanded:
        plain = _html_plain_text(raw)
        return {
            "title": "",
            "headers": [],
            "rows": [[plain]] if plain else [],
            "warning": warning or "HTML 表格无法解析，已使用安全文本恢复",
        }
    header_count = 0
    for _row, header_flags in expanded:
        if any(header_flags):
            header_count += 1
        else:
            break
    if not header_count:
        header_count = 1
    header_rows = [row for row, _flags in expanded[:header_count]]
    width = len(header_rows[0]) if header_rows else 0
    headers = []
    for column in range(width):
        values = []
        for row in header_rows:
            value = row[column].strip()
            if value and value not in values:
                values.append(value)
        headers.append(" / ".join(values))
    title = parser.caption.strip() or (headers[0] if headers else "")
    return {
        "title": title,
        "headers": headers,
        "rows": [row for row, _flags in expanded[header_count:]],
        "warning": warning,
    }


def _split_markdown_row(line):
    value = line.strip().strip("|")
    return [
        _normalize_cell_text(item.replace(r"\|", "|"))
        for item in re.split(r"(?<!\\)\|", value)
    ]


def _escape_markdown_cell(value):
    value = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\\", r"\\").replace("|", r"\|")
    return value.replace("\n", "<br>")


def _parse_markdown_table(raw):
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        return {"title": "", "headers": [], "rows": [], "warning": "Markdown 表格不完整"}
    headers = _split_markdown_row(lines[0])
    rows = [_split_markdown_row(line) for line in lines[2:]]
    width = len(headers)
    rows = [row[:width] + [""] * max(0, width - len(row)) for row in rows]
    return {
        "title": headers[0] if headers else "",
        "headers": headers,
        "rows": rows,
        "warning": "",
    }


def _markdown_table(headers, rows, *, title="", unit=""):
    headers = [_escape_markdown_cell(item) for item in (headers or [])]
    rows = [
        [_escape_markdown_cell(item) for item in row]
        for row in (rows or [])
    ]
    if not headers:
        return "\n".join(" | ".join(row) for row in rows).strip()
    lines = []
    if title and title not in headers:
        lines.append(f"表格：{title}")
    if unit:
        lines.append(unit if re.match(r"^单位\s*[:：]", unit) else f"单位：{unit}")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _split_table_block(block, target_size):
    parsed = block.parsed or (
        _parse_html_table(block.raw)
        if re.search(r"<table\b", block.raw, re.IGNORECASE)
        else _parse_markdown_table(block.raw)
    )
    headers = parsed["headers"]
    rows = parsed["rows"]
    title = parsed["title"]
    unit = parsed.get("unit") or ""
    warning = parsed["warning"] or block.warning
    if not rows:
        body = _markdown_table(headers, [], title=title, unit=unit) or _html_plain_text(block.raw)
        return [(body, title, warning)] if body else []
    expanded_rows = []
    for row in rows:
        if len(_markdown_table(headers, [row], title=title, unit=unit)) <= target_size:
            expanded_rows.append(row)
            continue
        cell_budget = max(80, target_size // max(1, len(row)))
        cell_parts = [
            _split_plain_block(cell, cell_budget) if len(cell) > cell_budget else [cell]
            for cell in row
        ]
        part_count = max((len(parts) for parts in cell_parts), default=1)
        for part_index in range(part_count):
            expanded_rows.append([
                parts[part_index] if len(parts) > 1 and part_index < len(parts) else (
                    parts[0] if len(parts) == 1 else ""
                )
                for parts in cell_parts
            ])
    rows = expanded_rows
    groups = []
    current = []
    for row in rows:
        candidate = _markdown_table(headers, current + [row], title=title, unit=unit)
        if current and len(candidate) > target_size:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
    if current:
        groups.append(current)
    return [
        (_markdown_table(headers, group, title=title, unit=unit), title, warning)
        for group in groups
    ]


def _marker(kind, value, replacements, *, occurrence_ids=None, warning=""):
    marker = replacements.next_marker()
    replacements[marker] = _ProtectedBlock(
        kind=kind.lower(),
        raw=value,
        occurrence_ids=list(dict.fromkeys(int(item) for item in (occurrence_ids or []))),
        warning=warning,
    )
    return marker


def _fenced_code_ranges(text):
    """Return source ranges occupied by Markdown fenced code blocks."""
    ranges = []
    opening = None
    offset = 0
    for line in str(text or "").splitlines(keepends=True):
        value = line.rstrip("\r\n")
        match = _FENCE_RE.match(value)
        if opening is None and match:
            opening = (offset, match.group(1)[0], len(match.group(1)))
        elif opening is not None:
            start, fence_char, fence_size = opening
            if re.match(
                r"^\s{0,3}" + re.escape(fence_char)
                + r"{" + str(fence_size) + r",}\s*$",
                value,
            ):
                ranges.append((start, offset + len(line)))
                opening = None
        offset += len(line)
    if opening is not None:
        ranges.append((opening[0], len(text)))
    return ranges


def _balanced_html_blocks(text):
    """Locate complete outer structural HTML blocks using nesting depth."""
    fenced_ranges = _fenced_code_ranges(text)
    range_index = 0
    stack = []
    outer_start = None
    outer_tag = ""
    blocks = []

    for match in _HTML_STRUCTURAL_TAG_RE.finditer(text):
        while (
            range_index < len(fenced_ranges)
            and fenced_ranges[range_index][1] <= match.start()
        ):
            range_index += 1
        if (
            range_index < len(fenced_ranges)
            and fenced_ranges[range_index][0] <= match.start() < fenced_ranges[range_index][1]
        ):
            continue

        tag = str(match.group("tag") or "").lower()
        closing = bool(match.group("closing"))
        self_closing = match.group(0).rstrip().endswith("/>")
        if not closing:
            if not stack:
                outer_start = match.start()
                outer_tag = tag
            if not self_closing:
                stack.append(tag)
            elif not stack and outer_start is not None:
                blocks.append((outer_start, match.end(), outer_tag))
                outer_start = None
                outer_tag = ""
            continue

        if not stack or tag not in stack:
            continue
        while stack:
            opened = stack.pop()
            if opened == tag:
                break
        if not stack and outer_start is not None:
            blocks.append((outer_start, match.end(), outer_tag))
            outer_start = None
            outer_tag = ""
    return blocks


def _protect_markdown_blocks(text, image_index=None):
    """Mask semantic Markdown/HTML structures without changing source offsets."""
    replacements = _ReplacementMap(text)
    masked = list(text)
    for start, end, tag in _balanced_html_blocks(text):
        raw = text[start:end]
        kind = {
            "table": "TABLE",
            "pre": "CODE",
            "blockquote": "QUOTE",
            "ul": "LIST",
            "ol": "LIST",
        }[tag]
        occurrence_ids = (
            image_index.occurrence_ids_between(start, end)
            if image_index is not None else []
        )
        marker = _marker(kind, raw, replacements, occurrence_ids=occurrence_ids)
        for position in range(start, end):
            if masked[position] not in "\r\n":
                masked[position] = " "
        masked[start] = marker

    text = "".join(masked)
    lines = text.splitlines()
    output = []
    index = 0

    def protect_images(value, base_offset):
        return _MD_IMAGE_RE.sub(
            lambda match: _marker(
                "FIGURE",
                match.group(0),
                replacements,
                occurrence_ids=image_index.occurrence_ids_between(
                    base_offset + match.start(), base_offset + match.end()
                ) if image_index is not None else [],
            ),
            value,
        )

    line_starts = []
    offset = 0
    for value in lines:
        line_starts.append(offset)
        offset += len(value) + 1

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped in replacements:
            output.append(stripped)
            index += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            fence_char = fence.group(1)[0]
            fence_size = len(fence.group(1))
            end = index + 1
            while end < len(lines):
                closing = re.match(
                    r"^\s{0,3}" + re.escape(fence_char)
                    + r"{" + str(fence_size) + r",}\s*$",
                    lines[end],
                )
                if closing:
                    end += 1
                    break
                end += 1
            block_start = line_starts[index]
            block_end = line_starts[end - 1] + len(lines[end - 1])
            output.append(_marker(
                "CODE",
                "\n".join(lines[index:end]),
                replacements,
                occurrence_ids=image_index.occurrence_ids_between(block_start, block_end)
                if image_index is not None else [],
            ))
            index = end
            continue

        if (
            index + 1 < len(lines)
            and "|" in line
            and _TABLE_SEPARATOR_RE.match(lines[index + 1])
        ):
            end = index + 2
            while end < len(lines) and lines[end].strip() and "|" in lines[end]:
                end += 1
            output.append(_marker("TABLE", "\n".join(lines[index:end]), replacements))
            index = end
            continue

        math_close = ""
        if stripped == "$$":
            math_close = "$$"
        elif stripped == r"\[":
            math_close = r"\]"
        else:
            math_begin = _MATH_BEGIN_RE.match(line)
            if math_begin:
                math_close = rf"\end{{{math_begin.group('env')}}}"
        if math_close:
            end = index + 1
            found_close = False
            while end < len(lines):
                if lines[end].strip() == math_close:
                    found_close = True
                    end += 1
                    break
                end += 1
            output.append(_marker(
                "EQUATION",
                "\n".join(lines[index:end]),
                replacements,
                warning="" if found_close else "公式定界符不完整",
            ))
            index = end
            continue

        if re.match(r"^\s{0,3}>", line):
            end = index + 1
            while end < len(lines) and (
                re.match(r"^\s{0,3}>", lines[end]) or not lines[end].strip()
            ):
                end += 1
            output.append(_marker("QUOTE", "\n".join(lines[index:end]), replacements))
            index = end
            continue

        if _FOOTNOTE_RE.match(line):
            end = index + 1
            while end < len(lines) and (
                not lines[end].strip() or lines[end][:1].isspace()
            ):
                end += 1
            output.append(_marker("NOTE", "\n".join(lines[index:end]), replacements))
            index = end
            continue

        if _LIST_ITEM_RE.match(line):
            end = index + 1
            while end < len(lines):
                candidate = lines[end]
                if _LIST_ITEM_RE.match(candidate) or (
                    candidate.strip() and candidate[:1].isspace()
                ):
                    end += 1
                    continue
                break
            output.append(_marker("LIST", "\n".join(lines[index:end]), replacements))
            index = end
            continue

        unclosed_html = _HTML_UNCLOSED_START_RE.search(line)
        if unclosed_html:
            tag = str(unclosed_html.group("tag") or "").lower()
            kind = {
                "table": "TABLE",
                "pre": "CODE",
                "blockquote": "QUOTE",
                "ul": "LIST",
                "ol": "LIST",
            }[tag]
            prefix = line[:unclosed_html.start()]
            if prefix.strip():
                output.append(protect_images(prefix, line_starts[index]))
            end = index + 1
            while (
                end < len(lines)
                and lines[end].strip()
                and not _MD_HEADING_LINE_RE.match(lines[end])
            ):
                end += 1
            raw_lines = [line[unclosed_html.start():], *lines[index + 1:end]]
            block_start = line_starts[index] + unclosed_html.start()
            block_end = line_starts[end - 1] + len(lines[end - 1])
            output.append(_marker(
                kind,
                "\n".join(raw_lines),
                replacements,
                occurrence_ids=image_index.occurrence_ids_between(block_start, block_end)
                if image_index is not None else [],
                warning="HTML 结构缺少完整结束标签，已使用安全恢复",
            ))
            index = end
            continue

        output.append(protect_images(line, line_starts[index]))
        index += 1
    return "\n".join(output), replacements


def _best_split_position(text, target):
    limit = min(len(text), target)
    for separator in ("\n\n", "\n", "。", "！", "？", ". ", "；", ";", " "):
        position = text.rfind(separator, 0, limit + 1)
        if position > max(0, limit // 3):
            return position + len(separator)
    return limit


def _split_plain_block(text, target_size):
    if len(text) <= target_size:
        return [text]
    overlap = min(max(0, target_size // 8), 128)
    chunks = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if len(remaining) <= target_size:
            chunks.append(remaining.strip())
            break
        cut = _best_split_position(remaining, target_size)
        piece = remaining[:cut].strip()
        if not piece:
            cut = target_size
            piece = remaining[:cut].strip()
        chunks.append(piece)
        next_start = start + cut - overlap
        start = max(start + 1, next_start)
    return [chunk for chunk in chunks if chunk]


def _split_by_lines(lines, target_size):
    groups = []
    current = []
    for line in lines:
        candidate = "\n".join(current + [line])
        if current and len(candidate) > target_size:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)
    return groups


def _split_code_block(raw, target_size):
    if re.search(r"<pre\b", raw, re.IGNORECASE):
        raw = "```\n" + _html_plain_text(raw) + "\n```"
    lines = raw.splitlines()
    fence = _FENCE_RE.match(lines[0] if lines else "")
    if not fence or len(raw) <= target_size:
        return [raw]
    opening = lines[0]
    has_close = len(lines) > 1 and bool(
        re.match(r"^\s*(?:`{3,}|~{3,})\s*$", lines[-1])
    )
    closing = lines[-1] if has_close else fence.group(1)
    inner = lines[1:-1] if has_close else lines[1:]
    budget = max(128, target_size - len(opening) - len(closing) - 2)
    return [
        opening + "\n" + "\n".join(group) + "\n" + closing
        for group in _split_by_lines(inner, budget)
    ]


def _split_list_block(raw, target_size):
    value = (
        _html_list_markdown(raw)
        if re.search(r"<(?:ul|ol)\b", raw, re.IGNORECASE)
        else raw
    )
    items = []
    current = []
    first_item = next(
        (_LIST_ITEM_DETAIL_RE.match(line) for line in value.splitlines()
         if _LIST_ITEM_DETAIL_RE.match(line)),
        None,
    )
    base_indent = len(first_item.group("indent").expandtabs(4)) if first_item else 0
    for line in value.splitlines():
        item = _LIST_ITEM_DETAIL_RE.match(line)
        indent = len(item.group("indent").expandtabs(4)) if item else -1
        if item and indent <= base_indent and current:
            items.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        items.append("\n".join(current).strip())
    return ["\n".join(group) for group in _split_by_lines(items, target_size)]


def _split_equation_block(raw, target_size):
    if len(raw) <= target_size:
        return [raw]
    lines = raw.splitlines()
    if len(lines) <= 2:
        return [raw]
    opening, closing = lines[0], lines[-1]
    budget = max(128, target_size - len(opening) - len(closing) - 2)
    return [
        opening + "\n" + "\n".join(group) + "\n" + closing
        for group in _split_by_lines(lines[1:-1], budget)
    ]


def _split_generic_structured(raw, target_size):
    value = _html_plain_text(raw) if re.search(r"<[^>]+>", raw) else raw
    paragraphs = [
        item.strip() for item in re.split(r"\n\s*\n", value) if item.strip()
    ]
    return [
        "\n\n".join(group)
        for group in _split_by_lines(paragraphs, target_size)
    ]


def _structure_title(kind, raw):
    if kind == "figure":
        match = _MD_IMAGE_RE.search(raw)
        if match:
            alt = re.match(r"!\[([^\]]*)\]", match.group(0))
            return re.sub(r"\s+", " ", alt.group(1)).strip() if alt else ""
    if kind == "note":
        match = re.match(r"\s*(\[\^[^\]]+\])", raw)
        return match.group(1) if match else ""
    return ""


def _block_parts(block, target_size):
    if block.kind == "table":
        return _split_table_block(block, target_size)
    if block.kind == "code":
        bodies = _split_code_block(block.raw, target_size)
    elif block.kind == "list":
        bodies = _split_list_block(block.raw, target_size)
    elif block.kind == "equation":
        bodies = _split_equation_block(block.raw, target_size)
    elif block.kind in {"quote", "note"}:
        bodies = _split_generic_structured(block.raw, target_size)
    else:
        bodies = [block.raw]
    title = _structure_title(block.kind, block.raw)
    return [(body, title, block.warning) for body in bodies if body.strip()]


def _section_tokens(body, replacements):
    tokens = []
    cursor = 0
    marker_pattern = re.compile(
        "|".join(re.escape(marker) for marker in replacements)
    ) if replacements else None
    matches = marker_pattern.finditer(body or "") if marker_pattern else ()
    for match in matches:
        before = body[cursor:match.start()].strip()
        if before:
            tokens.append(before)
        block = replacements.get(match.group(0))
        if block is not None:
            tokens.append(block)
        cursor = match.end()
    trailing = body[cursor:].strip()
    if trailing:
        tokens.append(trailing)

    for index, token in enumerate(tokens[:-1]):
        if not isinstance(token, _ProtectedBlock) or token.kind != "figure":
            continue
        following = tokens[index + 1]
        if not isinstance(following, str):
            continue
        parts = re.split(r"\n\s*\n", following, maxsplit=1)
        caption = parts[0].strip()
        if len(caption) <= 300 and _FIGURE_CAPTION_RE.match(caption):
            token.raw = token.raw.rstrip() + "\n" + caption
            tokens[index + 1] = parts[1].strip() if len(parts) > 1 else ""

    # MinerU and hand-written Markdown often place a table title and unit on
    # the lines immediately before the table instead of using ``<caption>``.
    # Bind only unambiguous metadata lines so ordinary prose remains prose.
    for index, token in enumerate(tokens):
        if not isinstance(token, _ProtectedBlock) or token.kind != "table" or index == 0:
            continue
        preceding = tokens[index - 1]
        if not isinstance(preceding, str):
            continue
        lines = preceding.rstrip().splitlines()
        metadata = []
        while lines and len(metadata) < 2:
            candidate = lines[-1].strip()
            if not candidate:
                lines.pop()
                continue
            if re.match(r"^单位\s*[:：]", candidate) or _TABLE_TITLE_RE.match(candidate):
                metadata.insert(0, candidate)
                lines.pop()
                continue
            break
        if not metadata:
            continue
        parsed = dict(_table_data(token))
        title = next(
            (value for value in metadata if not re.match(r"^单位\s*[:：]", value)),
            "",
        )
        unit = next(
            (value for value in metadata if re.match(r"^单位\s*[:：]", value)),
            "",
        )
        if title:
            parsed["title"] = title
        if unit:
            parsed["unit"] = unit
        token.parsed = parsed
        tokens[index - 1] = "\n".join(lines).rstrip()
    return [token for token in tokens if not isinstance(token, str) or token.strip()]


def _table_data(block):
    if block.parsed is not None:
        return block.parsed
    return (
        _parse_html_table(block.raw)
        if re.search(r"<table\b", block.raw, re.IGNORECASE)
        else _parse_markdown_table(block.raw)
    )


def _pack_semantic_blocks(blocks, replacements, target_size, *, file_name=""):
    records = []
    structure_ordinal = 0
    for header_path, body in blocks:
        prefix = _chunk_prefix(header_path)
        effective_target = max(128, target_size - len(prefix))
        tokens = _section_tokens(body, replacements)
        for token in tokens:
            if isinstance(token, str):
                for piece in _split_plain_block(token, effective_target):
                    value = prefix + piece
                    records.append(_chunk_record(
                        value,
                        header_path=header_path,
                        content_type="prose",
                        search_text=value,
                    ))
                continue
            structure_ordinal += 1
            raw_key = re.sub(r"\s+", " ", token.raw).strip()
            structure_id = hashlib.sha1(
                (
                    f"{file_name}\x00{header_path}\x00{token.kind}\x00"
                    f"{structure_ordinal}\x00{raw_key}"
                ).encode("utf-8")
            ).hexdigest()
            parts = _block_parts(token, effective_target)
            count = len(parts)
            for part_index, (piece, title, warning) in enumerate(parts):
                value = prefix + piece
                search_text = "\n".join(
                    item for item in (header_path.strip("/"), title, piece) if item
                )
                records.append(_chunk_record(
                    value,
                    header_path=header_path,
                    image_occurrence_ids=token.occurrence_ids,
                    content_type=token.kind,
                    structure_id=structure_id,
                    structure_title=title,
                    structure_part_index=part_index,
                    structure_part_count=count,
                    search_text=search_text,
                    structure_warning=warning,
                ))
    return records


def chunk_document_records(text, *, ext="", file_name="", image_index=None):
    ext = (ext or "").lower().lstrip(".")
    if ext not in {"md", "markdown"}:
        return []
    text = _clean(text)
    if not text:
        return []
    target_size = max(128, int(MD_CHUNK_SIZES[-1]))
    # Markers hide multi-line constructs so the heading scan sees only headings
    # and single-line marker rows; _markdown_sections then splits on headings
    # and always returns a non-empty result for non-empty text.
    protected, replacements = _protect_markdown_blocks(text, image_index)
    blocks = _markdown_sections(protected)
    return _pack_semantic_blocks(
        blocks,
        replacements,
        target_size,
        file_name=file_name,
    )


def chunk_document_text(text, *, ext="", file_name=""):
    return [record["body"] for record in chunk_document_records(text, ext=ext, file_name=file_name)]


__all__ = [
    "MD_CHUNK_SIZES",
    "SUPPORTED_EXTS",
    "chunk_document_records",
    "chunk_document_text",
    "chunking_meta",
    "extract_text",
    "fingerprint",
    "read_textfile",
    "scan_documents",
]
