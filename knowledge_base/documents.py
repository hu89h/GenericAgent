"""Document discovery, text extraction, and Markdown chunking for the KB.

This module deliberately has no knowledge of KB configuration, embeddings, or
Zvec.  Keeping document preparation pure makes it reusable by build and
read-only inspection paths without importing the indexing stack.
"""
from __future__ import annotations

import os
import re


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
        "markdown_parser": "ga_markdown_sections_v1",
        "markdown_packer": "ga_structural_blocks_v2_image_occurrences",
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


def _chunk_record(body, *, header_path="", image_occurrence_ids=None):
    return {
        "body": body,
        "header_path": str(header_path or ""),
        "_image_occurrence_ids": list(image_occurrence_ids or []),
    }


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
            buffer = [line]
            level = len(match.group(1))
            title = re.sub(r"\s+", " ", match.group(2).strip().strip("#").strip())
            stack[:] = stack[: level - 1]
            stack.append(title)
            current_header = "/" + "/".join(stack) + "/" if stack else "/"
            continue
        buffer.append(line)
    flush()
    if not sections and text.strip():
        return [("/", text.strip())]
    return sections


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^\r\n]+\)")
_MARKER_RE = re.compile(r"GAKB(?:IMAGE|TABLE|CODE|LIST)\d{6}")
_IMAGE_MARKER_RE = re.compile(r"GAKBIMAGE\d{6}")
_FENCE_RE = re.compile(r"^\s{0,3}(\`{3,}|~{3,})(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$"
)


def _marker(kind, value, replacements, marker_occurrences=None, occurrence_ids=None):
    marker = f"GAKB{kind}{len(replacements):06d}"
    replacements[marker] = value
    if marker_occurrences is not None and occurrence_ids:
        marker_occurrences[marker] = list(dict.fromkeys(int(item) for item in occurrence_ids))
    return marker


def _protect_markdown_blocks(text, image_index=None):
    """Collapse tables/code/lists/images into single-line markers before chunking.

    Section splitting works on headings alone, so multi-line constructs are
    hidden behind markers to keep them off the heading scan and intact as one
    unit.  The originals are restored after packing: the image index resolves
    those paths later and the source reader displays the original formatting.
    """
    lines = text.splitlines()
    replacements = {}
    marker_occurrences = {}
    output = []
    index = 0

    def protect_images(value, base_offset):
        return _MD_IMAGE_RE.sub(
            lambda match: _marker(
                "IMAGE",
                match.group(0),
                replacements,
                marker_occurrences,
                image_index.occurrence_ids_between(
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
                marker_occurrences,
                image_index.occurrence_ids_between(block_start, block_end)
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
            while end < len(lines) and ("|" in lines[end] or not lines[end].strip()):
                if (
                    not lines[end].strip()
                    and end + 1 < len(lines)
                    and "|" not in lines[end + 1]
                ):
                    break
                end += 1
            block_start = line_starts[index]
            block_end = line_starts[end - 1] + len(lines[end - 1])
            output.append(_marker(
                "TABLE",
                "\n".join(lines[index:end]),
                replacements,
                marker_occurrences,
                image_index.occurrence_ids_between(block_start, block_end)
                if image_index is not None else [],
            ))
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
            block_start = line_starts[index]
            block_end = line_starts[end - 1] + len(lines[end - 1])
            output.append(_marker(
                "LIST",
                "\n".join(lines[index:end]),
                replacements,
                marker_occurrences,
                image_index.occurrence_ids_between(block_start, block_end)
                if image_index is not None else [],
            ))
            index = end
            continue

        output.append(protect_images(line, line_starts[index]))
        index += 1
    return "\n".join(output), replacements, marker_occurrences


def _restore_markdown_blocks(text, replacements):
    if not replacements or not text:
        return text
    return _MARKER_RE.sub(
        lambda match: replacements.get(match.group(0), match.group(0)),
        text,
    )


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


def _split_structural_block(body, target_size, replacements):
    if len(_restore_markdown_blocks(body, replacements)) <= target_size:
        return [body]
    image_markers = list(_IMAGE_MARKER_RE.finditer(body))
    if image_markers:
        # MinerU often puts a row of figures and captions into one structural
        # block.  Split at image boundaries so each figure remains searchable
        # with its immediately following caption instead of creating a huge
        # image-only chunk.
        pieces = []
        cursor = 0
        for marker_index, match in enumerate(image_markers):
            before = body[cursor:match.start()].strip()
            if before:
                pieces.extend(_split_plain_block(before, target_size))
            end = (
                image_markers[marker_index + 1].start()
                if marker_index + 1 < len(image_markers)
                else len(body)
            )
            pieces.append(body[match.start():end].strip())
            cursor = end
        trailing = body[cursor:].strip()
        if trailing:
            pieces.extend(_split_plain_block(trailing, target_size))
        return [piece for piece in pieces if piece]
    # A table, list, code block, or image-containing block is kept intact.  A
    # broken Markdown construct is harder to retrieve and display than a
    # slightly oversized chunk.
    if _MARKER_RE.search(body):
        return [body]
    return _split_plain_block(body, target_size)


def _pack_blocks(blocks, replacements, target_size, marker_occurrences=None):
    records = []
    current_header = None
    current_body = ""
    # Restored length of current_body, tracked incrementally.  Markers never
    # span the "\n\n" join between pieces (see _split_structural_block), so
    # restoration distributes over concatenation and we avoid re-restoring the
    # whole accumulated body on every append (was O(n²) per structural block).
    current_restored = 0

    def flush():
        nonlocal current_body, current_header, current_restored
        if not current_body.strip():
            current_body = ""
            current_header = None
            current_restored = 0
            return
        raw_body = current_body.strip()
        occurrence_ids = []
        if marker_occurrences:
            for marker in _MARKER_RE.findall(raw_body):
                occurrence_ids.extend(marker_occurrences.get(marker, []))
        restored = _restore_markdown_blocks(raw_body, replacements)
        records.append(_chunk_record(
            _chunk_prefix(current_header) + restored,
            header_path=current_header,
            image_occurrence_ids=sorted(set(occurrence_ids)),
        ))
        current_body = ""
        current_header = None
        current_restored = 0

    for header_path, body in blocks:
        prefix = _chunk_prefix(header_path)
        prefix_len = len(prefix)
        effective_target = max(128, target_size - prefix_len)
        for piece in _split_structural_block(body, effective_target, replacements):
            piece_restored = len(_restore_markdown_blocks(piece, replacements))
            if current_header is not None and header_path != current_header:
                flush()
            if not current_body:
                current_header = header_path
                current_body = piece
                current_restored = piece_restored
                continue
            # restored length of (current_body + "\n\n" + piece)
            candidate_restored = current_restored + 2 + piece_restored
            if prefix_len + candidate_restored > target_size:
                flush()
                current_header = header_path
                current_body = piece
                current_restored = piece_restored
            else:
                current_body = current_body + "\n\n" + piece
                current_restored = candidate_restored
    flush()
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
    protected, replacements, marker_occurrences = _protect_markdown_blocks(text, image_index)
    blocks = _markdown_sections(protected)
    return _pack_blocks(blocks, replacements, target_size, marker_occurrences)


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
