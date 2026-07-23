"""Document discovery, text extraction, and Markdown chunking for the KB.

This module deliberately has no knowledge of KB configuration, embeddings, or
Zvec.  Keeping document preparation pure makes it reusable by build and
read-only inspection paths without importing the indexing stack.
"""
from __future__ import annotations

import os
import re
import warnings


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
    return {
        "chunker": "llamaindex_markdown",
        "markdown_hierarchical_chunk_sizes": list(MD_CHUNK_SIZES),
        "markdown_parser": "llama_index.core.node_parser.MarkdownNodeParser+HierarchicalNodeParser",
    }


def read_textfile(path):
    for encoding in ("utf-8", "gb18030", "latin-1"):
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
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_prefix(header_path):
    header_path = re.sub(r"\s+", " ", str(header_path or "")).strip()
    if not header_path or header_path == "/":
        return ""
    return f"章节路径：{header_path}\n\n"


def _chunk_record(body, *, header_path="", chunk_role="leaf", parent_chunk_index=-1):
    return {
        "body": body,
        "header_path": str(header_path or ""),
        "chunk_role": chunk_role,
        "parent_chunk_index": int(parent_chunk_index),
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


def _split_markdown_leaf_chunks(section, target_size, overlap):
    section = section.strip()
    if not section:
        return []
    if len(section) <= target_size:
        return [section]
    blocks = [block.strip() for block in re.split(r"\n{2,}", section) if block.strip()]
    chunks = []
    current = ""
    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= target_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) <= target_size:
            current = block
            continue
        start = 0
        while start < len(block):
            chunks.append(block[start : start + target_size])
            if start + target_size >= len(block):
                break
            start = max(start + target_size - overlap, start + 1)
        current = ""
    if current:
        chunks.append(current)
    return chunks


def _llamaindex_markdown_chunk_records(text, *, file_name=""):
    """Use LlamaIndex Markdown and hierarchical node parsers."""
    text = _clean(text)
    if not text:
        return []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, module=r"pydantic\..*")
        from llama_index.core import Document
        from llama_index.core.node_parser import (
            HierarchicalNodeParser,
            MarkdownNodeParser,
            get_leaf_nodes,
        )

    markdown_document = Document(text=text, metadata={"file_name": file_name or ""})
    markdown_nodes = MarkdownNodeParser().get_nodes_from_documents([markdown_document])
    if not markdown_nodes:
        return []

    chunks = []
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=list(MD_CHUNK_SIZES))
    for markdown_node in markdown_nodes:
        section = (markdown_node.get_content(metadata_mode="none") or "").strip()
        if not section:
            continue
        header_path = (getattr(markdown_node, "metadata", None) or {}).get("header_path", "")
        parent_index = len(chunks)
        chunks.append(_chunk_record(
            _chunk_prefix(header_path) + section,
            header_path=header_path,
            chunk_role="parent",
            parent_chunk_index=parent_index,
        ))
        section_document = Document(
            text=section,
            metadata={"file_name": file_name or "", "header_path": header_path or ""},
        )
        leaves = get_leaf_nodes(parser.get_nodes_from_documents([section_document]))
        if not leaves:
            chunks.append(_chunk_record(
                _chunk_prefix(header_path) + section,
                header_path=header_path,
                chunk_role="leaf",
                parent_chunk_index=parent_index,
            ))
            continue
        for leaf in leaves:
            body = (leaf.get_content(metadata_mode="none") or "").strip()
            if body:
                chunks.append(_chunk_record(
                    _chunk_prefix(header_path) + body,
                    header_path=header_path,
                    chunk_role="leaf",
                    parent_chunk_index=parent_index,
                ))
    return chunks


def chunk_document_records(text, *, ext="", file_name=""):
    ext = (ext or "").lower().lstrip(".")
    if ext not in {"md", "markdown"}:
        return []
    try:
        return _llamaindex_markdown_chunk_records(text, file_name=file_name)
    except ImportError:
        # Keep text-only inspection usable in minimal environments.
        chunks = []
        leaf_size = MD_CHUNK_SIZES[-1]
        overlap = max(0, leaf_size // 8)
        for header_path, section in _markdown_sections(_clean(text)):
            parent_index = len(chunks)
            chunks.append(_chunk_record(
                _chunk_prefix(header_path) + section,
                header_path=header_path,
                chunk_role="parent",
                parent_chunk_index=parent_index,
            ))
            leaves = _split_markdown_leaf_chunks(section, leaf_size, overlap) or [section]
            for leaf in leaves:
                chunks.append(_chunk_record(
                    _chunk_prefix(header_path) + leaf,
                    header_path=header_path,
                    chunk_role="leaf",
                    parent_chunk_index=parent_index,
                ))
        return chunks


def chunk_document_text(text, *, ext="", file_name=""):
    return [
        record["body"]
        for record in chunk_document_records(text, ext=ext, file_name=file_name)
        if record.get("chunk_role") != "parent"
    ]


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
