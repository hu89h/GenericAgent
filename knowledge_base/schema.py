"""Single source of truth for knowledge-base records and Zvec fields."""
from __future__ import annotations

import json


INDEX_SCHEMA_VERSION = 1

STRING_FIELDS = (
    "data_id",
    "kb_id",
    "file_name",
    "title",
    "kind",
    "image_path",
    "source_data_id",
    "header_path",
    "body",
    "image_id",
    "ref_key",
    "display_label",
    "caption",
    "description",
    "table_markdown",
    "source_file_name",
    "uncertain",
    "analysis_error",
    "related_text",
    "near_text",
)
INT_FIELDS = ("chunk_index", "source_chunk_index")
OUTPUT_FIELDS = list(STRING_FIELDS + INT_FIELDS)


def normalize_record(record: dict, *, kb_id: str) -> dict:
    value = {field: str(record.get(field) or "") for field in STRING_FIELDS}
    value["kb_id"] = str(kb_id)
    value["kind"] = value["kind"] or "text"
    value["chunk_index"] = int(record.get("chunk_index") or 0)
    source_index = record.get("source_chunk_index")
    value["source_chunk_index"] = int(source_index if source_index is not None else -1)
    uncertain = record.get("uncertain")
    if isinstance(uncertain, (list, dict)):
        value["uncertain"] = json.dumps(uncertain, ensure_ascii=False) if uncertain else ""
    return value


def collection_schema(zvec, dimension: int):
    fields = [
        zvec.FieldSchema(field, zvec.DataType.STRING)
        for field in STRING_FIELDS
    ]
    fields.extend(
        zvec.FieldSchema(field, zvec.DataType.INT64)
        for field in INT_FIELDS
    )
    return zvec.CollectionSchema(
        name="kb_records",
        fields=fields,
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                int(dimension),
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
            ),
            zvec.VectorSchema("sparse_embedding", zvec.DataType.SPARSE_VECTOR_FP32),
        ],
    )


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "INT_FIELDS",
    "OUTPUT_FIELDS",
    "STRING_FIELDS",
    "collection_schema",
    "normalize_record",
]
