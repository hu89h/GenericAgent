"""Continuously read one real KB chunk while another process mutates the index."""

from __future__ import annotations

import argparse
import json
import time

from knowledge_base import backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--interval", type=float, default=0.02)
    args = parser.parse_args()

    documents = backend.list_documents()
    if not documents:
        print(json.dumps({"ok_reads": 0, "error_count": 1, "errors": ["no documents"]}))
        return 1
    data_id = documents[0]["data_id"]
    deadline = time.time() + max(0.1, args.duration)
    ok_reads = 0
    errors = []
    while time.time() < deadline:
        result = backend.read_content(data_id=data_id, chunk_index=1, max_chars=800)
        value = result.get("content") if isinstance(result, dict) else ""
        if (
            value
            and not result.get("error_code")
        ):
            ok_reads += 1
        else:
            errors.append(value[:240])
        time.sleep(max(0.0, args.interval))
    print(json.dumps(
        {
            "ok_reads": ok_reads,
            "error_count": len(errors),
            "error_samples": errors[:8],
        },
        ensure_ascii=False,
    ))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
