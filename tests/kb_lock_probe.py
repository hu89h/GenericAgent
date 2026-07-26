"""Small cross-process probe for the KB mutation TCP/RLock contract."""

from __future__ import annotations

import argparse
import json
import time

from knowledge_base.locking import KnowledgeBaseMutationLock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=45764)
    parser.add_argument("--hold", type=float, default=0.0)
    args = parser.parse_args()
    try:
        with KnowledgeBaseMutationLock(port=args.port):
            print(json.dumps({"state": "acquired", "port": args.port}), flush=True)
            time.sleep(max(0.0, args.hold))
        return 0
    except Exception as error:
        print(json.dumps(
            {
                "state": "rejected",
                "error_type": type(error).__name__,
                "error": str(error),
                "port": args.port,
            }
        ), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
