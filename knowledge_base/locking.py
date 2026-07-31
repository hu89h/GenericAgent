"""One re-entrant mutation lock shared by all knowledge-base mutations."""
from __future__ import annotations

import socket
import threading


class KnowledgeBaseLockedError(RuntimeError):
    code = "kb_mutation_locked"

    def __init__(self) -> None:
        super().__init__(self.code)


class KnowledgeBaseMutationLock:
    def __init__(self, port: int = 45764) -> None:
        self._port = int(port)
        self._local_lock = threading.RLock()
        self._state = threading.local()

    def __enter__(self):
        if not self._local_lock.acquire(blocking=False):
            raise KnowledgeBaseLockedError()
        depth = int(getattr(self._state, "depth", 0))
        if depth:
            self._state.depth = depth + 1
            return self

        process_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            process_lock.bind(("127.0.0.1", self._port))
        except OSError as error:
            process_lock.close()
            self._local_lock.release()
            raise KnowledgeBaseLockedError() from error
        self._state.depth = 1
        self._state.process_lock = process_lock
        return self

    def __exit__(self, exc_type, exc, traceback):
        depth = int(getattr(self._state, "depth", 1)) - 1
        self._state.depth = depth
        if depth == 0:
            process_lock = getattr(self._state, "process_lock", None)
            if process_lock is not None:
                process_lock.close()
            self._state.process_lock = None
        self._local_lock.release()
        return False


mutation_lock = KnowledgeBaseMutationLock()


__all__ = ["KnowledgeBaseLockedError", "KnowledgeBaseMutationLock", "mutation_lock"]
