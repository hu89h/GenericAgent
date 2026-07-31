"""Small filesystem helpers for application-managed knowledge-base data."""
from __future__ import annotations

import os
import shutil
import stat


def remove_tree(path: str | os.PathLike[str], *, ignore_errors: bool = True) -> None:
    """Remove a managed directory, including read-only files copied from sources."""
    target = os.fspath(path)
    if not os.path.lexists(target):
        return

    def onerror(function, failed_path, exc_info):
        try:
            os.chmod(failed_path, stat.S_IWRITE)
            function(failed_path)
        except OSError:
            if not ignore_errors:
                raise

    try:
        shutil.rmtree(target, onerror=onerror)
    except OSError:
        if not ignore_errors:
            raise


__all__ = ["remove_tree"]
