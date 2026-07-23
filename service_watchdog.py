"""Exit a managed service when the desktop bridge that owns it disappears."""
from __future__ import annotations

import os
import sys
import threading
import time


_WATCHDOG_STARTED = threading.Event()


def _parent_pid() -> int | None:
    raw = str(os.environ.get("GA_SERVICE_PARENT_PID", "")).strip()
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    return pid


def start_parent_watchdog(interval: float = 0.75) -> None:
    """Stop this service if the Bridge process disappears unexpectedly."""
    pid = _parent_pid()
    if pid is None or _WATCHDOG_STARTED.is_set():
        return

    try:
        import psutil
    except Exception as exc:
        print(f"[service-watchdog] unavailable: {exc}", file=sys.stderr)
        return

    _WATCHDOG_STARTED.set()

    def watch() -> None:
        while True:
            try:
                if not psutil.pid_exists(pid):
                    os._exit(0)
                parent = psutil.Process(pid)
                if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                    os._exit(0)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                os._exit(0)
            except Exception as exc:
                print(f"[service-watchdog] parent check failed: {exc}", file=sys.stderr)
            time.sleep(interval)

    threading.Thread(target=watch, name="service-parent-watchdog", daemon=True).start()
