from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from time import time, sleep
import html, os, subprocess, sys, tempfile, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["phase", "parallel", "mapchain"]

_T0 = time(); _RUN_DIR = None; _phases = []; _phase_stack = []; _tasks = []; _current = "idle"; _events = []; _srv = None; _last = time(); _lock = threading.Lock()

def _note(s):
    global _last
    with _lock:
        _last = time(); _events.append(f"{_last-_T0:7.1f}s  {s}"); del _events[:-60]

def _phase_lines(nodes, depth=0):
    out = []
    for p in nodes:
        pre = "  " * depth; mark = ">>" if p["on"] else "  "
        out.append(f"{pre}{mark} {p['status']:<7} {p['name']}" + (f" - {p['desc']}" if p['desc'] else ""))
        out += [f"{pre}   | {op}" for op in p.get("ops", [])[-8:]]
        out += [f"{pre}   - {t['status']:<5} {t['desc']}" for t in p.get("tasks", [])[-20:]]
        out += _phase_lines(p.get("children", []), depth + 1)
    return out

def _page():
    with _lock:
        lines = ["GA UltraPlan", "", f"current: {_current}", "", "phases:"]
        lines += _phase_lines(_phases) or ["(none)"]
        lines += ["", "recent tasks:"]
        lines += [f"{t['status']:<7} {t['desc']}" for t in _tasks[-12:]] or ["(none)"]
        lines += ["", "events:", *_events[-30:]]
    return "<meta http-equiv=refresh content=1><pre>" + html.escape("\n".join(lines)) + "</pre>"

class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        b = _page().encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass

def _show():
    global _srv
    if _srv or os.environ.get("GA_ULTRAPLAN_HTML") == "0": return
    _srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    url = f"http://127.0.0.1:{_srv.server_port}/"
    print(f"[ultraplan] {url}", flush=True)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()
    if os.environ.get("GA_ULTRAPLAN_BROWSER") != "0": webbrowser.open(url)
    def reap():
        while True:
            sleep(60)
            if time() - _last > 3600: _srv.shutdown(); break
    threading.Thread(target=reap, daemon=True).start()

@contextmanager
def phase(name, desc=""):
    global _current
    _show(); t = time(); p = {"name": name, "desc": desc, "status": "run", "on": True, "children": [], "tasks": [], "ops": []}
    with _lock:
        (_phase_stack[-1]["children"] if _phase_stack else _phases).append(p)
        _phase_stack.append(p); _current = f"phase: {name}"
    print(f"[phase] {name}" + (f" - {desc}" if desc else ""), flush=True); _note(f"phase start: {name}")
    failed = False
    try:
        yield
    except Exception:
        failed = True; raise
    finally:
        dt = time() - t; status = "fail" if failed else "done"
        with _lock:
            p["status"] = status; p["on"] = False
            if _phase_stack and _phase_stack[-1] is p: _phase_stack.pop()
            elif p in _phase_stack: _phase_stack.remove(p)
            if _phase_stack: _current = f"phase: {_phase_stack[-1]['name']}"
            else: _current = ("failed" if failed else "all phases done") + f"; last: {name} ({dt:.1f}s)"
        print(f"[{status}] {name} ({dt:.1f}s)", flush=True); _note(f"phase {status}:  {name} ({dt:.1f}s)")

def _task(desc, status="run"):
    with _lock:
        t = {"desc": str(desc), "status": status}; _tasks.append(t); del _tasks[:-80]
        if _phase_stack: _phase_stack[-1]["tasks"].append(t)
    return t

def _task_done(t, status="done"):
    with _lock: t["status"] = status

def _op(s):
    with _lock:
        if _phase_stack: _phase_stack[-1]["ops"].append(s)

def _fmt(x, data):
    return x.format(**data) if isinstance(x, str) else x

def _subagent(desc, prompt=None, *, llm_no=0, timeout=3600):
    global _RUN_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(tmp := os.path.join(root, "temp"), exist_ok=True)
    if _RUN_DIR is None:
        _RUN_DIR = os.path.join(tmp, f"ultraplan_{int(_T0)}_{os.getpid()}"); os.makedirs(_RUN_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", prefix="ultra_", dir=_RUN_DIR, delete=False) as f:
        f.write(desc if prompt is None else prompt)
    print(f"[subagent] {desc} -> {f.name}", flush=True); _note(f"agent: {desc}")
    cmd = [sys.executable, os.path.join(root, "agentmain.py"), "--func", f.name, "--llm_no", str(llm_no), "--nobg", "--nolog"]
    r = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=timeout)
    if r.returncode: raise RuntimeError(f"subagent failed: {desc}\n{r.stdout}\n{r.stderr}")
    return os.path.splitext(f.name)[0] + ".out.txt"

def _run(task, data):
    task = task() if callable(task) else task
    if isinstance(task, (tuple, list)):
        desc = _fmt(task[0], data); t = _task(desc)
        try: return _subagent(desc, _fmt(task[1] if len(task) > 1 else task[0], data), llm_no=data.get("llm_no", 0), timeout=data.get("timeout", 3600))
        except Exception: _task_done(t, "fail"); raise
        finally:
            if t["status"] == "run": _task_done(t)
    if isinstance(task, dict):
        d = {**data, **task.get("data", {})}; desc = _fmt(task.get("desc", "task"), d); t = _task(desc)
        try: return _subagent(desc, _fmt(task.get("prompt", task.get("desc", "task")), d), llm_no=task.get("llm_no", d.get("llm_no", 0)), timeout=task.get("timeout", d.get("timeout", 3600)))
        except Exception: _task_done(t, "fail"); raise
        finally:
            if t["status"] == "run": _task_done(t)
    return task

def parallel(tasks, max_workers=None, _label=None, **data):
    global _current
    tasks = list(tasks); label = _label or f"parallel: {len(tasks)} tasks"
    with _lock: _current = label
    _op(label); _note(label)
    with ThreadPoolExecutor(max_workers=max_workers or min(3, len(tasks) or 1)) as ex:
        return list(ex.map(lambda t: _run(t, data), tasks))

def mapchain(items, *steps, max_workers=None, **data):
    global _current
    items = list(items); label = f"mapchain: {len(items)} items x {len(steps)} steps"
    with _lock: _current = label
    def run(x):
        for step in steps:
            d = {**data, "item": x, "previous": x}; x = _run(step(x) if callable(step) else step, d)
        return x
    return parallel([lambda x=x: run(x) for x in items], max_workers=max_workers, _label=label)
