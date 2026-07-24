"""Read KB-related model settings from GenericAgent's package-local mykey.py."""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict


APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024


def embedding_cache_dir() -> str:
    """Return a writable runtime directory for local embedding responses."""
    override = os.environ.get("GA_KB_EMBED_CACHE_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    runtime_root = os.environ.get("GA_KB_RUNTIME_DIR", "").strip()
    if runtime_root:
        return os.path.join(os.path.abspath(os.path.expanduser(runtime_root)), "embedding_cache")
    return os.path.join(APP_ROOT, "data", "embedding_cache")


def _load_mykey_vars() -> Dict[str, Any]:
    try:
        import mykey  # type: ignore
        importlib.reload(mykey)
        return {k: v for k, v in vars(mykey).items() if not k.startswith("_")}
    except Exception:
        return {}


def _is_model_cfg(name: str, cfg: Any) -> bool:
    if not isinstance(cfg, dict):
        return False
    if name in {"kb_embedding_config", "langfuse_config"}:
        return False
    if "mixin" in name:
        return False
    mode = str(cfg.get("api_mode", "chat_completions")).strip().lower().replace("-", "_")
    if mode not in ("", "chat_completions"):
        return False
    return bool(cfg.get("apikey") and cfg.get("apibase") and cfg.get("model"))


def _preferred_llm_cfg(vars_: Dict[str, Any]) -> Dict[str, Any]:
    for name, cfg in vars_.items():
        if "native_oai" in name and _is_model_cfg(name, cfg):
            return dict(cfg)
    for name, cfg in vars_.items():
        if "oai" in name and _is_model_cfg(name, cfg):
            return dict(cfg)
    return {}


def _qwen_cfg(vars_: Dict[str, Any]) -> Dict[str, Any]:
    for name, cfg in vars_.items():
        if not isinstance(cfg, dict):
            continue
        haystack = " ".join(str(cfg.get(key) or "") for key in ("name", "model", "apibase"))
        haystack = f"{name} {haystack}".lower()
        if "qwen" in haystack or "通义" in haystack or "dashscope.aliyuncs.com" in haystack:
            if cfg.get("apikey"):
                return dict(cfg)
    return {}


def llm_config() -> Dict[str, Any]:
    """Return a chat-completions compatible KB LLM config from mykey.py.

    KB image preprocessing and image QA reuse the first regular GA
    ``native_oai_*`` model configuration that has ``apikey/apibase/model``.
    """
    return _preferred_llm_cfg(_load_mykey_vars())


def embedding_config() -> Dict[str, Any]:
    """Return KB embedding config from mykey.py.

    Dense and sparse retrieval intentionally share one ``model``. Sparse calls
    add ``output_type='sparse'`` at request time.
    """
    vars_ = _load_mykey_vars()
    cfg = vars_.get("kb_embedding_config")
    if isinstance(cfg, dict) and cfg.get("apikey"):
        return dict(cfg)

    qwen = _qwen_cfg(vars_)
    if qwen:
        return {
            "apikey": qwen.get("apikey"),
            "apibase": EMBEDDING_BASE_URL,
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
        }

    # DashScope uses one API key for chat and embedding. This fallback keeps a
    # Desktop install usable when only native_oai_config is filled in
    # mykey.py, while avoiding guesses for unrelated OAI-compatible providers.
    llm = _preferred_llm_cfg(vars_)
    base = str(llm.get("apibase") or "").lower()
    if "dashscope.aliyuncs.com" in base:
        return {
            "apikey": llm.get("apikey"),
            "apibase": EMBEDDING_BASE_URL,
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
        }
    return {}


def mineru_config() -> Dict[str, Any]:
    """Return MinerU import settings without requiring the importer to know mykey.

    The API key may be supplied in ``mykey.py`` as ``mineru_config`` or through
    ``MINERU_API_KEY``.  The endpoint and model have stable defaults, while
    environment overrides keep the package usable with a private compatible
    gateway in development and CI.
    """
    vars_ = _load_mykey_vars()
    raw = vars_.get("mineru_config")
    raw = raw if isinstance(raw, dict) else {}
    api_key = str(
        raw.get("api_key")
        or raw.get("apikey")
        or raw.get("token")
        or vars_.get("mineru_api_key")
        or vars_.get("mineru_key")
        or os.environ.get("MINERU_API_KEY")
        or os.environ.get("MINERU_TOKEN")
        or ""
    ).strip()
    base_url = str(
        raw.get("base_url")
        or os.environ.get("MINERU_BASE_URL")
        or "https://mineru.net/api/v4"
    ).strip().rstrip("/")
    model_version = str(
        raw.get("model_version")
        or os.environ.get("MINERU_MODEL_VERSION")
        or "vlm"
    ).strip() or "vlm"
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_version": model_version,
    }
