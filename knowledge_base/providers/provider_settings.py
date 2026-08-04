"""Read KB-related model settings from GenericAgent's package-local mykey.py."""
from __future__ import annotations

import importlib
import os
import re
import sys
from typing import Any, Dict


APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024
MINERU_BASE_URL = "https://mineru.net/api/v4"
MINERU_MODEL_VERSION = "vlm"


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


def _is_verified_vision_cfg(name: str, cfg: Any) -> bool:
    return bool(
        _is_model_cfg(name, cfg)
        and str(cfg.get("vision_mode") or "").strip().lower() == "multimodal"
        and cfg.get("vision") is True
        and cfg.get("vision_verified") is True
    )


def _preferred_llm_cfg(vars_: Dict[str, Any]) -> Dict[str, Any]:
    for name, cfg in vars_.items():
        if "native_oai" in name and _is_model_cfg(name, cfg):
            return dict(cfg)
    for name, cfg in vars_.items():
        if "oai" in name and _is_model_cfg(name, cfg):
            return dict(cfg)
    return {}


def _preferred_vision_cfg(vars_: Dict[str, Any], protocol: str = "") -> Dict[str, Any]:
    """Choose a native model config for KB vision when no override is set."""
    requested = str(protocol or "").strip().lower().replace("-", "_")
    if requested not in {"anthropic", "messages", "anthropic_messages", "claude"}:
        for name, cfg in vars_.items():
            if "native_oai" in name and _is_verified_vision_cfg(name, cfg):
                return {**dict(cfg), "protocol": "openai"}
    if requested in {"", "anthropic", "messages", "anthropic_messages", "claude"}:
        for name, cfg in vars_.items():
            if "native_claude" in name and isinstance(cfg, dict):
                if _is_verified_vision_cfg(name, cfg):
                    return {**dict(cfg), "protocol": "anthropic"}
    if requested in {"", "openai", "chat_completions", "oai"}:
        for name, cfg in vars_.items():
            if "native_oai" in name and _is_verified_vision_cfg(name, cfg):
                return {**dict(cfg), "protocol": "openai"}
    return {}


def _selected_vision_cfg(vars_: Dict[str, Any], selector: str) -> Dict[str, Any]:
    selector = str(selector or "").strip()
    if not selector:
        return {}
    for name, cfg in vars_.items():
        if name == selector and _is_verified_vision_cfg(name, cfg):
            protocol = "anthropic" if "claude" in name.lower() else "openai"
            return {**dict(cfg), "protocol": protocol}
    # Model profile variables can be renumbered after another profile is
    # removed (for example native_oai_config1 -> native_oai_config).  The
    # desktop wizard can already identify this unambiguously; runtime config
    # resolution must do the same or an enabled KB image-analysis job will be
    # silently disabled despite the model still being present.
    selector_base = re.sub(r"\d+$", "", selector)
    if selector_base:
        renamed = [
            (name, cfg)
            for name, cfg in vars_.items()
            if re.sub(r"\d+$", "", str(name)) == selector_base
            and _is_verified_vision_cfg(name, cfg)
        ]
        if len(renamed) == 1:
            name, cfg = renamed[0]
            protocol = "anthropic" if "claude" in name.lower() else "openai"
            return {**dict(cfg), "protocol": protocol}
    return {}


def _vision_protocol(cfg: Dict[str, Any], fallback: Dict[str, Any]) -> str:
    value = str(
        cfg.get("protocol")
        or cfg.get("provider")
        or cfg.get("api_mode")
        or fallback.get("protocol")
        or "openai"
    ).strip().lower().replace("-", "_")
    if value in {"anthropic", "messages", "anthropic_messages", "claude"}:
        return "anthropic"
    return "openai"


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


def vision_config() -> Dict[str, Any]:
    """Return the vision model config for image analysis / image QA.

    Read fresh from mykey.py on every call (like :func:`embedding_config`),
    so editing mykey.py or the environment takes effect without restarting
    the process.  A dedicated ``kb_vision_config`` block selects an explicitly
    verified multimodal model profile independent of the default chat model.
    No unverified chat profile is used as an implicit image provider.
    """
    vars_ = _load_mykey_vars()
    raw = vars_.get("kb_vision_config")
    cfg = dict(raw) if isinstance(raw, dict) else {}
    selected_profile = str(cfg.get("model_profile") or "").strip()
    if selected_profile:
        fallback = _selected_vision_cfg(vars_, selected_profile)
    elif isinstance(raw, dict) and cfg.get("enabled") is True:
        # An enabled KB image-analysis config must name the verified model
        # group explicitly; never silently fall back to the chat model.
        fallback = {}
    else:
        fallback = _preferred_vision_cfg(vars_, cfg.get("protocol") or cfg.get("api_mode"))
    # Image understanding is explicitly opt-in.  Never turn an unverified chat
    # profile into a KB vision provider merely because it happens to be present.
    allow_env_overrides = not (isinstance(raw, dict) and cfg.get("enabled") is True)

    def pick(*keys, env: str = "", default: Any = "") -> Any:
        for key in keys:
            val = cfg.get(key)
            if val:
                return val
        if env and allow_env_overrides:
            val = os.environ.get(env, "").strip()
            if val:
                return val
        for key in keys:
            val = fallback.get(key)
            if val:
                return val
        return default

    apibase = str(pick("apibase", "base_url", env="GA_KB_VISION_BASE_URL")).strip().rstrip("/")
    apikey = str(pick("apikey", "api_key", env="GA_KB_VISION_API_KEY")).strip()
    model = str(pick("model", env="GA_KB_VISION_MODEL")).strip()
    protocol = _vision_protocol(cfg, fallback)
    timeout = int(pick("read_timeout", "timeout", default=180) or 180)
    retries = int(pick("max_retries", default=4) or 4)
    max_tokens = int(
        pick("max_tokens", env="GA_KB_VISION_MAX_TOKENS", default=8192) or 8192
    )
    # Durable image-analysis switch: mykey.py's kb_vision_config['enabled']
    # persists the "images on" decision across restarts and future builds.
    # ``enabled`` is None here when the config does not mention it; the
    # provider may then consult the environment as an opt-in fallback.
    enabled = cfg.get("enabled") if isinstance(raw, dict) and "enabled" in cfg else None
    return {
        "apibase": apibase,
        "apikey": apikey,
        "model": model,
        "protocol": protocol,
        "read_timeout": timeout,
        "max_retries": retries,
        "max_tokens": max(512, max_tokens),
        "enabled": enabled,
        "model_profile": selected_profile,
    }


def embedding_config() -> Dict[str, Any]:
    """Return KB embedding config from mykey.py.

    Dense and sparse retrieval intentionally share one ``model``. Sparse calls
    add ``output_type='sparse'`` at request time.
    """
    vars_ = _load_mykey_vars()
    raw = vars_.get("kb_embedding_config")
    cfg = dict(raw) if isinstance(raw, dict) else {}

    # A dedicated KB block may override the endpoint/model while intentionally
    # inheriting the Qwen key from the regular GA model configuration.  The old
    # all-or-nothing check discarded those overrides whenever the dedicated
    # block omitted its own key.
    qwen = _qwen_cfg(vars_)
    fallback_key = qwen.get("apikey") if qwen else ""

    # DashScope uses one API key for chat and embedding. This fallback keeps a
    # Desktop install usable when only native_oai_config is filled in
    # mykey.py, while avoiding guesses for unrelated OAI-compatible providers.
    llm = _preferred_llm_cfg(vars_)
    base = str(llm.get("apibase") or "").lower()
    if not fallback_key and "dashscope.aliyuncs.com" in base:
        fallback_key = llm.get("apikey") or ""

    if not cfg and not fallback_key:
        return {}
    result = dict(cfg)
    result["apikey"] = str(
        result.get("apikey") or result.get("api_key") or fallback_key or ""
    ).strip()
    result["apibase"] = str(
        result.get("apibase") or result.get("base_url") or EMBEDDING_BASE_URL
    ).strip().rstrip("/")
    result["model"] = str(result.get("model") or EMBEDDING_MODEL).strip()
    result["dimension"] = int(result.get("dimension") or EMBEDDING_DIMENSION)
    return result


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
        or MINERU_BASE_URL
    ).strip().rstrip("/")
    model_version = str(
        raw.get("model_version")
        or os.environ.get("MINERU_MODEL_VERSION")
        or MINERU_MODEL_VERSION
    ).strip() or MINERU_MODEL_VERSION
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_version": model_version,
    }
