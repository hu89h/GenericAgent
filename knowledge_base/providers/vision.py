"""Build-time knowledge-base image analysis provider."""
from __future__ import annotations

import json
import os
import re
from typing import Dict

import multimodal

from . import provider_http, provider_settings, rate_limit


_VISION_UNSUPPORTED_RE = re.compile(
    r"(?:"
    r"(?:does not support|doesn't support|not support(?:ed)?|unsupported|does not accept|cannot accept|only supports? text)"
    r".{0,100}(?:image|image_url|vision|visual|multimodal)"
    r"|"
    r"(?:image|image_url|vision|visual|multimodal)"
    r".{0,100}(?:does not support|doesn't support|not support(?:ed)?|unsupported|does not accept|cannot accept|only supports? text)"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def is_vision_unsupported_error(error: object) -> bool:
    """Return true only for an explicit model/image capability rejection.

    Transport errors such as timeouts, TLS EOFs and rate limits deliberately do
    not match this classifier; those must continue through the normal retry
    path instead of disabling image analysis for the whole import.
    """
    if isinstance(error, dict):
        error = error.get("error") or error.get("message") or ""
    text = str(error or "").strip()
    if not text:
        return False
    if "不支持图片" in text or "不支持视觉" in text or "仅支持文本" in text:
        return True
    return bool(_VISION_UNSUPPORTED_RE.search(text))


def _config() -> Dict[str, object]:
    """Resolve vision endpoint/model/timeouts fresh on every call.

    Historically these were frozen at import time, so editing mykey.py or the
    environment required a process restart and could silently drift from the
    embedding config (which is read at runtime).  Reading here keeps the whole
    KB provider layer consistently runtime-configured.
    """
    cfg = provider_settings.vision_config()
    return {
        "base_url": str(cfg.get("apibase") or cfg.get("base_url") or "").rstrip("/"),
        "api_key": cfg.get("apikey") or "",
        "model": cfg.get("model") or "",
        "protocol": str(cfg.get("protocol") or "openai").strip().lower(),
        "timeout": int(cfg.get("read_timeout") or cfg.get("timeout") or 120),
        "retries": int(cfg.get("max_retries") or 4),
        "max_tokens": int(cfg.get("max_tokens") or 8192),
        "rpm_limit": max(1, int(os.environ.get("GA_KB_VLM_RPM", "30000"))),
        "tpm_limit": max(1, int(os.environ.get("GA_KB_VLM_TPM", "5000000"))),
        "rate_headroom": min(
            0.95,
            max(0.1, float(os.environ.get("GA_KB_VLM_RATE_HEADROOM", "0.8"))),
        ),
        "token_reserve": max(
            1, int(os.environ.get("GA_KB_VLM_TOKEN_RESERVE", "12000"))
        ),
    }


def prompt_version() -> int:
    return int(os.environ.get("GA_KB_IMAGE_PROMPT_VERSION", "7"))


_TABLE_FOCUS_RE = re.compile(
    r"(?i)(?:表格|附表|表\s*[0-9０-９一二三四五六七八九十百]+(?:\s*[-－–—.．·]\s*[0-9０-９]+)*|table\s*[0-9０-９]*)"
)
_FIGURE_FOCUS_RE = re.compile(
    r"(?i)(?:图表|示意图(?!片)|流程图|结构图|折线图|柱状图|饼图|附图|"
    r"图\s*[0-9０-９]+(?:\s*[-－–—.．·]\s*[0-9０-９]+)*|"
    r"figure\s*[0-9０-９]*|fig\.\s*[0-9０-９]*)"
)


def enabled() -> bool:
    """Whether build-time image analysis is on.

    Precedence: an explicit ``GA_KB_IMAGE_ANALYSIS`` env var wins (so a one-off
    build can force it on/off), otherwise the durable
    ``kb_vision_config['enabled']`` flag in mykey.py decides, defaulting to
    off when neither is set.
    """
    env_val = os.environ.get("GA_KB_IMAGE_ANALYSIS")
    if env_val is not None and env_val.strip() != "":
        return env_val.strip().lower() in ("1", "true", "yes", "on")
    try:
        cfg_enabled = provider_settings.vision_config().get("enabled")
    except Exception:
        cfg_enabled = None
    return bool(cfg_enabled)


def build_analysis_meta() -> Dict[str, object]:
    """Fields that affect what gets baked into the index and VLM cache."""
    cfg = _config()
    return {
        "enabled": enabled(),
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "protocol": cfg["protocol"],
        "prompt_version": prompt_version(),
        "preprocess_version": multimodal.IMAGE_PREPROCESS_VERSION,
    }


def analysis_meta() -> Dict[str, object]:
    return build_analysis_meta()


def understanding_focus(
    title: str = "",
    near_text: str = "",
    ref_candidates: list[str] | None = None,
) -> str:
    """Classify the prompt emphasis without selecting a model or endpoint."""
    text = "\n".join(
        [str(title or ""), str(near_text or ""), *[str(x or "") for x in (ref_candidates or [])]]
    )
    if _TABLE_FOCUS_RE.search(text):
        return "table"
    if _FIGURE_FOCUS_RE.search(text):
        return "figure"
    return "general"


def _prompt(focus: str, title: str, near_text: str, ref_candidates: list[str]) -> str:
    focus_hint = {
        "table": (
            "本张图片的重点是表格或带明显行列结构的内容。"
            "优先准确读取表头、行列关系、单位、数字和脚注，并尽量还原为 Markdown 表格。"
        ),
        "figure": (
            "本张图片的重点是图表、结构图或流程图。"
            "优先说明图例、坐标、趋势、结构关系、流程方向、可见标签和关键数值。"
        ),
        "general": (
            "重点描述主体、布局、结构关系、可见文字、图例和关键数值。"
        ),
    }.get(focus, "general")
    return (
        "你是图书知识库的图片预处理器。任务是把图书中的图片转成可检索的文本证据。\n"
        "只基于图片、图题/alt 和邻近正文作答；邻近正文只作为理解上下文，不能把正文中没有被图片支持的内容当作图片事实。\n"
        "输出必须是严格 JSON 对象，不要输出 Markdown 代码块，不要输出解释性前后缀。\n\n"
        f"{focus_hint}\n\n"
        "JSON 字段必须完整包含：\n"
        "- description: 详细、客观描述图片内容，包括主体、布局、结构关系、趋势、可见文字、图例、关键数值；不要只写一句图注。\n"
        "- table_markdown: 图片是表格或明显有行列结构时输出 Markdown 表格；否则为空字符串。\n"
        "- ref_key: 图片对应的图表编号。仅当图片、图题或候选中能确认时输出标准形式，例如“图3-2”或“表4.1”；无法确认时为空字符串。\n"
        "- uncertain: 数组，记录看不清、无法确认或可能误读的内容；没有则为空数组。\n\n"
        "生成要求：\n"
        "1. 客观、保守，不补充图片外事实。\n"
        "2. 表格、图表、结构图要优先保留编号、标题、行列名、指标名、单位和关键数值。\n"
        "3. description 和 table_markdown 是图片资产的检索文本，不要再输出额外摘要字段。\n"
        "4. ref_key 不能猜测；不要把“上一图”“下表”等指代当成编号。\n"
        "5. 所有字段都必须出现；无内容时用空字符串或空数组。\n\n"
        f"图题/alt：{title or ''}\n"
        f"邻近正文：{(near_text or '')[:800]}\n"
        f"候选图表编号：{json.dumps(ref_candidates or [], ensure_ascii=False)}"
    )


def _extract_json(text: str) -> Dict[str, object]:
    """Parse the model reply into a dict.

    On any parse failure this returns an *error-marked* dict
    (``{"error": ..., "raw": ...}``) rather than a fabricated
    "success" payload.  The build path (:meth:`analyze_image_job`)
    treats a truthy ``error`` as a failed analysis: it is neither
    cached nor written into the index, and it is retried on the next
    build.  This prevents non-JSON garbage from being frozen into the
    permanent VLM cache.
    """
    raw = (text or "").strip()
    text = raw
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    def parse_object(candidate: str) -> Dict[str, object] | None:
        variants = [candidate]
        # Vision models commonly put LaTeX commands such as ``\checkmark``
        # inside an otherwise valid JSON string.  A single backslash before
        # an unsupported JSON escape makes the whole response invalid.  Only
        # escape those unsupported backslashes; do not repair arbitrary
        # truncated or structurally invalid output.
        escaped = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
        if escaped != candidate:
            variants.append(escaped)
        for value in variants:
            try:
                obj = json.loads(value)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    parsed = parse_object(text)
    if parsed is not None:
        return parsed
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        parsed = parse_object(m.group(0))
        if parsed is not None:
            return parsed
    return {"error": "model did not return valid JSON", "raw": raw[:1000]}


def _vision_chat(path: str, prompt_text: str) -> Dict[str, object]:
    """POST one build-time text+image request and parse its JSON reply."""
    cfg = _config()
    if not (cfg["api_key"] and cfg["base_url"] and cfg["model"]):
        raise RuntimeError("mykey.py 需要配置支持视觉输入的模型（kb_vision_config、native_oai_* 或 native_claude_*）")
    limiter = rate_limit.get_limiter(
        "kb_vlm",
        rpm=cfg["rpm_limit"],
        tpm=cfg["tpm_limit"],
        headroom=cfg["rate_headroom"],
    )

    def usage_tokens(response: dict) -> int | None:
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return None
        return int(
            usage.get("total_tokens")
            or (
                int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                + int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            )
            or 0
        ) or None

    image_ref = multimodal.image_path_block(path)
    protocol = str(cfg.get("protocol") or "openai").strip().lower()
    if protocol == "anthropic":
        body = provider_http.anthropic_messages(
            model=cfg["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        multimodal.anthropic_image_part(image_ref),
                    ],
                }
            ],
            base=cfg["base_url"],
            key=cfg["api_key"],
            timeout=cfg["timeout"],
            retries=cfg["retries"],
            max_tokens=cfg["max_tokens"],
            extra={"temperature": 0},
            auth_mode="x-api-key" if str(cfg["api_key"]).startswith("sk-ant-") else "bearer",
            rate_limiter=limiter,
            estimated_tokens=cfg["token_reserve"],
            usage_tokens=usage_tokens,
        )
        content = "".join(
            str(block.get("text") or "")
            for block in (body.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        finish_reason = body.get("stop_reason")
    else:
        body = provider_http.chat_completions(
            model=cfg["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        multimodal.openai_image_part(image_ref),
                    ],
                }
            ],
            base=cfg["base_url"],
            key=cfg["api_key"],
            timeout=cfg["timeout"],
            retries=cfg["retries"],
            extra={"temperature": 0, "max_tokens": cfg["max_tokens"]},
            rate_limiter=limiter,
            estimated_tokens=cfg["token_reserve"],
            usage_tokens=usage_tokens,
        )
        choice = body["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    result = _extract_json(content)
    if result.get("error") and finish_reason:
        result["finish_reason"] = str(finish_reason)
    result["model"] = cfg["model"]
    result["prompt_version"] = prompt_version()
    if isinstance(body.get("usage"), dict):
        result["_usage"] = body["usage"]
    if body.get("id"):
        result["_request_id"] = body.get("id")
    return result


def analyze_image(
    path: str,
    *,
    focus: str = "general",
    title: str = "",
    near_text: str = "",
    ref_candidates: list[str] | None = None,
) -> Dict[str, object]:
    if not enabled():
        return {}
    return _vision_chat(path, _prompt(focus, title, near_text, ref_candidates or []))
