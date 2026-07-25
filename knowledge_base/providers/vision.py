"""Synchronous DashScope/OpenAI-compatible vision provider."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from typing import Dict

from . import provider_http, provider_settings


_MYKEY_LLM = provider_settings.llm_config()
BASE_URL = str(_MYKEY_LLM.get("apibase") or _MYKEY_LLM.get("base_url") or "").rstrip("/")
API_KEY = _MYKEY_LLM.get("apikey") or ""
MODEL = _MYKEY_LLM.get("model") or ""
PROMPT_VERSION = int(os.environ.get("GA_KB_IMAGE_PROMPT_VERSION", "6"))
TIMEOUT = int(_MYKEY_LLM.get("read_timeout") or _MYKEY_LLM.get("timeout") or 120)
RETRIES = int(_MYKEY_LLM.get("max_retries") or 2)
MAX_IMAGE_BYTES = int(os.environ.get("GA_KB_IMAGE_MAX_BYTES", str(8 * 1024 * 1024)))
RUNTIME_IMAGE_QA_ON = provider_http.env_bool("GA_KB_RUNTIME_IMAGE_QA", "1")
_TABLE_FOCUS_RE = re.compile(
    r"(?i)(?:表格|附表|表\s*[0-9０-９一二三四五六七八九十百]+(?:\s*[-－–—.．·]\s*[0-9０-９]+)*|table\s*[0-9０-９]*)"
)
_FIGURE_FOCUS_RE = re.compile(
    r"(?i)(?:图表|示意图(?!片)|流程图|结构图|折线图|柱状图|饼图|附图|"
    r"图\s*[0-9０-９]+(?:\s*[-－–—.．·]\s*[0-9０-９]+)*|"
    r"figure\s*[0-9０-９]*|fig\.\s*[0-9０-９]*)"
)


def enabled() -> bool:
    return provider_http.env_bool("GA_KB_IMAGE_ANALYSIS")


def analysis_meta() -> Dict[str, object]:
    return {
        "enabled": enabled(),
        "base_url": BASE_URL,
        "model": MODEL,
        "runtime_image_qa": RUNTIME_IMAGE_QA_ON,
        "prompt_version": PROMPT_VERSION,
    }


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


def _mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/jpeg"


def _data_url(path: str) -> str:
    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        raise RuntimeError(f"image too large: {size} bytes > {MAX_IMAGE_BYTES}")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{_mime(path)};base64,{b64}"


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
    permanent VLM cache (bug S1).
    """
    raw = (text or "").strip()
    text = raw
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"error": "model did not return valid JSON", "raw": raw[:1000]}


def _qa_prompt(question: str, title: str, context: Dict[str, object]) -> str:
    context_json = json.dumps(context or {}, ensure_ascii=False, indent=2)
    return (
        "你是图书知识库的实时图片问答器。用户已经通过检索定位到一张图书图片，"
        "离线预处理信息可能不完整；请同时依据图片本身、图题、已有预处理信息和邻近正文回答用户问题。\n\n"
        "要求：\n"
        "1. 优先回答用户问题，不要重新做通用图片描述。\n"
        "2. 必须区分图片中可直接看见的信息、已有图片预处理信息、邻近正文提供的解释。\n"
        "3. 如果问题需要的细节在图片中看不清或无法判断，明确说明不确定，不要编造。\n"
        "4. 输出严格 JSON 对象，不要输出 Markdown 代码块，不要输出解释性前后缀。\n\n"
        "JSON 字段必须完整包含：\n"
        "- answer: 针对用户问题的中文答案。\n"
        "- visual_evidence: 数组，列出来自图片本身或已有图片预处理结果的依据。\n"
        "- text_context_evidence: 数组，列出来自邻近正文或已有资产文本的依据。\n"
        "- uncertain: 数组，记录看不清、无法确认或可能误读的内容。\n\n"
        f"用户问题：{question or ''}\n"
        f"图题/alt：{title or ''}\n"
        f"已有图片资产信息：{context_json[:6000]}"
    )


def answer_image_question(
    path: str,
    *,
    question: str,
    title: str = "",
    context: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if not RUNTIME_IMAGE_QA_ON:
        raise RuntimeError("GA_KB_RUNTIME_IMAGE_QA 未开启，无法实时图片问答")
    if not (API_KEY and BASE_URL and MODEL):
        raise RuntimeError("mykey.py 需要配置支持视觉输入的 native_oai_* 模型")
    body = provider_http.chat_completions(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _qa_prompt(question, title, context or {})},
                    {"type": "image_url", "image_url": {"url": _data_url(path)}},
                ],
            }
        ],
        base=BASE_URL,
        key=API_KEY,
        timeout=TIMEOUT,
        retries=RETRIES,
        extra={"temperature": 0},
    )
    content = body["choices"][0]["message"]["content"]
    result = _extract_json(content)
    result["model"] = MODEL
    result["prompt_version"] = PROMPT_VERSION
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
    if not (API_KEY and BASE_URL and MODEL):
        raise RuntimeError("mykey.py 需要配置支持视觉输入的 native_oai_* 模型")
    body = provider_http.chat_completions(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(focus, title, near_text, ref_candidates or [])},
                    {"type": "image_url", "image_url": {"url": _data_url(path)}},
                ],
            }
        ],
        base=BASE_URL,
        key=API_KEY,
        timeout=TIMEOUT,
        retries=RETRIES,
        extra={"temperature": 0},
    )
    content = body["choices"][0]["message"]["content"]
    result = _extract_json(content)
    result["model"] = MODEL
    result["prompt_version"] = PROMPT_VERSION
    if isinstance(body.get("usage"), dict):
        result["_usage"] = body["usage"]
    if body.get("id"):
        result["_request_id"] = body.get("id")
    return result
