"""Protocol-neutral image references and provider payload conversion."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_DECODED_PIXELS = 40_000_000
MAX_INPUT_PIXELS = 4_000_000
MAX_ENCODED_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 10
MAX_TOOL_IMAGES_PER_TURN = 3
IMAGE_CONTEXT_CHAR_COST = 12_000
IMAGE_PREPROCESS_VERSION = 1
_CACHE_MAX_BYTES = 64 * 1024 * 1024

_RASTER_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".ico",
}
_RASTER_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF", "ICO"}
_DATA_URL_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+", re.IGNORECASE)


class ImageContentError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code or "image_invalid")


@dataclass(frozen=True, slots=True)
class PreparedImage:
    media_type: str
    data: str
    width: int
    height: int
    encoded_bytes: int
    name: str

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"


_cache_lock = threading.RLock()
_prepared_cache: OrderedDict[tuple[Any, ...], PreparedImage] = OrderedDict()
_prepared_cache_bytes = 0


def is_raster_image_path(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in _RASTER_EXTENSIONS


def image_path_block(path: str | os.PathLike[str], name: str = "") -> dict[str, str]:
    resolved = str(Path(path).expanduser().resolve())
    safe_name = Path(str(name or "").replace("\\", "/")).name
    return {
        "type": "image_path",
        "path": resolved,
        "name": safe_name or Path(resolved).name,
    }


def _cache_key(
    path: Path,
    max_source_bytes: int,
    max_decoded_pixels: int,
    max_pixels: int,
    max_encoded_bytes: int,
) -> tuple[Any, ...]:
    try:
        stat = path.stat()
    except OSError as error:
        raise ImageContentError("image_not_found", f"image file is unavailable: {path.name}") from error
    return (
        str(path), stat.st_mtime_ns, stat.st_size,
        int(max_source_bytes), int(max_decoded_pixels),
        int(max_pixels), int(max_encoded_bytes),
    )


def _cache_get(key: tuple[Any, ...]) -> PreparedImage | None:
    with _cache_lock:
        value = _prepared_cache.get(key)
        if value is not None:
            _prepared_cache.move_to_end(key)
        return value


def _cache_put(key: tuple[Any, ...], value: PreparedImage) -> None:
    global _prepared_cache_bytes
    with _cache_lock:
        old = _prepared_cache.pop(key, None)
        if old is not None:
            _prepared_cache_bytes -= len(old.data)
        _prepared_cache[key] = value
        _prepared_cache_bytes += len(value.data)
        while _prepared_cache and _prepared_cache_bytes > _CACHE_MAX_BYTES:
            _key, removed = _prepared_cache.popitem(last=False)
            _prepared_cache_bytes -= len(removed.data)


def clear_image_cache() -> None:
    global _prepared_cache_bytes
    with _cache_lock:
        _prepared_cache.clear()
        _prepared_cache_bytes = 0


def _flatten_alpha(image):
    from PIL import Image

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _resize_to_pixels(image, max_pixels: int):
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    from PIL import Image

    scale = math.sqrt(max_pixels / float(pixels))
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _encode_image(image, *, prefer_png: bool, max_encoded_bytes: int) -> tuple[bytes, str]:
    def encode(fmt: str, **kwargs) -> bytes:
        stream = io.BytesIO()
        image.save(stream, format=fmt, **kwargs)
        return stream.getvalue()

    if prefer_png:
        blob = encode("PNG", optimize=True)
        if len(blob) <= max_encoded_bytes:
            return blob, "image/png"

    rgb = _flatten_alpha(image)
    for quality in (88, 80, 72, 64, 56):
        blob = encode("JPEG", quality=quality, optimize=True, progressive=True)
        if len(blob) <= max_encoded_bytes:
            return blob, "image/jpeg"
    raise ImageContentError(
        "image_encoded_too_large",
        f"image remains larger than {max_encoded_bytes // (1024 * 1024)} MiB after normalization",
    )


def prepare_image(
    path: str | os.PathLike[str],
    *,
    max_source_bytes: int = MAX_SOURCE_BYTES,
    max_decoded_pixels: int = MAX_DECODED_PIXELS,
    max_pixels: int = MAX_INPUT_PIXELS,
    max_encoded_bytes: int = MAX_ENCODED_BYTES,
) -> PreparedImage:
    resolved = Path(path).expanduser().resolve()
    key = _cache_key(
        resolved, max_source_bytes, max_decoded_pixels, max_pixels, max_encoded_bytes,
    )
    if key[2] <= 0:
        raise ImageContentError("image_empty", f"image file is empty: {resolved.name}")
    if key[2] > max_source_bytes:
        raise ImageContentError(
            "image_source_too_large",
            f"image is larger than {max_source_bytes // (1024 * 1024)} MiB: {resolved.name}",
        )
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as error:
        raise ImageContentError("image_dependency_missing", "Pillow is required for image input") from error

    try:
        with Image.open(resolved) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > max_decoded_pixels:
                raise ImageContentError(
                    "image_dimensions_invalid",
                    f"image exceeds the {max_decoded_pixels:,}-pixel decode limit: {resolved.name}",
                )
            if getattr(opened, "is_animated", False):
                opened.seek(0)
            image = ImageOps.exif_transpose(opened).copy()
            source_format = str(opened.format or "").upper()
            if source_format not in _RASTER_FORMATS:
                raise ImageContentError(
                    "image_format_unsupported",
                    f"unsupported image format: {source_format or 'unknown'}",
                )
    except ImageContentError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageContentError("image_decode_failed", f"unsupported or corrupt image: {resolved.name}") from error
    except Exception as error:
        raise ImageContentError("image_decode_failed", f"could not decode image: {resolved.name}") from error

    image = _resize_to_pixels(image, max_pixels)
    image = _flatten_alpha(image)
    prefer_png = source_format in {"PNG", "GIF", "BMP", "TIFF", "ICO"}
    blob, media_type = _encode_image(image, prefer_png=prefer_png, max_encoded_bytes=max_encoded_bytes)
    prepared = PreparedImage(
        media_type=media_type,
        data=base64.b64encode(blob).decode("ascii"),
        width=image.width,
        height=image.height,
        encoded_bytes=len(blob),
        name=resolved.name,
    )
    _cache_put(key, prepared)
    return prepared


def normalize_image_inputs(images: Iterable[Any] | None) -> list[dict[str, str]]:
    values = list(images or [])
    if len(values) > MAX_IMAGES_PER_MESSAGE:
        raise ImageContentError(
            "image_count_exceeded",
            f"at most {MAX_IMAGES_PER_MESSAGE} images can be sent in one message",
        )
    blocks = []
    for item in values:
        if isinstance(item, (str, os.PathLike)):
            path, name = str(item), ""
        elif isinstance(item, dict):
            path, name = str(item.get("path") or ""), str(item.get("name") or "")
        else:
            raise ImageContentError("image_input_invalid", "image input must be a path or an object with path")
        if not path:
            raise ImageContentError("image_path_missing", "image path is required")
        if not is_raster_image_path(path):
            raise ImageContentError("image_format_unsupported", f"unsupported image format: {Path(path).suffix or '(none)'}")
        block = image_path_block(path, name)
        prepare_image(block["path"])
        blocks.append(block)
    return blocks


def contains_image_paths(messages: Iterable[dict[str, Any]]) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") == "image_path"
        for message in messages
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
    )


def anthropic_image_part(block: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_image(str(block.get("path") or ""))
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": prepared.media_type,
            "data": prepared.data,
        },
    }


def openai_image_part(block: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_image(str(block.get("path") or ""))
    return {"type": "image_url", "image_url": {"url": prepared.data_url}}


def materialize_anthropic_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        copied = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            copied["content"] = [
                anthropic_image_part(block)
                if isinstance(block, dict) and block.get("type") == "image_path"
                else dict(block) if isinstance(block, dict) else block
                for block in content
            ]
        result.append(copied)
    return result


def restore_image_references(messages: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Copy persisted history and replace missing image references with explicit text."""
    restored = []
    for message in messages or []:
        copied = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_path":
                    path = str(block.get("path") or "")
                    if not path or not Path(path).is_file():
                        blocks.append({
                            "type": "text",
                            "text": f"[Historical image is no longer available: {Path(path).name or 'image'}]",
                        })
                        continue
                blocks.append(dict(block) if isinstance(block, dict) else block)
            copied["content"] = blocks
        restored.append(copied)
    return restored


def estimated_history_chars(message: dict[str, Any]) -> int:
    base = len(json.dumps(message, ensure_ascii=False, default=str))
    content = message.get("content")
    if isinstance(content, list):
        base += sum(
            IMAGE_CONTEXT_CHAR_COST
            for block in content
            if isinstance(block, dict) and block.get("type") == "image_path"
        )
    return base


def safe_log_value(value: Any) -> Any:
    if isinstance(value, list):
        return [safe_log_value(item) for item in value]
    if not isinstance(value, dict):
        if isinstance(value, str):
            return _DATA_URL_RE.sub("[image data omitted]", value)
        return value
    kind = value.get("type")
    if kind == "image_path":
        name = Path(str(value.get("path") or value.get("name") or "image")).name
        record = {"type": "image_path", "name": name}
        try:
            prepared = prepare_image(str(value.get("path") or ""))
            record.update({
                "media_type": prepared.media_type,
                "width": prepared.width,
                "height": prepared.height,
            })
        except ImageContentError as error:
            record["error"] = error.code
        return record
    copied = {key: safe_log_value(item) for key, item in value.items()}
    if kind == "image_url" and isinstance(copied.get("image_url"), dict):
        copied["image_url"] = {"url": "[image data omitted]"}
    if kind == "image" and isinstance(copied.get("source"), dict) and copied["source"].get("type") == "base64":
        copied["source"]["data"] = "[image data omitted]"
    return copied
