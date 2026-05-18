from __future__ import annotations

"""LLM judge backed by Gemma 4 via the Gemini API."""

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from PIL import Image

from .schemas import InputRecord, LLMJudgeRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_ERROR_MARKERS = (
    "429",
    "500",
    "503",
    "504",
    "resource_exhausted",
    "internal",
    "unavailable",
    "deadline_exceeded",
    "timeout",
    "timed out",
    "connection reset",
    "temporarily overloaded",
)


def _load_env_file() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    round1_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(round1_root / ".env", override=True)


_load_env_file()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "prompt.txt"
_PROMPT_TEMPLATE: Optional[str] = None
_CLIENT = None
_ASYNC_CLIENT = None

_REPAIR_SUFFIX = (
    "\n\nPhản hồi trước của bạn không phải JSON hợp lệ. "
    "Hãy chỉ trả về đúng một đối tượng JSON hợp lệ, không markdown, không giải thích thêm. "
    "Giữ nguyên các khóa và kiểu dữ liệu như output schema đã yêu cầu."
)


def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


def _import_google_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'google-genai'. Install requirements.txt before running the pipeline."
        ) from exc
    return genai, types


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'Pillow'. Install requirements.txt before running the pipeline."
        ) from exc
    return Image


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    resolved = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not resolved:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY / GOOGLE_API_KEY or pass --api_key.")
    return resolved


def load_api_client(api_key: Optional[str] = None):
    global _CLIENT, _ASYNC_CLIENT
    if _CLIENT is None:
        genai, _ = _import_google_genai()
        _CLIENT = genai.Client(api_key=_resolve_api_key(api_key))
        _ASYNC_CLIENT = _CLIENT.aio
    return _CLIENT


def load_async_api_client(api_key: Optional[str] = None):
    load_api_client(api_key)
    return _ASYNC_CLIENT


async def close_async_api_client() -> None:
    global _CLIENT, _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None:
        await _ASYNC_CLIENT.aclose()
    if _CLIENT is not None:
        _CLIENT.close()
    _ASYNC_CLIENT = None
    _CLIENT = None


def _resize_image(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if max_pixels <= 0 or w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    Image = _import_pil_image()
    resampling = getattr(Image, "Resampling", Image)
    return img.resize((new_w, new_h), resampling.LANCZOS)


def _open_image(image_path: str, max_pixels: int = 1_048_576) -> Optional[Image.Image]:
    Image = _import_pil_image()
    p = Path(image_path)

    if p.is_absolute():
        candidates = [p]
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [repo_root / image_path, Path.cwd() / image_path]

    for candidate in candidates:
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGB")
                return _resize_image(img, max_pixels)
            except Exception:
                return None
    return None


def _load_images(record: InputRecord, max_pixels: int = 1_048_576) -> tuple[List[Image.Image], bool]:
    paths: List[str] = []
    if record.image_paths:
        paths = record.image_paths
    elif record.image_path:
        paths = [record.image_path]

    if not paths:
        return [], False

    images_pil = [img for p in paths for img in [_open_image(p, max_pixels)] if img is not None]
    return images_pil, len(images_pil) == 0


def _build_contents(text: str, images_pil: List[Image.Image], ocr_text: Optional[str] = None) -> list[Any]:
    template = _load_prompt_template()
    if images_pil:
        images_placeholder = (
            f"[{len(images_pil)} ảnh đính kèm — xem ảnh trong nội dung tin nhắn]"
            if len(images_pil) > 1 else "[Xem ảnh đính kèm]"
        )
    else:
        images_placeholder = "[Không có ảnh hoặc ảnh không đọc được]"
    ocr_placeholder = ocr_text.strip() if ocr_text and ocr_text.strip() else "[Không có OCR text]"
    prompt = (
        template
        .replace("{text}", text)
        .replace("{images}", images_placeholder)
        .replace("{ocr_text}", ocr_placeholder)
    )
    return [*images_pil, prompt]


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(cleaned)


def _validate(data: dict) -> LLMJudgeRecord:
    raw_label = data.get("llm_label", "INVALID")
    if raw_label == "INVALID":
        label = "INVALID"
    elif raw_label in (0, 1, -1):
        label = int(raw_label)
    elif str(raw_label) in ("0", "1", "-1"):
        label = int(raw_label)
    else:
        label = "INVALID"

    has_emoji_raw = data.get("has_emoji")
    has_emoji = int(has_emoji_raw) if (has_emoji_raw in (0, 1) or str(has_emoji_raw) in ("0", "1")) else None

    needs_human_check_raw = data.get("needs_human_check")
    if needs_human_check_raw in (0, 1) or str(needs_human_check_raw) in ("0", "1"):
        needs_human_check = int(needs_human_check_raw)
    else:
        needs_human_check = None

    notes = str(data.get("notes") or data.get("Notes") or "")[:500]
    reasoning = data.get("reasoning", {})
    if isinstance(reasoning, str) and reasoning.strip():
        reasoning = {"verdict": reasoning}
    elif not isinstance(reasoning, dict):
        reasoning = {}

    return LLMJudgeRecord(
        id=-1,
        label_llm1=label,
        has_emoji=has_emoji,
        needs_human_check=needs_human_check,
        notes=notes,
        reasoning=reasoning,
    )


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_ERROR_MARKERS)


def _compute_retry_delay_seconds(base_delay_seconds: float, attempt: int, max_delay_seconds: float) -> float:
    exponential = base_delay_seconds * (2 ** max(0, attempt - 1))
    capped = min(exponential, max_delay_seconds)
    jitter = random.uniform(0.0, min(1.0, capped * 0.2))
    return capped + jitter


async def _call_gemini_api_async(
    async_client,
    model_name: str,
    contents: list[Any],
    temperature: float,
    max_output_tokens: int = 256,
) -> str:
    _, types = _import_google_genai()
    response = await async_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    if getattr(response, "text", None):
        return response.text
    if getattr(response, "parsed", None) is not None:
        return json.dumps(response.parsed, ensure_ascii=False)
    raise ValueError("Empty response from Gemini API.")


async def _judge_once_async(
    async_client,
    model_name: str,
    record: InputRecord,
    temperature: float,
    max_image_pixels: int,
    max_output_tokens: int,
) -> LLMJudgeRecord:
    images_pil, image_missing = _load_images(record, max_image_pixels)
    contents = _build_contents(record.text, images_pil, record.ocr_text)
    raw = await _call_gemini_api_async(async_client, model_name, contents, temperature, max_output_tokens)

    try:
        result = _validate(_extract_json(raw))
    except (json.JSONDecodeError, ValueError, KeyError):
        repair_prompt = raw + _REPAIR_SUFFIX
        raw2 = await _call_gemini_api_async(async_client, model_name, [repair_prompt], temperature, max_output_tokens)
        result = _validate(_extract_json(raw2))

    return result.model_copy(update={"id": record.id, "image_missing": image_missing})


async def judge_single_async(
    async_client,
    model_name: str,
    record: InputRecord,
    temperature: float,
    max_image_pixels: int = 300_000,
    max_output_tokens: int = 256,
    max_retries: int = 3,
    retry_delay_seconds: int = 5,
    max_retry_delay_seconds: int = 20,
) -> LLMJudgeRecord:
    last_error = "Unknown error"
    image_missing = _load_images(record, max_image_pixels)[1]

    for attempt in range(1, max_retries + 1):
        try:
            return await _judge_once_async(
                async_client,
                model_name,
                record,
                temperature,
                max_image_pixels,
                max_output_tokens,
            )
        except Exception as exc:
            last_error = str(exc)[:200]
            should_retry = attempt < max_retries and _is_retryable_error(exc)
            if should_retry:
                delay_seconds = _compute_retry_delay_seconds(
                    retry_delay_seconds,
                    attempt,
                    max_retry_delay_seconds,
                )
                logger.warning(
                    "Retry | id=%d | attempt=%d/%d | wait=%.1fs | reason=%s",
                    record.id,
                    attempt,
                    max_retries,
                    delay_seconds,
                    str(exc)[:120],
                )
                await asyncio.sleep(delay_seconds)
                continue
            break

    return LLMJudgeRecord(
        id=record.id,
        label_llm1=-1,
        notes=f"Failed after {max_retries} attempts: {last_error}",
        parse_error=True,
        image_missing=image_missing,
    )
