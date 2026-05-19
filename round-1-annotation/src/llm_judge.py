from __future__ import annotations

"""LLM judge for round-1."""

import asyncio
import base64
import io
import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
    "rate limit",
    "too many requests",
)

_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class QuotaExceededError(RuntimeError):
    pass


_QUOTA_EXCEEDED_ERROR_MARKERS = (
    "quota exceeded",
    "exceeded your current quota",
    "insufficient credits",
    "payment required",
    "credit balance",
    "limit_remaining",
    "free-models-per-day",
    "resource_exhausted",
    "quota",
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
_ACTIVE_PROVIDER: Optional[str] = None

_REPAIR_SUFFIX = (
    "\n\nPhản hồi trước của bạn không phải JSON hợp lệ. "
    "Hãy chỉ trả về đúng một đối tượng JSON hợp lệ, không markdown, không giải thích thêm. "
    "Giữ nguyên các khóa và kiểu dữ liệu như output schema đã yêu cầu."
)

_OPENROUTER_RESPONSE_FORMAT = {"type": "json_object"}


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
        raise ImportError("Missing dependency 'google-genai'. Install requirements before running the pipeline.") from exc
    return genai, types


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Missing dependency 'Pillow'. Install requirements before running the pipeline.") from exc
    return Image


def _split_api_keys(raw_value: str) -> List[str]:
    normalized = raw_value.replace("\n", ",").replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _resolve_openrouter_api_keys(api_key: Optional[str] = None) -> List[str]:
    raw_value = api_key or os.environ.get("OPENROUTER_API_KEYS") or os.environ.get("OPENROUTER_API_KEY")
    if not raw_value:
        raise ValueError("OpenRouter API key list not found. Set OPENROUTER_API_KEYS / OPENROUTER_API_KEY or pass --api_key.")
    keys = _split_api_keys(raw_value)
    if not keys:
        raise ValueError("OpenRouter API key list is empty.")
    return keys


def _resolve_gemini_api_keys(api_key: Optional[str] = None) -> List[str]:
    raw_value = (
        api_key
        or os.environ.get("GEMINI_API_KEYS")
        or os.environ.get("GOOGLE_API_KEYS")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not raw_value:
        raise ValueError("Gemini API key list not found. Set GEMINI_API_KEYS / GEMINI_API_KEY / GOOGLE_API_KEY or pass --api_key.")
    keys = _split_api_keys(raw_value)
    if not keys:
        raise ValueError("Gemini API key list is empty.")
    return keys


def _resolve_api_key(provider: str, api_key: Optional[str] = None) -> str:
    if provider == "openrouter":
        return _resolve_openrouter_api_keys(api_key)[0]
    if provider == "gemini_api":
        return _resolve_gemini_api_keys(api_key)[0]

    resolved = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not resolved:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY / GOOGLE_API_KEY or pass --api_key.")
    return resolved


def load_api_client(provider: str, api_key: Optional[str] = None):
    global _CLIENT, _ASYNC_CLIENT, _ACTIVE_PROVIDER
    if provider == "openrouter":
        resolved_keys = _resolve_openrouter_api_keys(api_key)
        _CLIENT = {
            "provider": provider,
            "api_keys": resolved_keys,
            "active_key_index": 0,
            "exhausted_key_indices": set(),
            "lock": None,
        }
        _ASYNC_CLIENT = _CLIENT
        _ACTIVE_PROVIDER = provider
        return _CLIENT

    if provider == "gemini_api":
        resolved_keys = _resolve_gemini_api_keys(api_key)
        genai, _ = _import_google_genai()
        _CLIENT = {
            "provider": provider,
            "api_keys": resolved_keys,
            "active_key_index": 0,
            "exhausted_key_indices": set(),
            "lock": None,
            "clients": [genai.Client(api_key=key) for key in resolved_keys],
        }
        _ASYNC_CLIENT = _CLIENT
        _ACTIVE_PROVIDER = provider
        return _CLIENT

    if _CLIENT is None or _ACTIVE_PROVIDER != provider:
        genai, _ = _import_google_genai()
        _CLIENT = genai.Client(api_key=_resolve_api_key(provider, api_key))
        _ASYNC_CLIENT = _CLIENT.aio
        _ACTIVE_PROVIDER = provider
    return _CLIENT


def load_async_api_client(provider: str, api_key: Optional[str] = None):
    load_api_client(provider, api_key)
    if provider in ("openrouter", "gemini_api") and _ASYNC_CLIENT is not None and _ASYNC_CLIENT.get("lock") is None:
        _ASYNC_CLIENT["lock"] = asyncio.Lock()
    return _ASYNC_CLIENT


def get_openrouter_key_count(api_key: Optional[str] = None) -> int:
    return len(_resolve_openrouter_api_keys(api_key))


def get_gemini_key_count(api_key: Optional[str] = None) -> int:
    return len(_resolve_gemini_api_keys(api_key))


async def close_async_api_client() -> None:
    global _CLIENT, _ASYNC_CLIENT, _ACTIVE_PROVIDER
    if _ACTIVE_PROVIDER == "gemini_api" and isinstance(_CLIENT, dict):
        for client in _CLIENT.get("clients", []):
            await client.aio.aclose()
            client.close()
    elif _ACTIVE_PROVIDER == "gemini_api" and _ASYNC_CLIENT is not None:
        await _ASYNC_CLIENT.aclose()
        if _CLIENT is not None:
            _CLIENT.close()
    _ASYNC_CLIENT = None
    _CLIENT = None
    _ACTIVE_PROVIDER = None


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


def _open_image(image_path: str, max_pixels: int = 300_000) -> Optional[Image.Image]:
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


def _load_images(record: InputRecord, max_pixels: int = 300_000) -> tuple[List[Image.Image], bool]:
    paths: List[str] = []
    if record.image_paths:
        paths = record.image_paths
    elif record.image_path:
        paths = [record.image_path]
    if not paths:
        return [], False
    images_pil = [img for p in paths for img in [_open_image(p, max_pixels)] if img is not None]
    return images_pil, len(images_pil) == 0


def _build_prompt(text: str, image_count: int, ocr_text: Optional[str] = None) -> str:
    template = _load_prompt_template()
    if image_count:
        images_placeholder = (
            f"[{image_count} ảnh đính kèm — xem ảnh trong nội dung tin nhắn]"
            if image_count > 1 else "[Xem ảnh đính kèm]"
        )
    else:
        images_placeholder = "[Không có ảnh hoặc ảnh không đọc được]"
    ocr_placeholder = ocr_text.strip() if ocr_text and ocr_text.strip() else "[Không có OCR text]"
    return template.replace("{text}", text).replace("{images}", images_placeholder).replace("{ocr_text}", ocr_placeholder)


def _build_gemini_contents(text: str, images_pil: List[Image.Image], ocr_text: Optional[str] = None) -> list[Any]:
    prompt = _build_prompt(text, len(images_pil), ocr_text)
    return [*images_pil, prompt]


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_openrouter_messages(text: str, images_pil: List[Image.Image], ocr_text: Optional[str] = None) -> list[dict[str, Any]]:
    prompt = _build_prompt(text, len(images_pil), ocr_text)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images_pil:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
    return [{"role": "user", "content": content}]


def _build_repair_prompt(prompt: str, invalid_response: str) -> str:
    return (
        f"{prompt}\n\n"
        "Phản hồi trước của bạn:\n"
        f"{invalid_response}\n"
        f"{_REPAIR_SUFFIX}"
    )


def _build_openrouter_repair_messages(
    text: str,
    images_pil: List[Image.Image],
    invalid_response: str,
    ocr_text: Optional[str] = None,
) -> list[dict[str, Any]]:
    prompt = _build_prompt(text, len(images_pil), ocr_text)
    return [
        *_build_openrouter_messages(text, images_pil, ocr_text),
        {"role": "assistant", "content": invalid_response},
        {"role": "user", "content": [{"type": "text", "text": _build_repair_prompt(prompt, invalid_response)}]},
    ]


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


def _is_quota_exceeded_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _QUOTA_EXCEEDED_ERROR_MARKERS)


async def _get_provider_key_state(async_client: Dict[str, Any]) -> tuple[str, int, int]:
    lock = async_client["lock"]
    async with lock:
        key_index = int(async_client["active_key_index"])
        api_keys = async_client["api_keys"]
        return api_keys[key_index], key_index, len(api_keys)


async def _rotate_provider_api_key(async_client: Dict[str, Any]) -> tuple[bool, int, int, int]:
    lock = async_client["lock"]
    async with lock:
        current_index = int(async_client["active_key_index"])
        exhausted = async_client["exhausted_key_indices"]
        exhausted.add(current_index)
        api_keys = async_client["api_keys"]
        total = len(api_keys)

        for candidate_index in range(total):
            if candidate_index not in exhausted:
                async_client["active_key_index"] = candidate_index
                return True, current_index, candidate_index, total

        return False, current_index, current_index, total


def _compute_retry_delay_seconds(base_delay_seconds: float, attempt: int, max_delay_seconds: float) -> float:
    exponential = base_delay_seconds * (2 ** max(0, attempt - 1))
    capped = min(exponential, max_delay_seconds)
    jitter = random.uniform(0.0, min(1.0, capped * 0.2))
    return capped + jitter


async def _call_gemini_api_async(
    async_client: Dict[str, Any],
    model_name: str,
    contents: list[Any],
    temperature: float,
    max_output_tokens: int = 256,
) -> str:
    _, types = _import_google_genai()
    _, key_index, _ = await _get_provider_key_state(async_client)
    response = await async_client["clients"][key_index].aio.models.generate_content(
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


def _build_openrouter_payload(
    model_config: Dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_config["model_name"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": _OPENROUTER_RESPONSE_FORMAT,
    }
    reasoning = dict(model_config.get("reasoning") or {})
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def _extract_openrouter_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("Empty response from OpenRouter API.")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        merged = "".join(texts).strip()
        if merged:
            return merged
    raise ValueError("OpenRouter response did not contain text content.")


def _openrouter_request(api_key: str, payload: dict[str, Any]) -> str:
    request = urllib.request.Request(
        _OPENROUTER_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/codex",
            "X-Title": "social-media-mining-round1",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter connection error: {exc}") from exc

    try:
        response_json = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenRouter returned non-JSON body: {body[:300]}") from exc
    return _extract_openrouter_text(response_json)


async def _call_openrouter_api_async(
    async_client: Dict[str, Any],
    model_config: Dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
) -> str:
    payload = _build_openrouter_payload(model_config, messages, temperature, max_output_tokens)
    api_key, _, _ = await _get_provider_key_state(async_client)
    return await asyncio.to_thread(_openrouter_request, api_key, payload)


async def _judge_once_async(
    async_client,
    model_config: Dict[str, Any],
    record: InputRecord,
    temperature: float,
    max_image_pixels: int,
    max_output_tokens: int,
) -> LLMJudgeRecord:
    provider = model_config["provider"]
    images_pil, image_missing = _load_images(record, max_image_pixels)

    if provider == "openrouter":
        messages = _build_openrouter_messages(record.text, images_pil, record.ocr_text)
        raw = await _call_openrouter_api_async(async_client, model_config, messages, temperature, max_output_tokens)
    else:
        contents = _build_gemini_contents(record.text, images_pil, record.ocr_text)
        raw = await _call_gemini_api_async(async_client, model_config["model_name"], contents, temperature, max_output_tokens)

    try:
        result = _validate(_extract_json(raw))
    except (json.JSONDecodeError, ValueError, KeyError):
        if provider == "openrouter":
            raw2 = await _call_openrouter_api_async(
                async_client,
                model_config,
                _build_openrouter_repair_messages(record.text, images_pil, raw, record.ocr_text),
                temperature,
                max_output_tokens,
            )
        else:
            prompt = _build_prompt(record.text, len(images_pil), record.ocr_text)
            repair_prompt = _build_repair_prompt(prompt, raw)
            raw2 = await _call_gemini_api_async(async_client, model_config["model_name"], [*images_pil, repair_prompt], temperature, max_output_tokens)
        result = _validate(_extract_json(raw2))

    return result.model_copy(update={"id": record.id, "image_missing": image_missing})


async def judge_single_async(
    async_client,
    model_config: Dict[str, Any],
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
    attempt = 1

    while attempt <= max_retries:
        try:
            return await _judge_once_async(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens)
        except Exception as exc:
            provider = model_config.get("provider")
            if provider in ("openrouter", "gemini_api") and _is_quota_exceeded_error(exc):
                rotated, from_index, to_index, total_keys = await _rotate_provider_api_key(async_client)
                if rotated:
                    logger.warning(
                        "RotateKey | id=%d | model=%s | from_key=%d/%d | to_key=%d/%d | reason=quota_exceeded",
                        record.id,
                        model_config.get("tag", model_config.get("model_name", "unknown")),
                        from_index + 1,
                        total_keys,
                        to_index + 1,
                        total_keys,
                    )
                    continue
                provider_name = "OpenRouter" if provider == "openrouter" else "Gemini"
                raise QuotaExceededError(f"All {provider_name} API keys exhausted: {str(exc)[:220]}") from exc
            if _is_quota_exceeded_error(exc):
                raise QuotaExceededError(str(exc)[:300]) from exc
            last_error = str(exc)[:200]
            should_retry = attempt < max_retries and _is_retryable_error(exc)
            if should_retry:
                delay_seconds = _compute_retry_delay_seconds(retry_delay_seconds, attempt, max_retry_delay_seconds)
                logger.warning(
                    "Retry | id=%d | model=%s | attempt=%d/%d | wait=%.1fs | reason=%s",
                    record.id,
                    model_config.get("tag", model_config.get("model_name", "unknown")),
                    attempt,
                    max_retries,
                    delay_seconds,
                    str(exc)[:120],
                )
                await asyncio.sleep(delay_seconds)
                attempt += 1
                continue
            break

    return LLMJudgeRecord(
        id=record.id,
        label_llm1=-1,
        notes=f"Failed after {max_retries} attempts: {last_error}",
        parse_error=True,
        image_missing=image_missing,
    )
