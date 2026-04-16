from __future__ import annotations

"""
LLM judge backed by local model inference (Qwen/Qwen3.5-2B).

Requires CUDA GPU. Designed for Kaggle / Colab environments with free GPU.

The model is downloaded from HuggingFace Hub on first run and cached locally.
No HF_TOKEN needed — Qwen3.5-2B is public. If you add a private model later,
set HF_TOKEN or pass it to judge_batch().

The model singleton is loaded once and reused across all batch calls in the
same process. Supports 4-bit quantization (bitsandbytes) to reduce VRAM usage.

Model: Qwen/Qwen3.5-2B
  - Class    : AutoModelForImageTextToText  (natively multimodal, no -VL- suffix)
  - Processor: AutoProcessor
  - Inference API: processor.apply_chat_template(tokenize=True, return_dict=True)
  - Operates in non-thinking mode by default (no <think> block in output)
  - Requires: transformers >= 4.51.0

Prompt contract (prompt.txt):
  - Placeholder {text}   : the post text (may contain emoji)
  - Placeholder {images} : short description of image availability shown in
                           the text section; actual PIL images are prepended as
                           content items in the VL message so the model sees them.
  - The model must return a single JSON object with keys:
    reasoning, Label_LLM1, Text_Only, ImageSet_Only, Key_Images, Difficulty, Notes
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm

from .schemas import InputRecord, LLMJudgeRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "prompt.txt"
_FEW_SHOT_PATH = Path(__file__).parent.parent / "prompts" / "few-short-examples.txt"
_PROMPT_TEMPLATE: Optional[str] = None

_MODEL = None
_PROCESSOR = None
_LOADED_MODEL_NAME: Optional[str] = None

_REPAIR_SUFFIX = (
    "\n\nPhản hồi trước của bạn không phải JSON hợp lệ. "
    "Hãy chỉ trả về đúng một đối tượng JSON với các trường bắt buộc: "
    "reasoning, Label_LLM1, Text_Only, ImageSet_Only, Key_Images, Difficulty, Notes. "
    "Không thêm bất kỳ nội dung nào khác ngoài đối tượng JSON."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        if _FEW_SHOT_PATH.exists():
            few_shot = _FEW_SHOT_PATH.read_text(encoding="utf-8")
            prompt = prompt.rstrip() + "\n\n" + few_shot.lstrip()
        _PROMPT_TEMPLATE = prompt
    return _PROMPT_TEMPLATE


def _is_vl_model(model_name: str) -> bool:
    """
    Detect multimodal (vision-language) models by name convention.

    Covers:
      - Qwen2.5-VL-*, Qwen3-VL-* (explicit -vl- suffix)
      - Qwen3.5-* series (natively multimodal despite no -vl- in name)
    """
    n = model_name.lower()
    if "-vl-" in n or n.endswith("-vl"):
        return True
    if "qwen3.5" in n or "qwen3_5" in n:
        return True
    return False


def load_local_model(
    model_name: str,
    device: str = "cuda",
    load_in_4bit: bool = False,
    hf_token: Optional[str] = None,
) -> Tuple:
    """
    Load model and processor as a process-level singleton.
    Returns (model, processor). Subsequent calls with the same model_name
    return the cached instance immediately.
    """
    global _MODEL, _PROCESSOR, _LOADED_MODEL_NAME

    if _MODEL is not None and _LOADED_MODEL_NAME == model_name:
        return _MODEL, _PROCESSOR

    logger.info(
        "Loading model: %s  (device=%s, 4bit=%s)", model_name, device, load_in_4bit
    )

    token = hf_token or os.environ.get("HF_TOKEN")

    load_kwargs: dict = {
        "pretrained_model_name_or_path": model_name,
        "device_map": "auto",
    }
    if token:
        load_kwargs["token"] = token

    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["dtype"] = "auto"

    _MODEL = AutoModelForImageTextToText.from_pretrained(**load_kwargs)
    _MODEL.eval()

    proc_kwargs: dict = {}
    if token:
        proc_kwargs["token"] = token
    _PROCESSOR = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

    _LOADED_MODEL_NAME = model_name
    logger.info("Model loaded successfully.")
    return _MODEL, _PROCESSOR


def _resize_image(img: Image.Image, max_pixels: int) -> Image.Image:
    """
    Resize img so that width * height <= max_pixels, preserving aspect ratio.
    Uses LANCZOS resampling. Returns the original image if already within budget.
    """
    w, h = img.size
    if max_pixels <= 0 or w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.debug(
        "Resizing image %dx%d -> %dx%d (max_pixels=%d)", w, h, new_w, new_h, max_pixels
    )
    return img.resize((new_w, new_h), Image.LANCZOS)


def _open_image(image_path: str, max_pixels: int = 1_048_576) -> Optional[Image.Image]:
    """
    Open image as a PIL Image (RGB) and resize if needed.

    Resolution order (first existing path wins):
      1. Absolute path — used as-is.
      2. Relative path resolved from repo root (REPO_DIR).
         On Kaggle, REPO_DIR/data/ is a symlink to the mounted dataset, so
         "data/images/foo.png" → /kaggle/input/DATASET_SLUG/images/foo.png.
         This is always tried with an absolute path so it is cwd-independent.
      3. Relative path resolved from cwd — convenience fallback for local dev.

    After loading, the image is downscaled so that width * height <= max_pixels
    to cap VRAM usage in the vision encoder (set max_pixels=0 to disable).
    """
    p = Path(image_path)

    if p.is_absolute():
        candidates = [p]
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / image_path,
            Path.cwd() / image_path,
        ]

    for candidate in candidates:
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGB")
                return _resize_image(img, max_pixels)
            except Exception as exc:
                logger.warning("Cannot open image %s: %s", candidate, exc)
                return None

    logger.debug("Image not found at any candidate path: %s", image_path)
    return None


def _load_images(
    record: InputRecord,
    is_vl: bool,
    max_pixels: int = 1_048_576,
) -> Tuple[List[Image.Image], bool]:
    """
    Load all images for a record.
    Returns (list_of_pil_images, image_missing_flag).

    image_missing is True when the record references at least one image path
    but none of the images could be opened from disk.
    Each image is downscaled to fit within max_pixels (width * height).
    """
    if not is_vl:
        return [], False

    paths: List[str] = []
    if record.image_paths:
        paths = record.image_paths
    elif record.image_path:
        paths = [record.image_path]

    if not paths:
        return [], False

    images_pil = [img for p in paths for img in [_open_image(p, max_pixels)] if img is not None]
    image_missing = len(images_pil) == 0
    return images_pil, image_missing


def _build_messages(
    text: str,
    images_pil: List[Image.Image],
    is_vl: bool,
) -> list:
    """Build a chat messages list for Qwen3.5-2B using the loaded prompt."""
    template = _load_prompt_template()

    if images_pil:
        n = len(images_pil)
        images_placeholder = (
            f"[{n} ảnh đính kèm — xem ảnh trong nội dung tin nhắn]"
            if n > 1
            else "[Xem ảnh đính kèm]"
        )
    else:
        images_placeholder = "[Không có ảnh hoặc ảnh không đọc được]"

    # IMPORTANT: do NOT use str.format — the prompt contains raw JSON braces
    # for output contract examples, which would raise KeyError.
    prompt = (
        template
        .replace("{text}", text)
        .replace("{images}", images_placeholder)
    )

    if is_vl and images_pil:
        content: list = [{"type": "image", "image": img} for img in images_pil]
        content.append({"type": "text", "text": prompt})
    else:
        content = [{"type": "text", "text": prompt}]

    return [{"role": "user", "content": content}]


def _extract_json(raw: str) -> dict:
    # Strip <think>...</think> blocks generated by Qwen3/Qwen3.5 in thinking mode
    # before attempting JSON extraction so the greedy regex doesn't grab brace
    # characters that appear inside the thinking section.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(cleaned)


def _validate(data: dict) -> LLMJudgeRecord:
    """Parse and validate the new prompt output schema into a LLMJudgeRecord."""
    raw_label = data.get("Label_LLM1", "INVALID")
    if raw_label == "INVALID":
        label = "INVALID"
    elif raw_label in (0, 1):
        label = int(raw_label)
    elif str(raw_label) in ("0", "1"):
        # Model occasionally returns the label as a quoted string ("0" / "1")
        # instead of a bare integer. Accept both forms.
        label = int(raw_label)
    else:
        label = "INVALID"

    text_only_raw = data.get("Text_Only")
    text_only = int(text_only_raw) if (text_only_raw in (0, 1) or str(text_only_raw) in ("0", "1")) else None

    imageset_only_raw = data.get("ImageSet_Only")
    imageset_only = int(imageset_only_raw) if (imageset_only_raw in (0, 1) or str(imageset_only_raw) in ("0", "1")) else None

    key_images = [
        int(i) for i in (data.get("Key_Images") or [])
        if isinstance(i, (int, float)) and not isinstance(i, bool)
    ]

    difficulty_raw = data.get("Difficulty")
    difficulty = difficulty_raw if difficulty_raw in ("Easy", "Hard") else None

    notes = str(data.get("Notes", ""))[:500]

    reasoning = data.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {}

    return LLMJudgeRecord(
        id=-1,
        label_llm1=label,
        text_only=text_only,
        imageset_only=imageset_only,
        key_images=key_images,
        difficulty=difficulty,
        notes=notes,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Local inference call
# ---------------------------------------------------------------------------

def _call_local(
    model,
    processor,
    messages: list,
    temperature: float,
    max_new_tokens: int = 2048,
) -> str:
    """
    Run a single forward+generate pass using the unified processor API.

    processor.apply_chat_template with tokenize=True + return_dict=True
    handles both text tokenization and image preprocessing in one step,
    producing a dict with input_ids, attention_mask, and pixel_values.
    enable_thinking=False suppresses <think> blocks on Qwen3/Qwen3.5 models
    (ignored by processors that do not support this kwarg).
    """
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for processors that do not accept enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    # Use next(model.parameters()).device so this works with device_map="auto"
    # (model.device is not defined when the model is sharded across devices)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0] if output else ""


# ---------------------------------------------------------------------------
# Per-record judgment
# ---------------------------------------------------------------------------

def judge_single(
    model,
    processor,
    record: InputRecord,
    temperature: float,
    is_vl: bool,
    max_image_pixels: int = 1_048_576,
) -> LLMJudgeRecord:
    """Run local inference for one record with one repair retry on bad JSON."""
    images_pil, image_missing = _load_images(record, is_vl, max_image_pixels)

    if image_missing:
        logger.warning("All images missing for id=%d: %s", record.id, record.image_path)

    messages = _build_messages(record.text, images_pil, is_vl)
    raw = ""

    try:
        raw = _call_local(model, processor, messages, temperature)
        result = _validate(_extract_json(raw))
        result = result.model_copy(update={
            "id": record.id,
            "image_missing": image_missing,
        })
        return result

    except (json.JSONDecodeError, ValueError, KeyError):
        logger.warning("Bad JSON for id=%d, retrying repair. Raw: %s", record.id, raw[:200])
        repair_messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": raw}]},
            {"role": "user", "content": [{"type": "text", "text": _REPAIR_SUFFIX}]},
        ]
        try:
            raw2 = _call_local(model, processor, repair_messages, temperature)
            result2 = _validate(_extract_json(raw2))
            result2 = result2.model_copy(update={
                "id": record.id,
                "image_missing": image_missing,
            })
            return result2
        except Exception as exc2:
            logger.error("Repair failed for id=%d: %s", record.id, exc2)
            return LLMJudgeRecord(
                id=record.id,
                label_llm1="INVALID",
                notes="JSON parse error after retry.",
                parse_error=True,
                image_missing=image_missing,
            )

    except Exception as exc:
        logger.error("Unexpected error for id=%d: %s", record.id, exc)
        return LLMJudgeRecord(
            id=record.id,
            label_llm1="INVALID",
            notes=f"Unexpected error: {str(exc)[:200]}",
            parse_error=True,
            image_missing=image_missing,
        )


# ---------------------------------------------------------------------------
# Batch entry-point
# ---------------------------------------------------------------------------

def judge_batch(
    records: List[InputRecord],
    model_name: str,
    temperature: float,
    hf_token: Optional[str] = None,
    device: str = "cuda",
    load_in_4bit: bool = False,
    max_image_pixels: int = 1_048_576,
) -> List[LLMJudgeRecord]:
    """
    Judge a list of records via local model inference.

    Parameters
    ----------
    records          : list of InputRecord
    model_name       : HuggingFace model ID, e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
    temperature      : sampling temperature (0.1 recommended)
    hf_token         : HF token for downloading the model from Hub
    device           : "cuda" or "cpu"
    load_in_4bit     : enable 4-bit quantization via bitsandbytes to reduce VRAM
    max_image_pixels : cap image width*height before vision encoding to limit VRAM
                       (default 1_048_576 = 1024x1024; set 0 to disable)
    """
    model, processor = load_local_model(model_name, device, load_in_4bit, hf_token)
    is_vl = _is_vl_model(model_name)
    logger.info(
        "Local inference | model=%s | VL=%s | device=%s | 4bit=%s | max_img_px=%s",
        model_name, is_vl, device, load_in_4bit,
        max_image_pixels if max_image_pixels > 0 else "unlimited",
    )

    results: List[LLMJudgeRecord] = []
    for record in tqdm(records, desc="LLM judging", unit="rec"):
        result = judge_single(model, processor, record, temperature, is_vl, max_image_pixels)
        results.append(result)
        logger.debug(
            "id=%d | label=%s | difficulty=%s | parse_err=%s",
            record.id, result.label_llm1, result.difficulty, result.parse_error,
        )

    return results
