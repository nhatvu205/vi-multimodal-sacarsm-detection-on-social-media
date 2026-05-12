from __future__ import annotations

"""
LLM judge backed by local model inference (InternVL3.5-8B).
Optimized for Kaggle T4 GPU (16GB VRAM) using 4-bit quantization.

Fixes vs previous version:
  - dynamic_preprocess: ảnh được chia thành nhiều tiles đúng cách
  - load_image: dùng đúng API của InternVL (không phải single-transform)
  - judge_single: prompt có <image> token, pixel_values shape đúng
  - judge_single: model.chat() return value được unpack an toàn
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig

from .schemas import InputRecord, LLMJudgeRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt paths
# ---------------------------------------------------------------------------
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_PROMPT_TURN1_PATH = _PROMPT_DIR / "prompt_turn1_text.txt"
_PROMPT_TURN2_PATH = _PROMPT_DIR / "prompt_turn2_image.txt"
_PROMPT_TURN3_PATH = _PROMPT_DIR / "prompt_turn3_multimodal.txt"

_PROMPT_T1: Optional[str] = None
_PROMPT_T2: Optional[str] = None
_PROMPT_T3: Optional[str] = None

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME: Optional[str] = None

# ---------------------------------------------------------------------------
# InternVL image utils  (ported từ HuggingFace official example)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Tile ảnh thành nhiều patches theo aspect ratio — bắt buộc với InternVL."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width  = int(image_size * target_aspect_ratio[0])
    target_height = int(image_size * target_aspect_ratio[1])
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        col = i % (target_width // image_size)
        row = i // (target_width // image_size)
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_path: str, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
    """
    Load ảnh và trả về pixel_values shape (N, 3, H, W) — N = số tiles.
    max_num=6 phù hợp với T4 16GB.  Tăng lên 12 nếu có A100.
    """
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(tile) for tile in tiles])  # (N, 3, H, W)
    return pixel_values


# ---------------------------------------------------------------------------
# Prompt loaders
# ---------------------------------------------------------------------------
def _load_prompts() -> Tuple[str, str, str]:
    global _PROMPT_T1, _PROMPT_T2, _PROMPT_T3
    if _PROMPT_T1 is None:
        _PROMPT_T1 = _PROMPT_TURN1_PATH.read_text(encoding="utf-8")
        logger.info("Loaded Turn-1 prompt (%d chars)", len(_PROMPT_T1))
    if _PROMPT_T2 is None:
        _PROMPT_T2 = _PROMPT_TURN2_PATH.read_text(encoding="utf-8")
        logger.info("Loaded Turn-2 prompt (%d chars)", len(_PROMPT_T2))
    if _PROMPT_T3 is None:
        _PROMPT_T3 = _PROMPT_TURN3_PATH.read_text(encoding="utf-8")
        logger.info("Loaded Turn-3 prompt (%d chars)", len(_PROMPT_T3))
    return _PROMPT_T1, _PROMPT_T2, _PROMPT_T3


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> dict:
    """Extract JSON từ model response, bỏ qua <think> tags và garbage text."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Thử parse toàn bộ trước
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Tìm block JSON đầu tiên
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _safe_chat(model, tokenizer, pixel_values, prompt: str, generation_config: dict) -> str:
    """
    Wrap model.chat() — unpack an toàn vì một số version trả (response, history),
    version khác trả response thẳng.
    """
    outputs = model.chat(tokenizer, pixel_values, prompt, generation_config)
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_local_model(
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    device: str = "cuda",
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple:
    global _MODEL, _TOKENIZER, _LOADED_MODEL_NAME

    if _MODEL is not None and _LOADED_MODEL_NAME == model_name:
        return _MODEL, _TOKENIZER

    token = hf_token.strip() if hf_token and isinstance(hf_token, str) and hf_token.strip() else None
    logger.info("Loading model %s | 4bit=%s | token=%s", model_name, load_in_4bit, bool(token))

    # Monkey-patch để tránh AttributeError trên một số version transformers
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)
    from transformers import dynamic_module_utils
    model_class = dynamic_module_utils.get_class_from_dynamic_module(
        config.auto_map["AutoModel"], model_name
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = property(lambda self: {})

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if load_in_4bit
        else None
    )

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token, use_fast=False
    )
    _MODEL = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    ).eval()

    _MODEL.config.max_dynamic_patch = 6
    _LOADED_MODEL_NAME = model_name
    logger.info("✅ Model loaded successfully!")
    return _MODEL, _TOKENIZER


# ---------------------------------------------------------------------------
# 3-turn inference
# ---------------------------------------------------------------------------
def _run_turn1(
    model, tokenizer, record: InputRecord, generation_config: dict, prompt_t1: str
) -> dict:
    """Turn 1 — text only. pixel_values=None."""
    prompt = (
        prompt_t1
        .replace("{text}", str(record.text or ""))
        .replace("{ocr_text}", str(record.ocr_text or "[Không có]"))
    )
    raw = _safe_chat(model, tokenizer, None, prompt, generation_config)
    result = _extract_json(raw)
    if not result:
        logger.warning("Turn-1 parse failed for id=%d | raw: %s", record.id, raw[:200])
    return result


def _run_turn2(
    model, tokenizer, record: InputRecord, generation_config: dict,
    prompt_t2: str, pixel_values: Optional[torch.Tensor], image_missing: bool
) -> dict:
    """Turn 2 — image only. Không đưa text caption vào prompt."""
    prompt = (
        prompt_t2
        .replace("{ocr_text}", str(record.ocr_text or "[Không có]"))
    )
    # Thêm <image> token vào đầu prompt nếu có ảnh
    if not image_missing and pixel_values is not None:
        prompt = "<image>\n" + prompt

    raw = _safe_chat(
        model, tokenizer,
        pixel_values if not image_missing else None,
        prompt, generation_config
    )
    result = _extract_json(raw)
    if not result:
        logger.warning("Turn-2 parse failed for id=%d | raw: %s", record.id, raw[:200])
    return result


def _run_turn3(
    model, tokenizer, record: InputRecord, generation_config: dict,
    prompt_t3: str, pixel_values: Optional[torch.Tensor],
    image_missing: bool, t1: dict, t2: dict
) -> dict:
    """Turn 3 — multimodal, nhận kết quả T/I từ lượt trước."""
    prompt = (
        prompt_t3
        .replace("{text}", str(record.text or ""))
        .replace("{ocr_text}", str(record.ocr_text or "[Không có]"))
        # Inject Turn-1 results
        .replace("{T}", str(t1.get("T", "N/A")))
        .replace("{T_confidence}", str(t1.get("T_confidence", "low")))
        .replace("{T_signals}", str(t1.get("T_signals", [])))
        .replace("{T_reason}", str(t1.get("T_reason", "")))
        # Inject Turn-2 results
        .replace("{I}", str(t2.get("I", "N/A")))
        .replace("{I_confidence}", str(t2.get("I_confidence", "low")))
        .replace("{I_category}", str(t2.get("I_category", "F")))
        .replace("{I_description}", str(t2.get("I_description", "")))
        .replace("{I_reason}", str(t2.get("I_reason", "")))
    )
    if not image_missing and pixel_values is not None:
        prompt = "<image>\n" + prompt

    raw = _safe_chat(
        model, tokenizer,
        pixel_values if not image_missing else None,
        prompt, generation_config
    )
    result = _extract_json(raw)
    if not result:
        logger.warning("Turn-3 parse failed for id=%d | raw: %s", record.id, raw[:200])
    return result


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def judge_single(
    model,
    tokenizer,
    record: InputRecord,
    temperature: float,
    max_num_tiles: int = 6,
) -> LLMJudgeRecord:
    """
    3-turn inference cho một record.
    Turn 1: text only → T
    Turn 2: image only → I
    Turn 3: text + image + T/I → MM, llm_label
    """
    prompt_t1, prompt_t2, prompt_t3 = _load_prompts()

    generation_config = dict(
        max_new_tokens=1024,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )

    # --- Load image (dùng chung cho Turn 2 và Turn 3) ---
    pixel_values: Optional[torch.Tensor] = None
    image_missing = True

    if record.image_path and os.path.exists(record.image_path):
        try:
            pixel_values = (
                load_image(record.image_path, input_size=448, max_num=max_num_tiles)
                .to(torch.bfloat16)
                .to(model.device)
            )
            image_missing = False
            logger.debug(
                "id=%d | pixel_values shape: %s", record.id, tuple(pixel_values.shape)
            )
        except Exception as e:
            logger.warning("Image load failed for id=%d: %s", record.id, e)

    try:
        # ── Turn 1: text only ──────────────────────────────────────────────
        t1 = _run_turn1(model, tokenizer, record, generation_config, prompt_t1)

        # ── Turn 2: image only ────────────────────────────────────────────
        t2 = _run_turn2(
            model, tokenizer, record, generation_config,
            prompt_t2, pixel_values, image_missing
        )

        # ── Turn 3: multimodal fusion ─────────────────────────────────────
        t3 = _run_turn3(
            model, tokenizer, record, generation_config,
            prompt_t3, pixel_values, image_missing, t1, t2
        )

        # ── Parse final output từ Turn 3 ──────────────────────────────────
        raw_label = t3.get("llm_label", "INVALID")
        label = int(raw_label) if str(raw_label) in ("0", "1") else "INVALID"

        # Turn 3 có thể override T và I → dùng giá trị đã override
        T_final = t3.get("T", t1.get("T"))
        I_final = t3.get("I", t2.get("I"))

        # needs_human_check: OR của cả 3 lượt
        needs_human = int(
            bool(t1.get("needs_human_check", 0))
            or bool(t2.get("needs_human_check", 0))
            or bool(t3.get("needs_human_check", 0))
        )

        return LLMJudgeRecord(
            id=record.id,
            label_llm2=label,
            T=T_final,
            I=I_final,
            MM=t3.get("MM"),
            KI=t3.get("KI"),
            has_emoji=int(t1.get("has_emoji", 0) or t3.get("has_emoji", 0)),
            needs_human_check=needs_human,
            notes=(
                f"T:{T_final}(conf={t1.get('T_confidence','?')}) | "
                f"I:{I_final}(conf={t2.get('I_confidence','?')},cat={t2.get('I_category','?')}) | "
                f"MM:{t3.get('MM')} | KI:{t3.get('KI')} | "
                f"override T={t3.get('T_overridden',False)} I={t3.get('I_overridden',False)}"
            ),
            reasoning=t3.get("reasoning", {}),
            image_missing=image_missing,
        )

    except Exception as e:
        logger.error("Inference error for id=%d: %s", record.id, e, exc_info=True)
        return LLMJudgeRecord(
            id=record.id,
            label_llm2="INVALID",
            notes=f"Error: {str(e)[:200]}",
            parse_error=True,
            image_missing=image_missing,
        )


def judge_batch(
    records: List[InputRecord],
    model_name: str,
    temperature: float,
    hf_token: Optional[str] = None,
    device: str = "cuda",
    load_in_4bit: bool = True,
    max_image_pixels: int = 500_000,  # giữ signature cũ cho tương thích
    max_num_tiles: int = 6,
) -> List[LLMJudgeRecord]:
    """Batch entry-point cho pipeline. Load model 1 lần, loop từng record."""
    model, tokenizer = load_local_model(model_name, device, load_in_4bit, hf_token)

    results = []
    for i, record in enumerate(records):
        logger.info("Processing record %d/%d (id=%d)", i + 1, len(records), record.id)
        res = judge_single(model, tokenizer, record, temperature, max_num_tiles)
        results.append(res)

    return results