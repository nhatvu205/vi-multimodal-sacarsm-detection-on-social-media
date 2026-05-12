from __future__ import annotations

"""
LLM judge backed by local model inference (InternVL3.5-8B).
Optimized for Kaggle T4 GPU (16GB VRAM) using 4-bit quantization.
Includes patches for AttributeError and safe JSON extraction.
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

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_PROMPT_T1_PATH = _PROMPTS_DIR / "prompt_turn1_text.txt"
_PROMPT_T2_PATH = _PROMPTS_DIR / "prompt_turn2_image.txt"
_PROMPT_T3_PATH = _PROMPTS_DIR / "prompt_turn3_multimodal.txt"

_PROMPT_T1: Optional[str] = None
_PROMPT_T2: Optional[str] = None
_PROMPT_T3: Optional[str] = None
_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME: Optional[str] = None

# ====================== HELPERS ======================

def _load_prompts() -> tuple[str, str, str]:
    """Load và cache cả 3 prompt templates."""
    global _PROMPT_T1, _PROMPT_T2, _PROMPT_T3
    if _PROMPT_T1 is None:
        _PROMPT_T1 = _PROMPT_T1_PATH.read_text(encoding="utf-8")
    if _PROMPT_T2 is None:
        _PROMPT_T2 = _PROMPT_T2_PATH.read_text(encoding="utf-8")
    if _PROMPT_T3 is None:
        _PROMPT_T3 = _PROMPT_T3_PATH.read_text(encoding="utf-8")
    return _PROMPT_T1, _PROMPT_T2, _PROMPT_T3

def _extract_json(raw: str) -> dict:
    """Extract JSON from model response, handling thinking tags and garbage text."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

def build_transform(input_size=448):
    """Standard InternVL image transformation."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
def _build_prompt_t1(text: str, ocr_text: str, prompt_template: str) -> str:
    return (
        prompt_template
        .replace("{text}", text or "")
        .replace("{ocr_text}", ocr_text or "[Không có]")
    )

def _build_prompt_t2(ocr_text: str, prompt_template: str) -> str:
    # {image} KHÔNG replace ở đây — InternVL nhận pixel_values riêng
    return prompt_template.replace("{ocr_text}", ocr_text or "[Không có]")

def _build_prompt_t3(
    text: str,
    ocr_text: str,
    t1: dict,
    t2: dict,
    prompt_template: str,
) -> str:
    return (
        prompt_template
        .replace("{text}", text or "")
        .replace("{ocr_text}", ocr_text or "[Không có]")
        # Turn 1 results
        .replace("{T}", str(t1.get("T", 0)))
        .replace("{T_confidence}", t1.get("T_confidence", "low"))
        .replace("{T_signals}", str(t1.get("T_signals", [])))
        .replace("{T_reason}", t1.get("T_reason", ""))
        # Turn 2 results
        .replace("{I}", str(t2.get("I", 0)))
        .replace("{I_confidence}", t2.get("I_confidence", "low"))
        .replace("{I_category}", t2.get("I_category", "F"))
        .replace("{I_description}", t2.get("I_description", ""))
        .replace("{I_reason}", t2.get("I_reason", ""))
        # {image} giữ nguyên, InternVL xử lý riêng
    )

def _call_model(
    model,
    tokenizer,
    prompt: str,
    pixel_values,   # None nếu text-only
    temperature: float,
) -> str:
    """Gọi model.chat() và trả về raw string response."""
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )
    outputs = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def load_local_model(
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    device: str = "cuda",
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple:
    global _MODEL, _TOKENIZER, _LOADED_MODEL_NAME
    
    if _MODEL is not None and _LOADED_MODEL_NAME == model_name:
        return _MODEL, _TOKENIZER

    # Xử lý token cực kỳ nghiêm ngặt
    token_to_use = hf_token.strip() if hf_token and isinstance(hf_token, str) and hf_token.strip() else None
    
    logger.info(f"Loading model {model_name} | 4bit={load_in_4bit} | token={'Yes' if token_to_use else 'No'}")

    # Monkey Patch
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    from transformers import dynamic_module_utils
    model_class = dynamic_module_utils.get_class_from_dynamic_module(config.auto_map["AutoModel"], model_name)
    
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = property(lambda self: {})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if load_in_4bit else None

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        token=token_to_use
    )

    _MODEL = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token_to_use
    ).eval()

    _MODEL.config.max_dynamic_patch = 6
    _LOADED_MODEL_NAME = model_name

    logger.info("✅ InternVL3.5-8B loaded successfully!")
    return _MODEL, _TOKENIZER

# ====================== INFERENCE ======================

def judge_single(
    model,
    tokenizer,
    record: InputRecord,
    temperature: float,
    max_image_pixels: int = 500000,
) -> LLMJudgeRecord:
    pt1, pt2, pt3 = _load_prompts()
    ocr_text = record.ocr_text or ""
    text = record.text or ""

    # ── Xử lý ảnh ──────────────────────────────────────────────
    pixel_values = None
    image_missing = True
    if record.image_path and os.path.exists(record.image_path):
        try:
            img = Image.open(record.image_path).convert("RGB")
            transform = build_transform()
            pixel_values = transform(img).unsqueeze(0).to(torch.bfloat16).to(model.device)
            image_missing = False
        except Exception as e:
            logger.warning(f"Failed to load image {record.image_path}: {e}")

    try:
        # ── TURN 1: Text-only ───────────────────────────────────
        prompt_t1 = _build_prompt_t1(text, ocr_text, pt1)
        raw_t1 = _call_model(model, tokenizer, prompt_t1, None, temperature)
        t1 = _extract_json(raw_t1)
        logger.debug(f"[id={record.id}] T1 raw: {raw_t1[:200]}")

        # ── TURN 2: Image-only ──────────────────────────────────
        prompt_t2 = _build_prompt_t2(ocr_text, pt2)
        # Nếu ảnh bị missing → force I=0, skip inference
        if image_missing:
            t2 = {"I": 0, "I_confidence": "low", "I_category": "F",
                  "I_description": "Image missing", "I_reason": "Không tải được ảnh"}
        else:
            raw_t2 = _call_model(model, tokenizer, prompt_t2, pixel_values, temperature)
            t2 = _extract_json(raw_t2)
            logger.debug(f"[id={record.id}] T2 raw: {raw_t2[:200]}")

        # ── TURN 3: Multimodal tổng hợp ─────────────────────────
        prompt_t3 = _build_prompt_t3(text, ocr_text, t1, t2, pt3)
        raw_t3 = _call_model(model, tokenizer, prompt_t3, pixel_values, temperature)
        t3 = _extract_json(raw_t3)
        logger.debug(f"[id={record.id}] T3 raw: {raw_t3[:200]}")

        # ── Parse final label ───────────────────────────────────
        raw_label = t3.get("llm_label", "INVALID")
        label = int(raw_label) if str(raw_label) in ("0", "1") else "INVALID"

        return LLMJudgeRecord(
            id=record.id,
            label_llm2=label,
            # Turn 1
            T=t1.get("T"),
            T_confidence=t1.get("T_confidence"),
            T_signals=t1.get("T_signals"),
            T_reason=t1.get("T_reason"),
            T_overridden=t3.get("T_overridden", False),
            # Turn 2
            I=t2.get("I"),
            I_confidence=t2.get("I_confidence"),
            I_category=t2.get("I_category"),
            I_description=t2.get("I_description"),
            I_reason=t2.get("I_reason"),
            I_overridden=t3.get("I_overridden", False),
            # Turn 3
            MM=t3.get("MM"),
            MM_pattern=t3.get("MM_pattern"),
            KI=t3.get("KI"),
            reasoning=t3.get("reasoning", {}),
            has_emoji=int(t1.get("has_emoji", 0)),
            needs_human_check=int(t3.get("needs_human_check", 1)),
            image_missing=image_missing,
        )

    except Exception as e:
        logger.error(f"Inference error for id={record.id}: {e}")
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
    max_image_pixels: int = 500000,
) -> List[LLMJudgeRecord]:
    """Batch entry-point for pipeline."""
    model, tokenizer = load_local_model(model_name, device, load_in_4bit, hf_token)
    
    results = []
    for record in records:
        res = judge_single(model, tokenizer, record, temperature, max_image_pixels)
        results.append(res)
        
    return results
