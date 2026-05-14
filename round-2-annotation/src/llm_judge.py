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

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "prompt.txt"
_FEW_SHOT_PATH = Path(__file__).parent.parent / "prompts" / "few-short-examples.txt"
_PROMPT_TEMPLATE: Optional[str] = None

_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME: Optional[str] = None

# ====================== HELPERS ======================

def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        if _FEW_SHOT_PATH.exists():
            few_shot = _FEW_SHOT_PATH.read_text(encoding="utf-8")
            prompt = prompt.rstrip() + "\n\n" + few_shot.lstrip()
        _PROMPT_TEMPLATE = prompt
    return _PROMPT_TEMPLATE

def _extract_json(raw: str) -> dict:
    """Extract last valid JSON object from model response via balanced brace matching."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Balanced brace matching — finds all top-level JSON objects
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start:i + 1])
                start = None

    # Try from last to first; prefer one that has "llm_label"
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "llm_label" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: any parseable dict
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
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

def judge_single(model, tokenizer, record, temperature, max_image_pixels=500000):
    pixel_values = None
    image_missing = True
    
    try:
        if record.image_path and os.path.exists(record.image_path):
            try:
                img = Image.open(record.image_path).convert("RGB")
                transform = build_transform()
                pixel_values = transform(img).unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
                image_missing = False
            except Exception as e:
                logger.warning(f"Failed to load image {record.image_path}: {e}")

        template = _load_prompt_template()
        prompt = (
            template
            .replace("{text}", str(record.text or ""))
            .replace("{images}", "[Ảnh đính kèm]" if not image_missing else "[Không có ảnh]")
            .replace("{ocr_text}", str(record.ocr_text or "[Không có]"))
        )

        generation_config = dict(
            max_new_tokens=2048,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
        )
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.chat(tokenizer, pixel_values, prompt, generation_config)
        response = outputs[0] if isinstance(outputs, tuple) else outputs
        
        logger.info("RAW RESPONSE: %s", repr(response[:500]))
        
        data = _extract_json(response)
        logger.info("EXTRACTED JSON: %s", repr(data))

        raw_label = data.get("llm_label", "INVALID")
        label = int(raw_label) if str(raw_label) in ("0", "1") else "INVALID"

        return LLMJudgeRecord(
            id=record.id,
            label_llm2=label,
            has_emoji=int(data.get("has_emoji", 0)),
            needs_human_check=int(data.get("needs_human_check", 1)),
            notes=f"T:{data.get('T')} | I:{data.get('I')} | MM:{data.get('MM')} | KI:{data.get('KI')}",
            reasoning=data.get("reasoning", {}),
            image_missing=image_missing,
            T=data.get("T"),
            I=data.get("I"),
            MM=data.get("MM"),
            KI=data.get("KI")
        )

    except Exception as e:
        logger.error(f"Inference error for id={record.id}: {e}")
        return LLMJudgeRecord(
            id=record.id,
            label_llm2="INVALID",
            notes=f"Error: {str(e)[:200]}",
            parse_error=True,
            image_missing=image_missing
        )
    
    finally:
        # ✅ LUÔN giải phóng, dù thành công hay lỗi
        if pixel_values is not None:
            del pixel_values
        torch.cuda.empty_cache()

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
