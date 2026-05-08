from __future__ import annotations

"""
LLM judge backed by local model inference (InternVL3.5-8B).
Optimized for Kaggle T4 GPU (16GB VRAM) using 4-bit quantization.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from .schemas import InputRecord, LLMJudgeRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "prompt.txt"
_FEW_SHOT_PATH = Path(__file__).parent.parent / "prompts" / "few-short-examples.txt"
_PROMPT_TEMPLATE: Optional[str] = None

_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME: Optional[str] = None

_REPAIR_SUFFIX = (
    "\n\nPhản hồi trước của bạn không phải JSON hợp lệ. "
    "Hãy chỉ trả về đúng một đối tượng JSON với các trường bắt buộc: "
    "reasoning, Label_LLM1, Text_Only, ImageSet_Only, Key_Images, Difficulty. "
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


def load_local_model(
    model_name: str,
    device: str = "cuda",
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple:
    """
    Load InternVL model and tokenizer.
    Uses BitsAndBytesConfig to avoid the 'load_in_4bit' TypeError.
    """
    global _MODEL, _TOKENIZER, _LOADED_MODEL_NAME

    if _MODEL is not None and _LOADED_MODEL_NAME == model_name:
        return _MODEL, _TOKENIZER

    logger.info("Loading InternVL model: %s (4bit=%s)", model_name, load_in_4bit)

    token = hf_token or os.environ.get("HF_TOKEN")
    
    # Cấu hình Quantization để tối ưu VRAM trên Kaggle T4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "quantization_config": bnb_config if load_in_4bit else None,
    }
    if token:
        load_kwargs["token"] = token

    # InternVL sử dụng AutoModel thay vì AutoModelForImageTextToText
    _MODEL = AutoModel.from_pretrained(**load_kwargs).eval()
    
    # Giới hạn số lượng tile ảnh để tránh OOM (Out of Memory)
    _MODEL.config.max_dynamic_patch = 6 

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)

    _LOADED_MODEL_NAME = model_name
    logger.info("InternVL loaded successfully.")
    return _MODEL, _TOKENIZER


def _open_image(image_path: str, max_pixels: int = 500_000) -> Optional[Image.Image]:
    """Open and resize image for InternVL."""
    p = Path(image_path)
    # Logic tìm đường dẫn ảnh giữ nguyên của bạn
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [repo_root / image_path, Path.cwd() / image_path]
    else:
        candidates = [p]

    for candidate in candidates:
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGB")
                # InternVL xử lý resize tốt hơn nếu để ảnh gốc hoặc giảm nhẹ
                return img 
            except Exception as e:
                logger.warning("Error loading image %s: %s", candidate, e)
    return None


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(cleaned)


def _validate(data: dict) -> LLMJudgeRecord:
    """Map InternVL output JSON to project schema with Fine-grained support."""
    
    # Lấy nhãn chính
    raw_label = data.get("llm_label", "INVALID")
    if raw_label in (0, 1, "0", "1"):
        label = int(raw_label)
    else:
        label = "INVALID"

    # Trích xuất reasoning và các nhãn phụ
    reasoning = data.get("reasoning", {})
    
    # Bổ sung thông tin fine-grained vào phần notes hoặc mở rộng LLMJudgeRecord
    # Ở đây ta lồng các nhãn T, I, MM, KI vào trong reasoning để dễ theo dõi
    fine_grained_info = {
        "T": data.get("T"),
        "I": data.get("I"),
        "MM": data.get("MM"),
        "KI": data.get("KI")
    }
    
    # Cập nhật verdict nếu model trả về rời rạc
    if isinstance(reasoning, dict):
        reasoning.update(fine_grained_info)

    return LLMJudgeRecord(
        id=-1,
        label_llm1=label,
        has_emoji=int(data.get("has_emoji", 0)),
        needs_human_check=int(data.get("needs_human_check", 1)),
        notes=f"T:{fine_grained_info['T']} I:{fine_grained_info['I']} MM:{fine_grained_info['MM']} KI:{fine_grained_info['KI']}",
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# InternVL Inference Call
# ---------------------------------------------------------------------------

def judge_single(
    model,
    tokenizer,
    record: InputRecord,
    temperature: float,
    is_vl: bool = True,
    max_image_pixels: int = 500_000,
) -> LLMJudgeRecord:
    """Run inference using InternVL's custom .chat() API."""
    pixel_values = _open_image(record.image_path)
    image_missing = pixel_values is None

    template = _load_prompt_template()
    prompt = (
        template
        .replace("{text}", record.text)
        .replace("{images}", "[Ảnh đính kèm]" if not image_missing else "[Không có ảnh]")
        .replace("{ocr_text}", record.ocr_text or "[Không có OCR]")
    )

    # Cấu hình sinh văn bản
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=True if temperature > 0 else False,
        temperature=temperature if temperature > 0 else None,
    )

    try:
        # InternVL Chat API
        response, _ = model.chat(
            tokenizer, 
            pixel_values, 
            prompt, 
            generation_config
        )
        
        result = _validate(_extract_json(response))
        return result.model_copy(update={"id": record.id, "image_missing": image_missing})

    except Exception as e:
        logger.error("Inference failed for id=%d: %s", record.id, e)
        return LLMJudgeRecord(
            id=record.id,
            label_llm1="INVALID",
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
    max_image_pixels: int = 500_000,
) -> List[LLMJudgeRecord]:
    """Entry point for batch processing."""
    model, tokenizer = load_local_model(model_name, device, load_in_4bit, hf_token)
    
    results: List[LLMJudgeRecord] = []
    for record in records:
        res = judge_single(model, tokenizer, record, temperature)
        results.append(res)
    return results
