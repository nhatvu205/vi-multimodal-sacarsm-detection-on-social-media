from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from .schemas import InputRecord, LLMJudgeRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "prompt.txt"
_FEW_SHOT_PATH = Path(__file__).parent.parent / "prompts" / "few-short-examples.txt"

_MODEL = None
_PROCESSOR = None
_LOADED_MODEL_NAME: Optional[str] = None

_REPAIR_SUFFIX = """
Phản hồi trước của bạn không phải là JSON hợp lệ. 
Hãy trả về đúng một đối tượng JSON theo đúng format yêu cầu, không thêm bất kỳ nội dung nào khác.
"""

# ====================== HELPERS ======================
def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if '_PROMPT_TEMPLATE' not in globals():
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        if _FEW_SHOT_PATH.exists():
            few_shot = _FEW_SHOT_PATH.read_text(encoding="utf-8")
            prompt = prompt.rstrip() + "\n\n" + few_shot.lstrip()
        _PROMPT_TEMPLATE = prompt
    return _PROMPT_TEMPLATE


def load_local_model(
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    load_in_4bit: bool = False,
    hf_token: Optional[str] = None
):
    global _MODEL, _PROCESSOR, _LOADED_MODEL_NAME

    if _MODEL is not None and _LOADED_MODEL_NAME == model_name:
        return _MODEL, _PROCESSOR

    logger.info(f"Loading InternVL3.5-8B (4bit={load_in_4bit}) ...")

    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    _MODEL = AutoModel.from_pretrained(**kwargs)

    # === PATCH QUAN TRỌNG ===
    if not hasattr(_MODEL, "all_tied_weights_keys"):
        _MODEL.all_tied_weights_keys = {}
        logger.info("✅ Patched missing 'all_tied_weights_keys'")

    _MODEL.eval()

    _PROCESSOR = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _LOADED_MODEL_NAME = model_name

    logger.info("✅ InternVL3.5-8B loaded successfully!")
    return _MODEL, _PROCESSOR


def _resize_image(img: Image.Image, max_pixels: int = 500000) -> Image.Image:
    w, h = img.size
    if max_pixels <= 0 or w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _open_image(image_path: str, max_pixels: int = 500000) -> Optional[Image.Image]:
    """Mở ảnh từ nhiều đường dẫn có thể"""
    p = Path(image_path)
    candidates = [p] if p.is_absolute() else [
        p,
        Path("/kaggle/working") / p,
        Path.cwd() / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGB")
                return _resize_image(img, max_pixels)
            except Exception as e:
                logger.warning(f"Cannot open image {candidate}: {e}")
    logger.warning(f"Image not found: {image_path}")
    return None


def _load_images(record: InputRecord, max_pixels: int = 500000) -> Tuple[List[Image.Image], bool]:
    """Load images và trả về flag missing"""
    paths = []
    if hasattr(record, 'image_paths') and record.image_paths:
        paths = record.image_paths
    elif hasattr(record, 'image_path') and record.image_path:
        paths = [record.image_path]

    images = [img for p in paths if (img := _open_image(p, max_pixels)) is not None]
    image_missing = len(images) == 0 and len(paths) > 0
    return images, image_missing


def _build_messages(text: str, images_pil: List[Image.Image], prompt: str):
    content = [{"type": "image", "image": img} for img in images_pil]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(cleaned)


def _validate(data: dict) -> LLMJudgeRecord:
    """Parse output từ InternVL theo schema Round 2"""
    raw_label = data.get("llm_label")
    label = "INVALID"
    if isinstance(raw_label, int) or str(raw_label) in ("0", "1"):
        label = int(raw_label)
    elif raw_label == "INVALID":
        label = "INVALID"

    has_emoji = int(data.get("has_emoji", 0))
    needs_human = int(data.get("needs_human_check", 1))

    return LLMJudgeRecord(
        id=-1,
        label_llm1=label,
        has_emoji=has_emoji,
        needs_human_check=needs_human,
        notes=str(data.get("notes", ""))[:500],
        reasoning=data.get("reasoning", {}),
        # Các field mới của Round 2
        T=data.get("T"),
        I=data.get("I"),
        MM=data.get("MM"),
        KI=data.get("KI"),
    )


# ====================== INFERENCE ======================
def judge_single(
    record: InputRecord,
    temperature: float = 0.1,
    max_image_pixels: int = 500000,
    max_new_tokens: int = 512,
) -> LLMJudgeRecord:
    
    model, processor = load_local_model()
    images_pil, image_missing = _load_images(record, max_image_pixels)

    # Build prompt
    template = _load_prompt_template()
    prompt = template.replace("{text}", str(record.text or ""))
    prompt = prompt.replace("{images}", f"Tổng số ảnh: {len(images_pil)}")
    prompt = prompt.replace("{ocr_text}", str(getattr(record, 'ocr_text', '[Không có OCR text]')))

    messages = _build_messages(record.text or "", images_pil, prompt)

    # Inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    output = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    try:
        parsed = _extract_json(output)
        result = _validate(parsed)
        result = result.model_copy(update={
            "id": record.id,
            "image_missing": image_missing
        })
        return result

    except Exception as e:
        logger.warning(f"JSON parse failed for id={record.id}, retrying...")
        # Có thể thêm retry logic sau nếu cần
        return LLMJudgeRecord(
            id=record.id,
            label_llm1="INVALID",
            notes="JSON parse error",
            parse_error=True,
            image_missing=image_missing
        )


def judge_batch(
    records: List[InputRecord],
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    temperature: float = 0.1,
    hf_token: Optional[str] = None,
    max_image_pixels: int = 500000,
    max_new_tokens: int = 512,
) -> List[LLMJudgeRecord]:
    
    logger.info(f"Starting batch inference with {len(records)} records | Model: {model_name}")
    
    results = []
    for i, record in enumerate(records):
        result = judge_single(
            record=record,
            temperature=temperature,
            max_image_pixels=max_image_pixels,
            max_new_tokens=max_new_tokens
        )
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(records)} records")

    return results
