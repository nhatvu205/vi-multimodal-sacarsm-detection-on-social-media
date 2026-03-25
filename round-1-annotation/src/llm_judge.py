from __future__ import annotations

"""
LLM judge backed by local model inference (Qwen2.5-VL-7B-Instruct).

Requires CUDA GPU. Designed for Kaggle / Colab environments with free GPU.

The model is downloaded from HuggingFace Hub on first run and cached locally.
Set HF_TOKEN if the model requires Hub authentication:
    export HF_TOKEN="hf_..."
or pass it to judge_batch().

The model singleton is loaded once and reused across all batch calls in the
same process. Supports 4-bit quantization (bitsandbytes) to reduce VRAM usage.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

from .schemas import InputRecord, LLMJudgeRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "llm_sarcasm_vi.txt"
_PROMPT_TEMPLATE: Optional[str] = None

_MODEL = None
_PROCESSOR = None
_LOADED_MODEL_NAME: Optional[str] = None

_REPAIR_SUFFIX = (
    "\n\nPhản hồi trước của bạn không phải JSON hợp lệ. "
    "Hãy chỉ trả về đúng một đối tượng JSON với các trường: "
    "llm_pred_label, llm_prob_sarcastic, llm_confidence, llm_rationale_short. "
    "Không thêm bất kỳ nội dung nào khác ngoài đối tượng JSON."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


def _is_vl_model(model_name: str) -> bool:
    n = model_name.lower()
    return "-vl-" in n or n.endswith("-vl")


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
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
    _MODEL.eval()

    proc_kwargs: dict = {}
    if token:
        proc_kwargs["token"] = token
    _PROCESSOR = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

    _LOADED_MODEL_NAME = model_name
    logger.info("Model loaded successfully.")
    return _MODEL, _PROCESSOR


def _open_image(image_path: str) -> Optional[Image.Image]:
    """
    Open image as a PIL Image (RGB).
    Falls back to resolving relative paths from the repository root when the
    current working directory does not contain the file.
    """
    p = Path(image_path)
    if not p.is_absolute() and not p.exists():
        repo_root_candidate = Path(__file__).resolve().parents[2] / image_path
        if repo_root_candidate.exists():
            p = repo_root_candidate
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return None


def _extract_images_from_messages(messages: list) -> Optional[List[Image.Image]]:
    """Pull PIL Image objects out of a messages list (Qwen chat format)."""
    images = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        images.append(img)
    return images if images else None


def _build_messages(
    text: str,
    image_pil: Optional[Image.Image],
    is_vl: bool,
) -> list:
    """Build a chat messages list in Qwen2.5-VL format."""
    template = _load_prompt_template()
    image_description = (
        "[Xem ảnh đính kèm]" if image_pil else "[Không có ảnh hoặc ảnh không đọc được]"
    )
    # IMPORTANT: do NOT use str.format — the prompt contains raw JSON braces
    # for output contract examples, which would raise KeyError.
    prompt = (
        template.replace("{text}", text)
        .replace("{image_description}", image_description)
    )

    if is_vl and image_pil:
        content = [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [{"type": "text", "text": prompt}]

    return [{"role": "user", "content": content}]


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)


def _validate(data: dict) -> LLMJudgeRecord:
    label = data.get("llm_pred_label", "uncertain")
    if label not in ("sarcastic", "non_sarcastic", "uncertain"):
        label = "uncertain"
    prob = max(0.0, min(1.0, float(data.get("llm_prob_sarcastic", 0.5))))
    conf = max(0.0, min(1.0, float(data.get("llm_confidence", 0.5))))
    rationale = str(data.get("llm_rationale_short", ""))[:500]
    return LLMJudgeRecord(
        id=-1,
        llm_pred_label=label,
        llm_prob_sarcastic=prob,
        llm_confidence=conf,
        llm_rationale_short=rationale,
    )


# ---------------------------------------------------------------------------
# Local inference call
# ---------------------------------------------------------------------------

def _call_local(
    model,
    processor,
    messages: list,
    temperature: float,
    max_new_tokens: int = 512,
) -> str:
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = _extract_images_from_messages(messages)

    proc_kwargs: dict = {
        "text": [text_input],
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        proc_kwargs["images"] = images

    inputs = processor(**proc_kwargs)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
) -> LLMJudgeRecord:
    """Run local inference for one record with one repair retry on bad JSON."""
    image_pil = _open_image(record.image_path) if is_vl else None
    missing_image = is_vl and image_pil is None

    if missing_image:
        logger.warning("Image missing for id=%d: %s", record.id, record.image_path)

    messages = _build_messages(record.text, image_pil, is_vl)
    raw = ""

    try:
        raw = _call_local(model, processor, messages, temperature)
        result = _validate(_extract_json(raw))
        result.id = record.id

        if missing_image:
            result = LLMJudgeRecord(
                id=record.id,
                llm_pred_label="uncertain",
                llm_prob_sarcastic=result.llm_prob_sarcastic,
                llm_confidence=0.0,
                llm_rationale_short=result.llm_rationale_short,
            )
        return result

    except (json.JSONDecodeError, ValueError, KeyError):
        logger.warning("Bad JSON for id=%d, retrying repair. Raw: %s", record.id, raw[:200])
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_SUFFIX},
        ]
        try:
            raw2 = _call_local(model, processor, repair_messages, temperature)
            result2 = _validate(_extract_json(raw2))
            result2.id = record.id
            return result2
        except Exception as exc2:
            logger.error("Repair failed for id=%d: %s", record.id, exc2)
            return LLMJudgeRecord(
                id=record.id,
                llm_pred_label="uncertain",
                llm_prob_sarcastic=0.5,
                llm_confidence=0.0,
                llm_rationale_short="JSON parse error after retry.",
            )

    except Exception as exc:
        logger.error("Unexpected error for id=%d: %s", record.id, exc)
        return LLMJudgeRecord(
            id=record.id,
            llm_pred_label="uncertain",
            llm_prob_sarcastic=0.5,
            llm_confidence=0.0,
            llm_rationale_short=f"Unexpected error: {str(exc)[:100]}",
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
) -> List[LLMJudgeRecord]:
    """
    Judge a list of records via local model inference.

    Parameters
    ----------
    records      : list of InputRecord
    model_name   : HuggingFace model ID, e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
    temperature  : sampling temperature (0.1 recommended)
    hf_token     : HF token for downloading the model from Hub (optional for
                   public models, required for gated ones)
    device       : "cuda" or "cpu"
    load_in_4bit : enable 4-bit quantization via bitsandbytes to reduce VRAM
    """
    model, processor = load_local_model(model_name, device, load_in_4bit, hf_token)
    is_vl = _is_vl_model(model_name)
    logger.info(
        "Local inference | model=%s | VL=%s | device=%s | 4bit=%s",
        model_name, is_vl, device, load_in_4bit,
    )

    results: List[LLMJudgeRecord] = []
    for record in tqdm(records, desc="LLM judging", unit="rec"):
        result = judge_single(model, processor, record, temperature, is_vl)
        results.append(result)
        logger.debug(
            "id=%d | label=%s | prob=%.3f | conf=%.3f",
            record.id, result.llm_pred_label,
            result.llm_prob_sarcastic, result.llm_confidence,
        )

    return results
