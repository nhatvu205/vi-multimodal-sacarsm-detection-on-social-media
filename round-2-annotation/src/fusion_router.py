from __future__ import annotations
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .schemas import LLMJudgeRecord, Round1OutputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouterConfig:
    """
    Routing configuration cho Round-2 Fine-grained Annotation.
    
    random_audit_rate : Tỷ lệ lấy mẫu ngẫu nhiên các bản ghi LLM tự tin
                        để human review (quality control).
    seed              : Seed để audit sampling reproducible.
    """
    random_audit_rate: float = 0.08   # Giảm nhẹ so với Round 1
    seed: int = 42


def route_single(
    llm_rec: LLMJudgeRecord,
    cfg: RouterConfig,
    text: str,
    image_path: str,
    route_reason_override: Optional[str] = None,
) -> Round1OutputRecord:
    """
    Quyết định routing cho một record trong Round 2.
    
    Logic ưu tiên:
    1. missing_image hoặc parse_error → cần review
    2. label_llm1 = "INVALID" → cần review
    3. needs_human_check == 0 → auto accept (high confidence)
    4. needs_human_check == 1 hoặc None → cần review
    """
    label = llm_rec.label_llm1
    needs_human_check = llm_rec.needs_human_check

    # Xác định round1_label
    if label == 1:
        round1_label = "sarcastic"
    elif label == 0:
        round1_label = "non_sarcastic"
    else:
        round1_label = "invalid"

    # --- Xác định routing reason ---
    if route_reason_override == "missing_image":
        need_review = True
        route_reason = "missing_image"
    elif route_reason_override == "invalid_json":
        need_review = True
        route_reason = "invalid_json"
        round1_label = "invalid"
    elif label == "INVALID":
        need_review = True
        route_reason = "uncertain"
    elif needs_human_check == 0:
        # Tự tin → có thể auto-accept
        need_review = False
        route_reason = "high_conf"
    else:
        # Cần human review
        need_review = True
        route_reason = "low_conf"

    # Optional: Tăng độ strict nếu Multimodal yếu
    # if llm_rec.MM == 0 and round1_label == "sarcastic":
    #     need_review = True
    #     route_reason = "low_conf_mm"

    return Round1OutputRecord(
        id=llm_rec.id,
        text=text,
        image_path=image_path,
        label_llm1=label,
        
        # Fine-grained fields (Round 2)
        T=llm_rec.T,
        I=llm_rec.I,
        MM=llm_rec.MM,
        KI=llm_rec.KI,
        
        has_emoji=llm_rec.has_emoji,
        needs_human_check=needs_human_check,
        notes=llm_rec.notes,
        reasoning=llm_rec.reasoning,
        
        round1_label=round1_label,
        need_review=need_review,
        route_reason=route_reason,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def apply_audit_sampling(
    records: List[Round1OutputRecord],
    audit_rate: float,
    seed: int,
) -> Tuple[List[Round1OutputRecord], int]:
    """
    Lấy mẫu ngẫu nhiên một phần bản ghi đã auto-accept để human kiểm tra.
    """
    rng = random.Random(seed)
    
    auto_accepted_indices = [
        i for i, r in enumerate(records)
        if r.round1_label in ("sarcastic", "non_sarcastic") and not r.need_review
    ]
    
    k = max(0, round(len(auto_accepted_indices) * audit_rate))
    sampled_indices = set(rng.sample(auto_accepted_indices, k) if k > 0 else [])

    updated = []
    for i, rec in enumerate(records):
        if i in sampled_indices:
            rec = rec.model_copy(
                update={
                    "need_review": True,
                    "route_reason": "audit_sampled",
                }
            )
        updated.append(rec)

    logger.info(
        "Audit sampling: %d auto-accepted candidates → %d sampled (rate=%.3f)",
        len(auto_accepted_indices), k, audit_rate
    )
    return updated, k


def route_all(
    records_input: list,
    llm_results: List[LLMJudgeRecord],
    cfg: RouterConfig,
) -> List[Round1OutputRecord]:
    """Route toàn bộ records dựa trên kết quả LLM."""
    llm_by_id = {r.id: r for r in llm_results}
    routed: List[Round1OutputRecord] = []

    for inp in records_input:
        llm_rec = llm_by_id.get(inp.id)
        if llm_rec is None:
            logger.warning("No LLM result for id=%d, skipping", inp.id)
            continue

        override = None
        if getattr(llm_rec, 'image_missing', False):
            override = "missing_image"
        elif getattr(llm_rec, 'parse_error', False):
            override = "invalid_json"

        out = route_single(
            llm_rec=llm_rec,
            cfg=cfg,
            text=inp.text,
            image_path=inp.image_path,
            route_reason_override=override
        )
        routed.append(out)

    return routed
