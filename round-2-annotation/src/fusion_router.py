from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .schemas import LLMJudgeRecord, Round2OutputRecord


@dataclass
class RouterConfig:
    random_audit_rate: float = 0.08
    seed: int = 42


def route_single(
    llm_rec: LLMJudgeRecord,
    cfg: RouterConfig,
    text: str,
    image_path: str,
    route_reason_override: Optional[str] = None,
) -> Round2OutputRecord:
    del cfg
    label = llm_rec.label_llm2
    needs_human_check = llm_rec.needs_human_check

    if label == 1:
        round2_label = "sarcastic"
    elif label == 0:
        round2_label = "non_sarcastic"
    else:
        round2_label = "invalid"

    if route_reason_override == "missing_image":
        need_review = True
        route_reason = "missing_image"
    elif route_reason_override == "invalid_json":
        need_review = True
        route_reason = "invalid_json"
        round2_label = "invalid"
    elif label in ("INVALID", -1):
        need_review = True
        route_reason = "uncertain"
    elif needs_human_check == 0:
        need_review = False
        route_reason = "high_conf"
    else:
        need_review = True
        route_reason = "low_conf"

    return Round2OutputRecord(
        id=llm_rec.id,
        text=text,
        image_path=image_path,
        label_llm2=label,
        T=llm_rec.T,
        I=llm_rec.I,
        MM=llm_rec.MM,
        KI=llm_rec.KI,
        has_emoji=llm_rec.has_emoji,
        needs_human_check=needs_human_check,
        notes=llm_rec.notes,
        reasoning=llm_rec.reasoning,
        round2_label=round2_label,
        need_review=need_review,
        route_reason=route_reason,
        parse_error=llm_rec.parse_error,
        image_missing=llm_rec.image_missing,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def apply_audit_sampling(
    records: List[Round2OutputRecord],
    audit_rate: float,
    seed: int,
) -> Tuple[List[Round2OutputRecord], int]:
    rng = random.Random(seed)
    auto_accepted_indices = [
        i for i, r in enumerate(records)
        if r.round2_label in ("sarcastic", "non_sarcastic") and not r.need_review
    ]
    k = max(0, round(len(auto_accepted_indices) * audit_rate))
    sampled_indices = set(rng.sample(auto_accepted_indices, k) if k > 0 else [])

    updated = []
    for i, rec in enumerate(records):
        if i in sampled_indices:
            rec = rec.model_copy(update={"need_review": True, "route_reason": "audit_sampled"})
        updated.append(rec)
    return updated, k


def route_all(
    records_input: list,
    llm_results: List[LLMJudgeRecord],
    cfg: RouterConfig,
) -> List[Round2OutputRecord]:
    llm_by_id = {r.id: r for r in llm_results}
    routed: List[Round2OutputRecord] = []

    for inp in records_input:
        llm_rec = llm_by_id.get(inp.id)
        if llm_rec is None:
            continue

        override = None
        if llm_rec.image_missing:
            override = "missing_image"
        elif llm_rec.parse_error:
            override = "invalid_json"

        routed.append(route_single(llm_rec, cfg, inp.text, inp.image_path, override))
    return routed
