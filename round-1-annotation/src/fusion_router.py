from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from tqdm import tqdm

from .schemas import LLMJudgeRecord, Round1OutputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouterConfig:
    sarcastic_high: float = 0.85
    nonsarcastic_low: float = 0.15
    conf_threshold: float = 0.70
    random_audit_rate: float = 0.10
    seed: int = 42


def route_single(
    llm_rec: LLMJudgeRecord,
    cfg: RouterConfig,
    text: str,
    image_path: str,
    route_reason_override: Optional[str] = None,
) -> Round1OutputRecord:
    """Compute routing decision for a single record based on LLM output alone."""
    p_llm = llm_rec.llm_prob_sarcastic

    if route_reason_override == "missing_image":
        round1_label = "needs_human_review"
        route_reason = "missing_image"
    elif route_reason_override == "invalid_json":
        round1_label = "needs_human_review"
        route_reason = "invalid_json"
    elif llm_rec.llm_pred_label == "uncertain":
        round1_label = "needs_human_review"
        route_reason = "uncertain"
    elif llm_rec.llm_confidence < cfg.conf_threshold:
        round1_label = "needs_human_review"
        route_reason = "low_conf"
    elif p_llm >= cfg.sarcastic_high:
        round1_label = "sarcastic"
        route_reason = "high_conf"
    elif p_llm <= cfg.nonsarcastic_low:
        round1_label = "non_sarcastic"
        route_reason = "high_conf"
    else:
        round1_label = "needs_human_review"
        route_reason = "low_conf"

    return Round1OutputRecord(
        id=llm_rec.id,
        text=text,
        image_path=image_path,
        llm_pred_label=llm_rec.llm_pred_label,
        llm_prob_sarcastic=round(p_llm, 6),
        llm_confidence=round(llm_rec.llm_confidence, 6),
        round1_label=round1_label,
        route_reason=route_reason,
        llm_rationale_short=llm_rec.llm_rationale_short,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def apply_audit_sampling(
    records: List[Round1OutputRecord],
    audit_rate: float,
    seed: int,
) -> Tuple[List[Round1OutputRecord], int]:
    """
    Randomly sample from auto-accepted records and reroute to human queue.
    Returns updated record list and count of audit-sampled records.
    """
    rng = random.Random(seed)
    auto_accepted_indices = [
        i for i, r in enumerate(records)
        if r.round1_label in ("sarcastic", "non_sarcastic")
    ]

    k = max(0, round(len(auto_accepted_indices) * audit_rate))
    sampled_indices = set(rng.sample(auto_accepted_indices, k) if k > 0 else [])

    updated = []
    for i, rec in enumerate(records):
        if i in sampled_indices:
            rec = rec.model_copy(
                update={
                    "round1_label": "needs_human_review",
                    "route_reason": "audit_sampled",
                }
            )
        updated.append(rec)

    logger.info(
        "Audit sampling: %d auto-accepted candidates, %d sampled (rate=%.2f)",
        len(auto_accepted_indices), k, audit_rate,
    )
    return updated, k


def route_all(
    records_input: list,
    llm_results: List[LLMJudgeRecord],
    cfg: RouterConfig,
) -> List[Round1OutputRecord]:
    """Route all records using LLM results only."""
    llm_by_id = {r.id: r for r in llm_results}
    routed: List[Round1OutputRecord] = []

    for inp in tqdm(records_input, desc="Routing records", unit="rec"):
        llm_rec = llm_by_id.get(inp.id)

        if llm_rec is None:
            logger.warning("No LLM result for id=%d, skipping", inp.id)
            continue

        override = None
        if llm_rec.llm_pred_label == "uncertain" and llm_rec.llm_confidence == 0.0:
            rationale = llm_rec.llm_rationale_short.lower()
            if "json" in rationale or "parse" in rationale:
                override = "invalid_json"
            elif "image" in rationale or "missing" in rationale or "unreadable" in rationale:
                override = "missing_image"

        out = route_single(llm_rec, cfg, inp.text, inp.image_path, override)
        routed.append(out)

    return routed
