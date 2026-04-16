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
    Routing thresholds for the Round-1 pipeline.

    With the new prompt schema, confidence is expressed via the model's
    Difficulty field ("Easy" = high confidence, "Hard" = lower confidence)
    rather than a numeric probability score.

    random_audit_rate : fraction of auto-accepted records re-routed to human
                        queue for quality-control sampling.
    seed              : RNG seed for reproducible audit sampling.
    """
    random_audit_rate: float = 0.10
    seed: int = 42


def route_single(
    llm_rec: LLMJudgeRecord,
    cfg: RouterConfig,
    text: str,
    image_path: str,
    route_reason_override: Optional[str] = None,
) -> Round1OutputRecord:
    """
    Compute routing decision for a single record.

    round1_label is a 3-class field directly reflecting the LLM's verdict:
      - "sarcastic"     : Label_LLM1 == 1
      - "non_sarcastic" : Label_LLM1 == 0
      - "invalid"       : Label_LLM1 == "INVALID" or unrecoverable parse error

    need_review is a separate boolean flag indicating whether a human should
    verify the record. Routing rules (in order):
      1. missing_image  -> need_review=True  / route_reason=missing_image
      2. invalid_json   -> need_review=True  / route_reason=invalid_json   (round1_label forced to "invalid")
      3. label==INVALID -> need_review=True  / route_reason=uncertain
      4. Difficulty==Easy -> need_review=False / route_reason=high_conf
      5. Difficulty==Hard or None -> need_review=True / route_reason=low_conf
    """
    label = llm_rec.label_llm1
    difficulty = llm_rec.difficulty

    # --- Determine round1_label from LLM output ---
    if label == 1:
        round1_label = "sarcastic"
    elif label == 0:
        round1_label = "non_sarcastic"
    else:
        round1_label = "invalid"

    # --- Determine need_review and route_reason ---
    if route_reason_override == "missing_image":
        need_review = True
        route_reason: str = "missing_image"

    elif route_reason_override == "invalid_json":
        need_review = True
        route_reason = "invalid_json"
        round1_label = "invalid"

    elif label == "INVALID":
        need_review = True
        route_reason = "uncertain"

    elif difficulty == "Easy":
        need_review = False
        route_reason = "high_conf"

    else:
        need_review = True
        route_reason = "low_conf"

    return Round1OutputRecord(
        id=llm_rec.id,
        text=text,
        image_path=image_path,
        label_llm1=label,
        text_only=llm_rec.text_only,
        imageset_only=llm_rec.imageset_only,
        key_images=llm_rec.key_images,
        difficulty=difficulty,
        notes=llm_rec.notes,
        reasoning=llm_rec.reasoning,
        raw_llm_output=llm_rec.raw_llm_output,
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
                    "need_review": True,
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

    for inp in records_input:
        llm_rec = llm_by_id.get(inp.id)

        if llm_rec is None:
            logger.warning("No LLM result for id=%d, skipping", inp.id)
            continue

        override = None
        if llm_rec.image_missing:
            override = "missing_image"
        elif llm_rec.parse_error:
            override = "invalid_json"

        out = route_single(llm_rec, cfg, inp.text, inp.image_path, override)
        routed.append(out)

    return routed
