from __future__ import annotations

import pytest

from src.fusion_router import RouterConfig, apply_audit_sampling, route_single
from src.schemas import LLMJudgeRecord, Round1OutputRecord


def _llm(
    id: int,
    label,
    difficulty: str = "Easy",
    parse_error: bool = False,
    image_missing: bool = False,
) -> LLMJudgeRecord:
    return LLMJudgeRecord(
        id=id,
        label_llm1=label,
        difficulty=difficulty if label != "INVALID" else None,
        parse_error=parse_error,
        image_missing=image_missing,
    )


DEFAULT_CFG = RouterConfig(
    random_audit_rate=0.10,
    seed=42,
)


# ---------------------------------------------------------------------------
# Test 1: label=1 + Easy => auto-label sarcastic / high_conf
# ---------------------------------------------------------------------------
def test_sarcastic_easy_auto_accept():
    llm = _llm(1, 1, difficulty="Easy")
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "sarcastic"
    assert out.route_reason == "high_conf"
    assert out.label_llm1 == 1
    assert out.difficulty == "Easy"


# ---------------------------------------------------------------------------
# Test 2: label=0 + Easy => auto-label non_sarcastic / high_conf
# ---------------------------------------------------------------------------
def test_non_sarcastic_easy_auto_accept():
    llm = _llm(2, 0, difficulty="Easy")
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "non_sarcastic"
    assert out.route_reason == "high_conf"
    assert out.label_llm1 == 0


# ---------------------------------------------------------------------------
# Test 3: label=1 + Hard => human review / low_conf
# ---------------------------------------------------------------------------
def test_sarcastic_hard_human_queue():
    llm = _llm(3, 1, difficulty="Hard")
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "low_conf"


# ---------------------------------------------------------------------------
# Test 4: label=0 + Hard => human review / low_conf
# ---------------------------------------------------------------------------
def test_non_sarcastic_hard_human_queue():
    llm = _llm(4, 0, difficulty="Hard")
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "low_conf"


# ---------------------------------------------------------------------------
# Test 5: label="INVALID" => human queue, reason uncertain
# ---------------------------------------------------------------------------
def test_invalid_label_human_queue():
    llm = _llm(5, "INVALID")
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "uncertain"
    assert out.label_llm1 == "INVALID"
    assert out.difficulty is None


# ---------------------------------------------------------------------------
# Test 6: missing image override => human queue, reason missing_image
# ---------------------------------------------------------------------------
def test_missing_image_override():
    llm = _llm(6, 1, difficulty="Easy", image_missing=True)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg", route_reason_override="missing_image")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "missing_image"


# ---------------------------------------------------------------------------
# Test 7: parse error override => human queue, reason invalid_json
# ---------------------------------------------------------------------------
def test_parse_error_override():
    llm = _llm(7, "INVALID", parse_error=True)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg", route_reason_override="invalid_json")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "invalid_json"


# ---------------------------------------------------------------------------
# Test 8: audit sampling reroutes auto-accepted records
# ---------------------------------------------------------------------------
def test_audit_sampling_reroute():
    records = []
    for i in range(20):
        llm = _llm(i, 1, difficulty="Easy")
        out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")
        records.append(out)

    assert all(r.round1_label == "sarcastic" for r in records)

    updated, audit_k = apply_audit_sampling(records, audit_rate=0.20, seed=42)

    sampled = [r for r in updated if r.route_reason == "audit_sampled"]
    assert len(sampled) == audit_k
    assert audit_k == 4

    for r in sampled:
        assert r.round1_label == "needs_human_review"
        assert r.route_reason == "audit_sampled"


# ---------------------------------------------------------------------------
# Test 9: difficulty=None falls through to low_conf
# ---------------------------------------------------------------------------
def test_no_difficulty_low_conf():
    llm = LLMJudgeRecord(id=9, label_llm1=1, difficulty=None)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "low_conf"


# ---------------------------------------------------------------------------
# Test 10: mixed batch with audit sampling respects seed
# ---------------------------------------------------------------------------
def test_mixed_batch_audit_seed_reproducible():
    records = []
    for i in range(10):
        llm = _llm(i, 1 if i % 2 == 0 else 0, difficulty="Easy")
        out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")
        records.append(out)

    updated_a, k_a = apply_audit_sampling(records, audit_rate=0.20, seed=42)
    updated_b, k_b = apply_audit_sampling(records, audit_rate=0.20, seed=42)

    assert k_a == k_b
    sampled_ids_a = {r.id for r in updated_a if r.route_reason == "audit_sampled"}
    sampled_ids_b = {r.id for r in updated_b if r.route_reason == "audit_sampled"}
    assert sampled_ids_a == sampled_ids_b
