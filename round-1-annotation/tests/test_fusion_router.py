from __future__ import annotations

import pytest

from src.fusion_router import RouterConfig, apply_audit_sampling, route_single
from src.schemas import LLMJudgeRecord, Round1OutputRecord


def _llm(id: int, label: str, prob: float, conf: float = 0.9) -> LLMJudgeRecord:
    return LLMJudgeRecord(
        id=id,
        llm_pred_label=label,
        llm_prob_sarcastic=prob,
        llm_confidence=conf,
        llm_rationale_short="test rationale",
    )


DEFAULT_CFG = RouterConfig(
    sarcastic_high=0.85,
    nonsarcastic_low=0.15,
    conf_threshold=0.70,
    random_audit_rate=0.10,
    seed=42,
)


# ---------------------------------------------------------------------------
# Test 1: high sarcastic prob + sufficient confidence => auto-label sarcastic
# ---------------------------------------------------------------------------
def test_high_sarcastic_auto_accept():
    llm = _llm(1, "sarcastic", prob=0.92, conf=0.88)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "sarcastic"
    assert out.route_reason == "high_conf"
    assert out.llm_prob_sarcastic >= DEFAULT_CFG.sarcastic_high


# ---------------------------------------------------------------------------
# Test 2: high non-sarcastic prob + sufficient confidence => auto-label non_sarcastic
# ---------------------------------------------------------------------------
def test_high_non_sarcastic_auto_accept():
    llm = _llm(2, "non_sarcastic", prob=0.06, conf=0.85)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "non_sarcastic"
    assert out.route_reason == "high_conf"
    assert out.llm_prob_sarcastic <= DEFAULT_CFG.nonsarcastic_low


# ---------------------------------------------------------------------------
# Test 3: middle probability => low_conf => human review
# ---------------------------------------------------------------------------
def test_middle_prob_low_conf_human_queue():
    llm = _llm(3, "sarcastic", prob=0.55, conf=0.80)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "low_conf"
    assert DEFAULT_CFG.nonsarcastic_low < out.llm_prob_sarcastic < DEFAULT_CFG.sarcastic_high


# ---------------------------------------------------------------------------
# Test 4: LLM returns uncertain => human queue, reason uncertain
# ---------------------------------------------------------------------------
def test_llm_uncertain_human_queue():
    llm = _llm(4, "uncertain", prob=0.50, conf=0.40)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "uncertain"


# ---------------------------------------------------------------------------
# Test 5: high prob but confidence below threshold => human queue, low_conf
# ---------------------------------------------------------------------------
def test_low_confidence_blocks_auto_accept():
    # prob would pass sarcastic_high but confidence too low
    llm = _llm(5, "sarcastic", prob=0.90, conf=0.60)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "low_conf"
    assert out.llm_confidence < DEFAULT_CFG.conf_threshold


# ---------------------------------------------------------------------------
# Test 6: missing image override => human queue, reason missing_image
# ---------------------------------------------------------------------------
def test_missing_image_override():
    llm = _llm(6, "sarcastic", prob=0.92, conf=0.91)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg", route_reason_override="missing_image")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "missing_image"


# ---------------------------------------------------------------------------
# Test 7: audit sampling reroutes auto-accepted records
# ---------------------------------------------------------------------------
def test_audit_sampling_reroute():
    records = []
    for i in range(20):
        llm = _llm(i, "sarcastic", prob=0.92, conf=0.90)
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
# Test 8: boundary values exactly at thresholds
# ---------------------------------------------------------------------------
def test_boundary_at_sarcastic_high():
    # Exactly at sarcastic_high=0.85, confidence above threshold
    llm = _llm(8, "sarcastic", prob=0.85, conf=0.90)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.llm_prob_sarcastic == pytest.approx(0.85, abs=1e-9)
    assert out.round1_label == "sarcastic"
    assert out.route_reason == "high_conf"


def test_boundary_at_nonsarcastic_low():
    # Exactly at nonsarcastic_low=0.15, confidence above threshold
    llm = _llm(9, "non_sarcastic", prob=0.15, conf=0.90)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.llm_prob_sarcastic == pytest.approx(0.15, abs=1e-9)
    assert out.round1_label == "non_sarcastic"
    assert out.route_reason == "high_conf"


def test_boundary_conf_threshold_exact():
    # Confidence exactly at conf_threshold=0.70 should pass
    llm = _llm(10, "sarcastic", prob=0.90, conf=0.70)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "sarcastic"
    assert out.route_reason == "high_conf"


def test_boundary_conf_below_threshold():
    # Confidence just below conf_threshold=0.70 should fail
    llm = _llm(11, "sarcastic", prob=0.90, conf=0.699)
    out = route_single(llm, DEFAULT_CFG, "text", "img.jpg")

    assert out.round1_label == "needs_human_review"
    assert out.route_reason == "low_conf"
