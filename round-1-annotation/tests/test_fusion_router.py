from __future__ import annotations

from src.fusion_router import RouterConfig, apply_audit_sampling, route_single
from src.schemas import LLMJudgeRecord


def _llm(
    id: int,
    label,
    needs_human_check: int = 0,
    parse_error: bool = False,
    image_missing: bool = False,
) -> LLMJudgeRecord:
    return LLMJudgeRecord(
        id=id,
        label_llm1=label,
        needs_human_check=needs_human_check if label not in ("INVALID", -1) else None,
        parse_error=parse_error,
        image_missing=image_missing,
    )


DEFAULT_CFG = RouterConfig(random_audit_rate=0.10, seed=42)


def test_sarcastic_confident_auto_accept():
    out = route_single(_llm(1, 1, needs_human_check=0), DEFAULT_CFG, "text", "img.jpg")
    assert out.round1_label == "sarcastic"
    assert out.need_review is False
    assert out.route_reason == "high_conf"


def test_non_sarcastic_confident_auto_accept():
    out = route_single(_llm(2, 0, needs_human_check=0), DEFAULT_CFG, "text", "img.jpg")
    assert out.round1_label == "non_sarcastic"
    assert out.need_review is False
    assert out.route_reason == "high_conf"


def test_sarcastic_uncertain_needs_review():
    out = route_single(_llm(3, 1, needs_human_check=1), DEFAULT_CFG, "text", "img.jpg")
    assert out.round1_label == "sarcastic"
    assert out.need_review is True
    assert out.route_reason == "low_conf"


def test_invalid_label():
    out = route_single(_llm(5, "INVALID"), DEFAULT_CFG, "text", "img.jpg")
    assert out.round1_label == "invalid"
    assert out.need_review is True
    assert out.route_reason == "uncertain"


def test_failed_after_retries_label_minus_one():
    out = route_single(_llm(6, -1, parse_error=True), DEFAULT_CFG, "text", "img.jpg", route_reason_override="invalid_json")
    assert out.round1_label == "invalid"
    assert out.need_review is True
    assert out.route_reason == "invalid_json"
    assert out.label_llm1 == -1


def test_missing_image_override():
    out = route_single(_llm(7, 1, needs_human_check=0, image_missing=True), DEFAULT_CFG, "text", "img.jpg", route_reason_override="missing_image")
    assert out.need_review is True
    assert out.route_reason == "missing_image"


def test_audit_sampling_reroute():
    records = [route_single(_llm(i, 1, needs_human_check=0), DEFAULT_CFG, "text", "img.jpg") for i in range(20)]
    updated, audit_k = apply_audit_sampling(records, audit_rate=0.20, seed=42)
    sampled = [r for r in updated if r.route_reason == "audit_sampled"]
    assert len(sampled) == audit_k == 4
