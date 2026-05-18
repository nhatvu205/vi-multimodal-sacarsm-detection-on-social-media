from src.fusion_router import RouterConfig, route_single
from src.schemas import LLMJudgeRecord

CFG = RouterConfig(random_audit_rate=0.08, seed=42)


def test_high_conf_route():
    rec = LLMJudgeRecord(id=1, label_llm2=1, needs_human_check=0)
    out = route_single(rec, CFG, "text", "img.jpg")
    assert out.round2_label == "sarcastic"
    assert out.need_review is False
    assert out.route_reason == "high_conf"


def test_invalid_route():
    rec = LLMJudgeRecord(id=2, label_llm2=-1, parse_error=True)
    out = route_single(rec, CFG, "text", "img.jpg", route_reason_override="invalid_json")
    assert out.round2_label == "invalid"
    assert out.need_review is True
    assert out.route_reason == "invalid_json"
