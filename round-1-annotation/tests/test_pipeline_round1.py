from src.pipeline_round1 import resolve_model_config, select_records_for_run
from src.schemas import InputRecord


def _rec(i: int) -> InputRecord:
    return InputRecord(id=i, text=f"text {i}", image_path=f"img_{i}.jpg")


def test_select_records_for_run_takes_first_records_in_order():
    records = [_rec(i) for i in range(10)]
    sampled = select_records_for_run(records, test_mode=True, test_size=5, seed=42)
    assert [r.id for r in sampled] == [0, 1, 2, 3, 4]


def test_resolve_model_config_returns_nemotron_settings():
    cfg = {
        "default_model": "gemma",
        "models": {
            "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it"},
            "nemotron": {
                "provider": "openrouter",
                "model_name": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "concurrency": 1,
            },
        },
    }

    model_cfg = resolve_model_config(cfg, "nemotron")

    assert model_cfg["tag"] == "nemotron"
    assert model_cfg["provider"] == "openrouter"
    assert model_cfg["reasoning"] == {}
    assert model_cfg["concurrency"] == 1
