from src.pipeline_round2 import resolve_model_config, select_records_for_run
from src.schemas import InputRecord, LLMJudgeRecord


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
                "reasoning": {"enabled": True, "exclude": True},
            },
        },
    }

    model_cfg = resolve_model_config(cfg, "nemotron")

    assert model_cfg["tag"] == "nemotron"
    assert model_cfg["provider"] == "openrouter"
    assert model_cfg["reasoning"] == {"enabled": True, "exclude": True}


def test_run_pipeline_async_uses_test_input_data_in_test_mode(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "src.pipeline_round2.load_config",
        lambda path: {
            "default_model": "gemma",
            "models": {
                "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it", "concurrency": 1},
            },
            "seed": 42,
            "random_audit_rate": 0.1,
        },
    )

    def fake_load_input_records(path, ocr_path=None):
        captured["input_path"] = path
        return [InputRecord(id=1, text="t", image_path="img.jpg", label_round_1=1)]

    monkeypatch.setattr("src.pipeline_round2.load_input_records", fake_load_input_records)
    monkeypatch.setattr("src.pipeline_round2.load_checkpoint", lambda output_dir: {})
    monkeypatch.setattr("src.pipeline_round2.get_gemini_key_count", lambda api_key=None: 1)

    async def fake_run_llm_with_checkpoint(*args, **kwargs):
        return [LLMJudgeRecord(id=1, label_llm2=1, final_label=1, T=1, I=0, MM=1)]

    monkeypatch.setattr("src.pipeline_round2.run_llm_with_checkpoint", fake_run_llm_with_checkpoint)
    monkeypatch.setattr("src.pipeline_round2.write_results", lambda out_dir, routed: None)

    import asyncio
    asyncio.run(
        __import__("src.pipeline_round2", fromlist=["run_pipeline_async"]).run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="data/round-2/50-samples.json",
            test_input_data="data/round-2/50-samples-final-label.json",
            test_mode=True,
            test_size=50,
            model_tag="gemma",
        )
    )

    assert captured["input_path"] == "data/round-2/50-samples-final-label.json"


def test_run_pipeline_async_does_not_use_config_ocr_path_by_default(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "src.pipeline_round2.load_config",
        lambda path: {
            "default_model": "gemma",
            "models": {
                "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it", "concurrency": 1},
            },
            "seed": 42,
            "random_audit_rate": 0.1,
            "ocr_path": "data/raw-data/ocr_images.json",
        },
    )

    def fake_load_input_records(path, ocr_path=None):
        captured["ocr_path"] = ocr_path
        return [InputRecord(id=1, text="t", image_path="img.jpg", ocr_text="embedded", label_round_1=1)]

    monkeypatch.setattr("src.pipeline_round2.load_input_records", fake_load_input_records)
    monkeypatch.setattr("src.pipeline_round2.load_checkpoint", lambda output_dir: {})
    monkeypatch.setattr("src.pipeline_round2.get_gemini_key_count", lambda api_key=None: 1)

    async def fake_run_llm_with_checkpoint(*args, **kwargs):
        return [LLMJudgeRecord(id=1, label_llm2=1, final_label=1, T=1, I=0, MM=1)]

    monkeypatch.setattr("src.pipeline_round2.run_llm_with_checkpoint", fake_run_llm_with_checkpoint)
    monkeypatch.setattr("src.pipeline_round2.write_results", lambda out_dir, routed: None)

    import asyncio
    asyncio.run(
        __import__("src.pipeline_round2", fromlist=["run_pipeline_async"]).run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="data/round-2/50-samples.json",
            model_tag="gemma",
        )
    )

    assert captured["ocr_path"] is None
