from __future__ import annotations

import asyncio
from pathlib import Path

from src.pipeline_round2 import resolve_model_config, run_llm_with_checkpoint, select_records_for_run
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

    asyncio.run(
        __import__("src.pipeline_round2", fromlist=["run_pipeline_async"]).run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="data/round-2/50-samples.json",
            model_tag="gemma",
        )
    )

    assert captured["ocr_path"] is None


def test_run_llm_with_checkpoint_staggers_first_requests_across_keys(monkeypatch):
    records = [_rec(1), _rec(2)]
    sleep_calls = []

    monkeypatch.setattr(
        "src.pipeline_round2.load_async_api_client",
        lambda provider, api_key=None: {
            "provider": provider,
            "lock": asyncio.Lock(),
            "api_keys": ["k1", "k2"],
            "active_key_index": 0,
            "exhausted_key_indices": set(),
        },
    )

    async def fake_close_async_api_client():
        return None

    monkeypatch.setattr("src.pipeline_round2.close_async_api_client", fake_close_async_api_client)
    monkeypatch.setattr("src.pipeline_round2._save_checkpoint_results", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.pipeline_round2._sync_parse_error_artifact", lambda *args, **kwargs: None)

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        return None

    monkeypatch.setattr("src.pipeline_round2.asyncio.sleep", fake_sleep)

    async def fake_judge_single_async(
        async_client,
        model_config,
        record,
        temperature,
        max_image_pixels=300_000,
        max_output_tokens=256,
        max_retries=3,
        retry_delay_seconds=5,
        max_retry_delay_seconds=20,
        key_index=None,
        allow_key_rotation=True,
    ):
        return LLMJudgeRecord(id=record.id, label_llm2=1, final_label=1, T=1, I=0, MM=1)

    monkeypatch.setattr("src.pipeline_round2.judge_single_async", fake_judge_single_async)

    results = asyncio.run(
        run_llm_with_checkpoint(
            records,
            {"provider": "gemini_api", "tag": "gemma", "model_name": "gemma"},
            temperature=0.1,
            output_dir=Path("."),
            router_cfg=type("Cfg", (), {"random_audit_rate": 0.1, "seed": 42})(),
            load_checkpoint_file=False,
            parallel_keys=True,
            per_key_concurrency=1,
            stagger_key_starts=True,
            key_start_stagger_seconds=1.0,
            checkpoint_every=10,
        )
    )

    assert [result.label_llm2 for result in results] == [1, 1]
    assert sleep_calls == [1.0]
