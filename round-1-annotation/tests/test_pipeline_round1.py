import asyncio

from src.pipeline_round1 import resolve_model_config, run_pipeline_async, select_failed_records_for_rerun, select_records_for_run
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
                "concurrency": 1,
            },
        },
    }

    model_cfg = resolve_model_config(cfg, "nemotron")

    assert model_cfg["tag"] == "nemotron"
    assert model_cfg["provider"] == "openrouter"
    assert model_cfg["reasoning"] == {}
    assert model_cfg["concurrency"] == 1


def test_select_failed_records_for_rerun_returns_only_label_minus_one():
    records = [_rec(1), _rec(2), _rec(3)]
    cached = {
        1: LLMJudgeRecord(id=1, label_llm1=-1),
        2: LLMJudgeRecord(id=2, label_llm1=0),
    }

    selected = select_failed_records_for_rerun(records, cached)

    assert [record.id for record in selected] == [1]


def test_run_pipeline_async_rerun_minus_one_updates_existing_results(monkeypatch):
    records = [_rec(1), _rec(2)]
    captured = {}
    cached = {
        1: LLMJudgeRecord(id=1, label_llm1=-1, notes="old fail", reasoning={}, parse_error=True, image_missing=False),
        2: LLMJudgeRecord(id=2, label_llm1=0, notes="", reasoning={}, parse_error=False, image_missing=False),
    }

    monkeypatch.setattr(
        "src.pipeline_round1.load_config",
        lambda path: {
            "default_model": "gemma",
            "models": {
                "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it", "concurrency": 1},
            },
            "seed": 42,
            "random_audit_rate": 0.1,
        },
    )
    monkeypatch.setattr("src.pipeline_round1.load_input_records", lambda path: records)
    monkeypatch.setattr("src.pipeline_round1.load_checkpoint", lambda output_dir: dict(cached))
    monkeypatch.setattr("src.pipeline_round1.get_gemini_key_count", lambda api_key=None: 1)

    async def fake_run_llm_with_checkpoint(
        run_records,
        model_config,
        temperature,
        out_dir,
        router_cfg,
        api_key=None,
        max_image_pixels=300_000,
        max_output_tokens=256,
        max_retries=3,
        retry_delay_seconds=5,
        max_retry_delay_seconds=20,
        concurrency=4,
        checkpoint_every=10,
        *,
        load_checkpoint_file=True,
        parallel_keys=False,
        per_key_concurrency=1,
        cached_results_by_id=None,
        checkpoint_records=None,
        checkpoint_base_results_by_id=None,
        started_at=None,
    ):
        assert [record.id for record in run_records] == [1]
        assert load_checkpoint_file is False
        assert cached_results_by_id is not None
        assert checkpoint_records == records
        assert checkpoint_base_results_by_id is not None
        return [LLMJudgeRecord(id=1, label_llm1=1, parse_error=False, image_missing=False)]

    monkeypatch.setattr("src.pipeline_round1.run_llm_with_checkpoint", fake_run_llm_with_checkpoint)

    def fake_write_results(out_dir, routed):
        captured["records"] = routed

    monkeypatch.setattr("src.pipeline_round1.write_results", fake_write_results)

    asyncio.run(
        run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="unused.json",
            model_tag="gemma",
            rerun_minus_one=True,
        )
    )

    assert [record.id for record in captured["records"]] == [1, 2]
    assert {record.id: record.label_llm1 for record in captured["records"]} == {1: 1, 2: 0}


def test_run_pipeline_async_filters_with_from_and_to(monkeypatch):
    records = [_rec(1), _rec(2), _rec(3), _rec(4)]
    captured = {}

    monkeypatch.setattr(
        "src.pipeline_round1.load_config",
        lambda path: {
            "default_model": "gemma",
            "models": {
                "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it", "concurrency": 1},
            },
            "seed": 42,
            "random_audit_rate": 0.1,
        },
    )
    monkeypatch.setattr("src.pipeline_round1.load_input_records", lambda path: records)
    monkeypatch.setattr("src.pipeline_round1.get_gemini_key_count", lambda api_key=None: 1)

    async def fake_run_llm_with_checkpoint(
        run_records,
        model_config,
        temperature,
        out_dir,
        router_cfg,
        api_key=None,
        max_image_pixels=300_000,
        max_output_tokens=256,
        max_retries=3,
        retry_delay_seconds=5,
        max_retry_delay_seconds=20,
        concurrency=4,
        checkpoint_every=10,
        *,
        load_checkpoint_file=True,
        parallel_keys=False,
        per_key_concurrency=1,
        cached_results_by_id=None,
        checkpoint_records=None,
        checkpoint_base_results_by_id=None,
        started_at=None,
    ):
        captured["run_ids"] = [record.id for record in run_records]
        return [LLMJudgeRecord(id=record.id, label_llm1=0) for record in run_records]

    monkeypatch.setattr("src.pipeline_round1.run_llm_with_checkpoint", fake_run_llm_with_checkpoint)
    monkeypatch.setattr("src.pipeline_round1.write_results", lambda out_dir, routed: None)

    asyncio.run(
        run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="unused.json",
            model_tag="gemma",
            min_record_id=2,
            max_record_id=3,
        )
    )

    assert captured["run_ids"] == [2, 3]


def test_run_pipeline_async_from_to_preserves_cached_results_outside_range(monkeypatch):
    records = [_rec(1), _rec(2), _rec(3), _rec(4)]
    captured = {}
    cached = {
        1: LLMJudgeRecord(id=1, label_llm1=0, notes="", reasoning={}, parse_error=False, image_missing=False),
        4: LLMJudgeRecord(id=4, label_llm1=1, notes="", reasoning={}, parse_error=False, image_missing=False),
    }

    monkeypatch.setattr(
        "src.pipeline_round1.load_config",
        lambda path: {
            "default_model": "gemma",
            "models": {
                "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it", "concurrency": 1},
            },
            "seed": 42,
            "random_audit_rate": 0.1,
        },
    )
    monkeypatch.setattr("src.pipeline_round1.load_input_records", lambda path: records)
    monkeypatch.setattr("src.pipeline_round1.load_checkpoint", lambda output_dir: dict(cached))
    monkeypatch.setattr("src.pipeline_round1.get_gemini_key_count", lambda api_key=None: 1)

    async def fake_run_llm_with_checkpoint(
        run_records,
        model_config,
        temperature,
        out_dir,
        router_cfg,
        api_key=None,
        max_image_pixels=300_000,
        max_output_tokens=256,
        max_retries=3,
        retry_delay_seconds=5,
        max_retry_delay_seconds=20,
        concurrency=4,
        checkpoint_every=10,
        *,
        load_checkpoint_file=True,
        parallel_keys=False,
        per_key_concurrency=1,
        cached_results_by_id=None,
        checkpoint_records=None,
        checkpoint_base_results_by_id=None,
        started_at=None,
    ):
        assert [record.id for record in run_records] == [2, 3]
        assert load_checkpoint_file is False
        assert cached_results_by_id == cached
        assert checkpoint_records == records
        assert checkpoint_base_results_by_id == cached
        return [
            LLMJudgeRecord(id=2, label_llm1=0, parse_error=False, image_missing=False),
            LLMJudgeRecord(id=3, label_llm1=1, parse_error=False, image_missing=False),
        ]

    monkeypatch.setattr("src.pipeline_round1.run_llm_with_checkpoint", fake_run_llm_with_checkpoint)

    def fake_write_results(out_dir, routed):
        captured["records"] = routed

    monkeypatch.setattr("src.pipeline_round1.write_results", fake_write_results)

    asyncio.run(
        run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="unused.json",
            model_tag="gemma",
            min_record_id=2,
            max_record_id=3,
        )
    )

    assert [record.id for record in captured["records"]] == [1, 2, 3, 4]


def test_run_pipeline_async_parallel_keys_passes_mode(monkeypatch):
    records = [_rec(1), _rec(2)]
    captured = {}

    monkeypatch.setattr(
        "src.pipeline_round1.load_config",
        lambda path: {
            "default_model": "gemma",
            "models": {
                "gemma": {"provider": "gemini_api", "model_name": "gemma-4-31b-it", "concurrency": 1},
            },
            "seed": 42,
            "random_audit_rate": 0.1,
        },
    )
    monkeypatch.setattr("src.pipeline_round1.load_input_records", lambda path: records)
    monkeypatch.setattr("src.pipeline_round1.load_checkpoint", lambda output_dir: {})
    monkeypatch.setattr("src.pipeline_round1.get_gemini_key_count", lambda api_key=None: 2)

    async def fake_run_llm_with_checkpoint(
        run_records,
        model_config,
        temperature,
        out_dir,
        router_cfg,
        api_key=None,
        max_image_pixels=300_000,
        max_output_tokens=256,
        max_retries=3,
        retry_delay_seconds=5,
        max_retry_delay_seconds=20,
        concurrency=4,
        checkpoint_every=10,
        *,
        load_checkpoint_file=True,
        parallel_keys=False,
        per_key_concurrency=1,
        cached_results_by_id=None,
        checkpoint_records=None,
        checkpoint_base_results_by_id=None,
        started_at=None,
    ):
        captured["parallel_keys"] = parallel_keys
        captured["per_key_concurrency"] = per_key_concurrency
        return [LLMJudgeRecord(id=record.id, label_llm1=0) for record in run_records]

    monkeypatch.setattr("src.pipeline_round1.run_llm_with_checkpoint", fake_run_llm_with_checkpoint)
    monkeypatch.setattr("src.pipeline_round1.write_results", lambda out_dir, routed: None)

    asyncio.run(
        run_pipeline_async(
            config_path="unused.yaml",
            output_dir=".",
            input_data="unused.json",
            model_tag="gemma",
            parallel_keys=True,
            per_key_concurrency=3,
        )
    )

    assert captured == {"parallel_keys": True, "per_key_concurrency": 3}
