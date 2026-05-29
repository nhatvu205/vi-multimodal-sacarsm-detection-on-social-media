from __future__ import annotations

import asyncio

from src.llm_judge import KeyExhaustedError, _validate, judge_single_async
from src.schemas import InputRecord, LLMJudgeRecord


def test_validate_accepts_final_label_and_nested_labels():
    rec = _validate(
        {
            "labels": {"T": 1, "I": 0, "MM": 0},
            "final_label": 0,
            "reasoning": {
                "text_only": "x",
                "image_only": "y",
                "multimodal": "z",
                "verdict": "v",
            },
            "has_emoji": 1,
        }
    )

    assert rec.label_llm2 == 0
    assert rec.final_label == 0
    assert rec.T == 1
    assert rec.I == 0
    assert rec.MM == 0


def test_validate_derives_final_label_when_missing():
    rec = _validate(
        {
            "labels": {"T": 0, "I": 1, "MM": 0},
            "reasoning": {"verdict": "v"},
            "has_emoji": 0,
        }
    )

    assert rec.label_llm2 == 0
    assert rec.final_label == 0


def test_judge_single_async_retries_transient_resource_exhausted_without_exhausting_key(monkeypatch):
    attempts = {"count": 0}

    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None, reasoning_override=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: rate limit")
        return LLMJudgeRecord(id=record.id, label_llm2=1, final_label=1)

    monkeypatch.setattr("src.llm_judge._judge_once_async", fake_judge_once)

    result = asyncio.run(
        judge_single_async(
            async_client={
                "api_keys": ["key-a", "key-b"],
                "active_key_index": 0,
                "exhausted_key_indices": set(),
                "lock": asyncio.Lock(),
                "clients": [],
            },
            model_config={"provider": "gemini_api", "tag": "gemma"},
            record=InputRecord(id=1, text="caption", image_path="img.jpg"),
            temperature=0.1,
            max_retries=2,
            retry_delay_seconds=0,
            max_retry_delay_seconds=0,
        )
    )

    assert result.label_llm2 == 1
    assert attempts["count"] == 2


def test_judge_single_async_retries_openrouter_empty_content_without_reasoning(monkeypatch):
    seen_reasoning_overrides = []

    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None, reasoning_override=None):
        seen_reasoning_overrides.append(reasoning_override)
        if len(seen_reasoning_overrides) == 1:
            raise RuntimeError("OpenRouter response did not contain text content.")
        return LLMJudgeRecord(id=record.id, label_llm2=1, final_label=1)

    monkeypatch.setattr("src.llm_judge._judge_once_async", fake_judge_once)

    result = asyncio.run(
        judge_single_async(
            async_client={"lock": asyncio.Lock(), "api_keys": ["key-a"], "active_key_index": 0, "exhausted_key_indices": set(), "clients": []},
            model_config={"provider": "openrouter", "tag": "nemotron", "reasoning": {"effort": "medium", "exclude": True}},
            record=InputRecord(id=1, text="caption", image_path="img.jpg"),
            temperature=0.1,
            max_retries=2,
            retry_delay_seconds=0,
            max_retry_delay_seconds=0,
        )
    )

    assert result.label_llm2 == 1
    assert seen_reasoning_overrides == [None, {}]


def test_judge_single_async_rotates_key_only_on_hard_quota(monkeypatch):
    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None, reasoning_override=None):
        if async_client["active_key_index"] == 0:
            raise RuntimeError("insufficient credits")
        return LLMJudgeRecord(id=record.id, label_llm2=1, final_label=1)

    monkeypatch.setattr("src.llm_judge._judge_once_async", fake_judge_once)

    async_client = {
        "api_keys": ["key-a", "key-b"],
        "active_key_index": 0,
        "exhausted_key_indices": set(),
        "lock": asyncio.Lock(),
        "clients": [],
    }
    result = asyncio.run(
        judge_single_async(
            async_client=async_client,
            model_config={"provider": "gemini_api", "tag": "gemma"},
            record=InputRecord(id=1, text="caption", image_path="img.jpg"),
            temperature=0.1,
        )
    )

    assert result.label_llm2 == 1
    assert async_client["active_key_index"] == 1
    assert async_client["exhausted_key_indices"] == {0}


def test_judge_single_async_raises_key_exhausted_in_parallel_mode_only_for_hard_quota(monkeypatch):
    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None, reasoning_override=None):
        raise RuntimeError("insufficient credits")

    monkeypatch.setattr("src.llm_judge._judge_once_async", fake_judge_once)

    try:
        asyncio.run(
            judge_single_async(
                async_client={"api_keys": ["key-a"], "active_key_index": 0, "exhausted_key_indices": set(), "lock": asyncio.Lock(), "clients": []},
                model_config={"provider": "gemini_api", "tag": "gemma"},
                record=InputRecord(id=1, text="caption", image_path="img.jpg"),
                temperature=0.1,
                key_index=0,
                allow_key_rotation=False,
            )
        )
    except KeyExhaustedError as exc:
        assert exc.key_index == 0
    else:
        raise AssertionError("Expected KeyExhaustedError")


def test_judge_single_async_exhausts_parallel_key_after_two_resource_exhausted(monkeypatch):
    attempts = {"count": 0}

    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None, reasoning_override=None):
        attempts["count"] += 1
        raise RuntimeError("429 Resource has been exhausted (e.g. check quota).")

    monkeypatch.setattr("src.llm_judge._judge_once_async", fake_judge_once)

    try:
        asyncio.run(
            judge_single_async(
                async_client={"api_keys": ["key-a"], "active_key_index": 0, "exhausted_key_indices": set(), "lock": asyncio.Lock(), "clients": []},
                model_config={"provider": "gemini_api", "tag": "gemma"},
                record=InputRecord(id=1, text="caption", image_path="img.jpg"),
                temperature=0.1,
                max_retries=5,
                retry_delay_seconds=0,
                max_retry_delay_seconds=0,
                key_index=0,
                allow_key_rotation=False,
            )
        )
    except KeyExhaustedError as exc:
        assert exc.key_index == 0
        assert attempts["count"] == 2
    else:
        raise AssertionError("Expected KeyExhaustedError")
