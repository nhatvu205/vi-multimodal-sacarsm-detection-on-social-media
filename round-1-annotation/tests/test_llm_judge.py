from __future__ import annotations

import asyncio
from types import SimpleNamespace

import src.llm_judge as llm_judge
from src.llm_judge import (
    KeyExhaustedError,
    _build_openrouter_payload,
    _build_openrouter_repair_messages,
    _call_gemini_api_async,
    _get_provider_key_state_for_index,
    _judge_once_async,
    get_gemini_key_count,
    judge_single_async,
    load_async_api_client,
)
from src.schemas import LLMJudgeRecord
from src.schemas import InputRecord


def test_build_openrouter_payload_adds_json_response_format():
    payload = _build_openrouter_payload(
        {"model_name": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"},
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        temperature=0.1,
        max_output_tokens=256,
    )

    assert payload["response_format"] == {"type": "json_object"}
    assert "reasoning" not in payload


def test_build_openrouter_repair_messages_keeps_full_input_context():
    messages = _build_openrouter_repair_messages("caption", [], "not json", "ocr text")

    assert messages[0]["role"] == "user"
    assert "caption" in messages[0]["content"][0]["text"]
    assert "ocr text" in messages[0]["content"][0]["text"]
    assert messages[1] == {"role": "assistant", "content": "not json"}
    assert messages[2]["role"] == "user"
    assert "Phản hồi trước của bạn" in messages[2]["content"][0]["text"]
    assert "caption" in messages[2]["content"][0]["text"]
    assert "not json" in messages[2]["content"][0]["text"]


def test_judge_once_async_openrouter_repair_reuses_full_input(monkeypatch):
    calls = []

    async def fake_call(async_client, model_config, messages, temperature, max_output_tokens, key_index=None):
        calls.append(messages)
        if len(calls) == 1:
            return "not json", {"choices": [{"message": {"content": "not json"}}]}
        return '{"llm_label": 1, "reasoning": {}, "has_emoji": 0, "needs_human_check": 0}', {"choices": [{"message": {"content": '{"llm_label": 1}'}}]}

    monkeypatch.setattr("src.llm_judge._call_openrouter_api_async", fake_call)
    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    record = InputRecord(id=1, text="caption", image_path="img.jpg", ocr_text="ocr text")
    result = asyncio.run(
        _judge_once_async(
            async_client={},
            model_config={"provider": "openrouter", "model_name": "nemotron"},
            record=record,
            temperature=0.1,
            max_image_pixels=300_000,
            max_output_tokens=256,
        )
    )

    assert result.label_llm1 == 1
    assert len(calls) == 2
    assert calls[1][0] == calls[0][0]
    assert calls[1][1] == {"role": "assistant", "content": "not json"}
    assert "caption" in calls[1][2]["content"][0]["text"]
    assert "ocr text" in calls[1][2]["content"][0]["text"]


def test_judge_single_async_openrouter_returns_raw_response_on_parse_error(monkeypatch):
    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_call(async_client, model_config, messages, temperature, max_output_tokens, key_index=None):
        return "not json", {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr("src.llm_judge._call_openrouter_api_async", fake_call)

    result = asyncio.run(
        judge_single_async(
            async_client={"lock": asyncio.Lock()},
            model_config={"provider": "openrouter", "tag": "nemotron", "model_name": "nemotron"},
            record=InputRecord(id=9, text="caption", image_path="img.jpg"),
            temperature=0.1,
            max_retries=1,
        )
    )

    assert result.label_llm1 == -1
    assert result.parse_error is True
    assert result.raw_response is not None
    assert result.raw_response["initial_response"]["choices"][0]["message"]["content"] == "not json"


def test_load_async_api_client_supports_multiple_gemini_keys(monkeypatch):
    created_keys = []

    class FakeAio:
        def __init__(self):
            self.models = SimpleNamespace(generate_content=None)

        async def aclose(self):
            return None

    class FakeClient:
        def __init__(self, api_key):
            created_keys.append(api_key)
            self.aio = FakeAio()

        def close(self):
            return None

    monkeypatch.setattr(llm_judge, "_CLIENT", None)
    monkeypatch.setattr(llm_judge, "_ASYNC_CLIENT", None)
    monkeypatch.setattr(llm_judge, "_ACTIVE_PROVIDER", None)
    monkeypatch.setattr(llm_judge, "_import_google_genai", lambda: (SimpleNamespace(Client=FakeClient), SimpleNamespace()))

    async_client = load_async_api_client("gemini_api", "key-a,key-b")

    assert get_gemini_key_count("key-a,key-b") == 2
    assert async_client["api_keys"] == ["key-a", "key-b"]
    assert async_client["active_key_index"] == 0
    assert created_keys == ["key-a", "key-b"]

    asyncio.run(llm_judge.close_async_api_client())


def test_judge_single_async_rotates_gemini_key_on_quota(monkeypatch):
    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None):
        if async_client["active_key_index"] == 0:
            raise RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")
        return LLMJudgeRecord(id=record.id, label_llm1=1)

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
        )
    )

    assert result.label_llm1 == 1


def test_judge_single_async_raises_key_exhausted_in_parallel_mode(monkeypatch):
    monkeypatch.setattr("src.llm_judge._load_images", lambda record, max_pixels: ([], False))

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None):
        raise RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")

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


def test_get_provider_key_state_for_index_returns_requested_key():
    api_key, key_index, total = asyncio.run(
        _get_provider_key_state_for_index(
            {
                "api_keys": ["key-a", "key-b"],
                "active_key_index": 0,
                "exhausted_key_indices": set(),
                "lock": asyncio.Lock(),
            },
            1,
        )
    )

    assert api_key == "key-b"
    assert key_index == 1
    assert total == 2


def test_call_gemini_api_async_uses_explicit_key_index(monkeypatch):
    used_clients = []

    class FakeModels:
        def __init__(self, client_name):
            self.client_name = client_name

        async def generate_content(self, **kwargs):
            used_clients.append(self.client_name)
            return SimpleNamespace(text='{"llm_label": 0, "reasoning": {}, "has_emoji": 0, "needs_human_check": 0}', parsed=None)

    class FakeAio:
        def __init__(self, client_name):
            self.models = FakeModels(client_name)

    monkeypatch.setattr(llm_judge, "_import_google_genai", lambda: (SimpleNamespace(), SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)))

    result = asyncio.run(
        _call_gemini_api_async(
            async_client={
                "api_keys": ["key-a", "key-b"],
                "active_key_index": 0,
                "exhausted_key_indices": set(),
                "lock": asyncio.Lock(),
                "clients": [
                    SimpleNamespace(aio=FakeAio("client-a")),
                    SimpleNamespace(aio=FakeAio("client-b")),
                ],
            },
            model_name="gemma",
            contents=["hi"],
            temperature=0.1,
            max_output_tokens=64,
            key_index=1,
        )
    )

    assert used_clients == ["client-b"]
    assert result.startswith("{")


def test_judge_single_async_loads_images_once(monkeypatch):
    calls = {"load_images": 0}

    def fake_load_images(record, max_pixels):
        calls["load_images"] += 1
        return [], False

    async def fake_judge_once(async_client, model_config, record, temperature, max_image_pixels, max_output_tokens, images_pil=None, image_missing=None, key_index=None):
        return LLMJudgeRecord(id=record.id, label_llm1=0)

    monkeypatch.setattr("src.llm_judge._load_images", fake_load_images)
    monkeypatch.setattr("src.llm_judge._judge_once_async", fake_judge_once)

    result = asyncio.run(
        judge_single_async(
            async_client={"lock": asyncio.Lock()},
            model_config={"provider": "gemini_api", "tag": "gemma"},
            record=InputRecord(id=1, text="caption", image_path="img.jpg"),
            temperature=0.1,
        )
    )

    assert result.label_llm1 == 0
    assert calls["load_images"] == 1
