from __future__ import annotations

"""Round-1 weak annotation pipeline."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .fusion_router import RouterConfig, apply_audit_sampling, route_all
from .llm_judge import KeyExhaustedError, QuotaExceededError, close_async_api_client, get_gemini_key_count, get_openrouter_key_count, judge_single_async, load_async_api_client
from .loaders import load_input_records
from .schemas import InputRecord, LLMJudgeRecord, Round1OutputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT_DATA = "data/raw-data/raw_data.json"
RESULTS_FILENAME = "round1_results.jsonl"
RESULTS_JSON_FILENAME = "round1_results.json"
DEFAULT_CONCURRENCY = 4
SUPPORTED_MODELS = ("gemma", "nemotron")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_router_config(cfg: dict) -> RouterConfig:
    return RouterConfig(
        random_audit_rate=float(cfg.get("random_audit_rate", 0.10)),
        seed=int(cfg.get("seed", 42)),
    )


def resolve_model_config(cfg: dict, model_tag: Optional[str] = None) -> Dict[str, Any]:
    configured_models = cfg.get("models") or {}
    resolved_tag = model_tag or cfg.get("default_model") or "gemma"
    if resolved_tag not in configured_models:
        available = ", ".join(sorted(configured_models)) or ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model tag '{resolved_tag}'. Available: {available}")

    model_cfg = configured_models[resolved_tag] or {}
    provider = model_cfg.get("provider")
    model_name = model_cfg.get("model_name")
    if not provider or not model_name:
        raise ValueError(f"Model config for '{resolved_tag}' must define provider and model_name.")

    return {
        "tag": resolved_tag,
        "provider": str(provider),
        "model_name": str(model_name),
        "reasoning": dict(model_cfg.get("reasoning") or {}),
        "max_output_tokens": int(model_cfg.get("max_output_tokens")) if model_cfg.get("max_output_tokens") is not None else None,
        "concurrency": int(model_cfg.get("concurrency")) if model_cfg.get("concurrency") is not None else None,
    }


def _results_path(output_dir: Path) -> Path:
    return output_dir / RESULTS_FILENAME


def _results_json_path(output_dir: Path) -> Path:
    return output_dir / RESULTS_JSON_FILENAME



def _format_elapsed(start_time: float) -> str:
    return f"{time.monotonic() - start_time:.1f}s"

def _llm_from_result_record(rec: Round1OutputRecord) -> LLMJudgeRecord:
    return LLMJudgeRecord(
        id=rec.id,
        label_llm1=rec.label_llm1,
        has_emoji=rec.has_emoji,
        needs_human_check=rec.needs_human_check,
        notes=rec.notes,
        reasoning=rec.reasoning,
        parse_error=rec.parse_error,
        image_missing=rec.image_missing,
    )


def load_checkpoint(output_dir: Path) -> Dict[int, LLMJudgeRecord]:
    path = _results_path(output_dir)
    if not path.exists():
        return {}

    cached: Dict[int, LLMJudgeRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            cached[payload["id"]] = LLMJudgeRecord(
                id=payload["id"],
                label_llm1=payload["label_llm1"],
                has_emoji=payload.get("has_emoji"),
                needs_human_check=payload.get("needs_human_check"),
                notes=payload.get("notes", ""),
                reasoning=payload.get("reasoning") or {},
                parse_error=bool(payload.get("parse_error", False)),
                image_missing=bool(payload.get("image_missing", False)),
            )
        except Exception:
            pass

    if cached:
        logger.info("Resume | cached=%d | file=%s", len(cached), path.name)
    return cached


def select_failed_records_for_rerun(
    records: List[InputRecord],
    cached: Dict[int, LLMJudgeRecord],
) -> List[InputRecord]:
    return [
        record for record in records
        if (cached_rec := cached.get(record.id)) is not None and cached_rec.label_llm1 == -1
    ]


def write_results(output_dir: Path, records: List[Round1OutputRecord]) -> None:
    jsonl_path = _results_path(output_dir)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    json_path = _results_json_path(output_dir)
    payload = [rec.model_dump(mode="json") for rec in records]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_records_for_run(
    records: List[InputRecord],
    *,
    test_mode: bool,
    test_size: int,
    seed: int,
) -> List[InputRecord]:
    del seed
    if not test_mode:
        return records
    if len(records) <= test_size:
        return records
    return records[:test_size]


def build_stats(all_records: List[Round1OutputRecord], total_samples: int) -> dict:
    processed = len(all_records)
    auto_accepted = [r for r in all_records if not r.need_review]
    human_queue = [r for r in all_records if r.need_review]

    label_dist: dict = {"sarcastic": 0, "non_sarcastic": 0, "invalid": 0}
    for r in all_records:
        label_dist[r.round1_label] = label_dist.get(r.round1_label, 0) + 1

    route_dist: dict = {}
    for r in all_records:
        route_dist[r.route_reason] = route_dist.get(r.route_reason, 0) + 1

    return {
        "total_samples": total_samples,
        "processed_samples": processed,
        "auto_accepted_count": len(auto_accepted),
        "need_review_count": len(human_queue),
        "label_distribution": label_dist,
        "route_reason_distribution": route_dist,
    }


def _save_checkpoint_results(
    output_dir: Path,
    input_records: List[InputRecord],
    llm_results_by_id: Dict[int, LLMJudgeRecord],
    router_cfg: RouterConfig,
) -> None:
    completed_inputs = [record for record in input_records if record.id in llm_results_by_id]
    completed_results = [llm_results_by_id[record.id] for record in completed_inputs]
    routed = route_all(completed_inputs, completed_results, router_cfg)
    write_results(output_dir, routed)


def _merge_results(
    current_results_by_id: Dict[int, LLMJudgeRecord],
    base_results_by_id: Optional[Dict[int, LLMJudgeRecord]] = None,
) -> Dict[int, LLMJudgeRecord]:
    merged = dict(base_results_by_id or {})
    merged.update(current_results_by_id)
    return merged


async def _cancel_pending_tasks(tasks: List[asyncio.Task]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _get_provider_name(model_config: Dict[str, Any]) -> str:
    return "OpenRouter" if model_config.get("provider") == "openrouter" else "Gemini"


async def run_llm_with_checkpoint(
    records: List[InputRecord],
    model_config: Dict[str, Any],
    temperature: float,
    output_dir: Path,
    router_cfg: RouterConfig,
    api_key: Optional[str] = None,
    max_image_pixels: int = 300_000,
    max_output_tokens: int = 256,
    max_retries: int = 3,
    retry_delay_seconds: int = 5,
    max_retry_delay_seconds: int = 20,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint_every: int = 10,
    *,
    load_checkpoint_file: bool = True,
    parallel_keys: bool = False,
    per_key_concurrency: int = 1,
    cached_results_by_id: Optional[Dict[int, LLMJudgeRecord]] = None,
    checkpoint_records: Optional[List[InputRecord]] = None,
    checkpoint_base_results_by_id: Optional[Dict[int, LLMJudgeRecord]] = None,
    started_at: Optional[float] = None,
) -> List[LLMJudgeRecord]:
    cached = dict(cached_results_by_id) if cached_results_by_id is not None else (load_checkpoint(output_dir) if load_checkpoint_file else {})
    results_by_id: Dict[int, LLMJudgeRecord] = {record.id: cached[record.id] for record in records if record.id in cached}
    remaining = [record for record in records if record.id not in results_by_id]

    if not remaining:
        return [results_by_id[record.id] for record in records if record.id in results_by_id]

    started_at = started_at or time.monotonic()
    async_client = load_async_api_client(model_config["provider"], api_key)
    if parallel_keys:
        if not isinstance(async_client, dict) or "api_keys" not in async_client:
            raise ValueError("Parallel key mode requires a provider with multiple API keys.")
        key_count = len(async_client["api_keys"])
        per_key_concurrency = max(1, per_key_concurrency)
        checkpoint_every = max(1, checkpoint_every)
        completed_since_save = 0
        completed_total = len(results_by_id)
        queue: asyncio.Queue[InputRecord] = asyncio.Queue()
        for record in remaining:
            queue.put_nowait(record)
        exhausted_key_indices: set[int] = set(async_client.get("exhausted_key_indices", set()))
        state_lock = asyncio.Lock()

        def _save_parallel_checkpoint() -> None:
            _save_checkpoint_results(
                output_dir,
                checkpoint_records or records,
                _merge_results(results_by_id, checkpoint_base_results_by_id),
                router_cfg,
            )

        async def _worker(key_index: int) -> None:
            nonlocal completed_since_save, completed_total
            while key_index not in exhausted_key_indices:
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if queue.empty():
                        return
                    continue

                try:
                    result = await judge_single_async(
                        async_client,
                        model_config,
                        record,
                        temperature,
                        max_image_pixels,
                        max_output_tokens,
                        max_retries,
                        retry_delay_seconds,
                        max_retry_delay_seconds,
                        key_index=key_index,
                        allow_key_rotation=False,
                    )
                except KeyExhaustedError:
                    async with state_lock:
                        exhausted_key_indices.add(key_index)
                        async_client["exhausted_key_indices"].add(key_index)
                        if len(exhausted_key_indices) >= key_count:
                            queue.task_done()
                            return
                        queue.put_nowait(record)
                        logger.warning(
                            "ExhaustKey | model=%s | key=%d/%d | remaining_keys=%d",
                            model_config.get("tag", model_config.get("model_name", "unknown")),
                            key_index + 1,
                            key_count,
                            key_count - len(exhausted_key_indices),
                        )
                    queue.task_done()
                    return

                async with state_lock:
                    results_by_id[result.id] = result
                    completed_total += 1
                    completed_since_save += 1
                    logger.info("Progress | %d/%d | id=%d | label=%s | elapsed=%s", completed_total, len(records), result.id, result.label_llm1, _format_elapsed(started_at))
                    if completed_since_save >= checkpoint_every:
                        _save_parallel_checkpoint()
                        completed_since_save = 0
                        logger.info("Checkpoint | saved=%d | elapsed=%s | files=%s,%s", completed_total, _format_elapsed(started_at), _results_path(output_dir).name, _results_json_path(output_dir).name)
                queue.task_done()

        worker_count = key_count * per_key_concurrency
        workers = [
            asyncio.create_task(_worker(key_index))
            for key_index in range(key_count)
            for _ in range(per_key_concurrency)
        ]
        try:
            await asyncio.gather(*workers)
            if completed_since_save > 0:
                _save_parallel_checkpoint()
                logger.info("Checkpoint | saved=%d | elapsed=%s | files=%s,%s", completed_total, _format_elapsed(started_at), _results_path(output_dir).name, _results_json_path(output_dir).name)
            if not queue.empty() and len(exhausted_key_indices) >= key_count:
                raise QuotaExceededError(f"All {_get_provider_name(model_config)} API keys exhausted.")
        finally:
            await close_async_api_client()
        return [results_by_id[record.id] for record in records if record.id in results_by_id]

    semaphore = asyncio.Semaphore(max(1, concurrency))
    checkpoint_every = max(1, checkpoint_every)
    completed_since_save = 0
    completed_total = len(results_by_id)

    async def _run_one(record: InputRecord) -> LLMJudgeRecord:
        async with semaphore:
            return await judge_single_async(
                async_client,
                model_config,
                record,
                temperature,
                max_image_pixels,
                max_output_tokens,
                max_retries,
                retry_delay_seconds,
                max_retry_delay_seconds,
            )

    tasks = [asyncio.create_task(_run_one(record)) for record in remaining]

    try:
        for finished in asyncio.as_completed(tasks):
            try:
                result = await finished
            except QuotaExceededError as exc:
                await _cancel_pending_tasks(tasks)
                if results_by_id:
                    _save_checkpoint_results(
                        output_dir,
                        checkpoint_records or records,
                        _merge_results(results_by_id, checkpoint_base_results_by_id),
                        router_cfg,
                    )
                    logger.info("Checkpoint | saved=%d | elapsed=%s | files=%s,%s", completed_total, _format_elapsed(started_at), _results_path(output_dir).name, _results_json_path(output_dir).name)
                logger.error("Stop | reason=quota_exceeded | saved=%d | elapsed=%s | error=%s", completed_total, _format_elapsed(started_at), str(exc)[:200])
                raise

            results_by_id[result.id] = result
            completed_total += 1
            completed_since_save += 1

            logger.info("Progress | %d/%d | id=%d | label=%s | elapsed=%s", completed_total, len(records), result.id, result.label_llm1, _format_elapsed(started_at))

            if completed_since_save >= checkpoint_every:
                _save_checkpoint_results(
                    output_dir,
                    checkpoint_records or records,
                    _merge_results(results_by_id, checkpoint_base_results_by_id),
                    router_cfg,
                )
                completed_since_save = 0
                logger.info("Checkpoint | saved=%d | elapsed=%s | files=%s,%s", completed_total, _format_elapsed(started_at), _results_path(output_dir).name, _results_json_path(output_dir).name)

        if completed_since_save > 0:
            _save_checkpoint_results(
                output_dir,
                checkpoint_records or records,
                _merge_results(results_by_id, checkpoint_base_results_by_id),
                router_cfg,
            )
            logger.info("Checkpoint | saved=%d | elapsed=%s | files=%s,%s", completed_total, _format_elapsed(started_at), _results_path(output_dir).name, _results_json_path(output_dir).name)
    finally:
        await close_async_api_client()

    return [results_by_id[record.id] for record in records if record.id in results_by_id]


async def run_pipeline_async(
    config_path: str,
    output_dir: str,
    input_data: str = DEFAULT_INPUT_DATA,
    api_key: Optional[str] = None,
    *,
    min_record_id: Optional[int] = None,
    max_record_id: Optional[int] = None,
    no_checkpoint_load: bool = False,
    rerun_minus_one: bool = False,
    parallel_keys: bool = False,
    per_key_concurrency: int = 1,
    test_mode: bool = False,
    test_size: int = 5,
    model_tag: Optional[str] = None,
) -> None:
    cfg = load_config(config_path)
    router_cfg = build_router_config(cfg)
    resolved_model = resolve_model_config(cfg, model_tag)
    temperature = float(cfg.get("llm_temperature", 0.1))
    max_image_pixels = int(cfg.get("max_image_pixels", 300_000))
    max_output_tokens = int(resolved_model.get("max_output_tokens") or cfg.get("max_output_tokens", 256))
    max_retries = int(cfg.get("max_retries", 3))
    retry_delay_seconds = int(cfg.get("retry_delay_seconds", 5))
    max_retry_delay_seconds = int(cfg.get("max_retry_delay_seconds", 20))
    concurrency = int(resolved_model.get("concurrency") or cfg.get("concurrency", DEFAULT_CONCURRENCY))
    checkpoint_every = int(cfg.get("checkpoint_every", 10))
    seed = int(cfg.get("seed", 42))

    pipeline_started_at = time.monotonic()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rerun_minus_one and no_checkpoint_load:
        raise ValueError("--rerun-minus-one cannot be used with --no-checkpoint-load.")
    if no_checkpoint_load:
        _results_path(out_dir).unlink(missing_ok=True)
        _results_json_path(out_dir).unlink(missing_ok=True)

    all_input_records = load_input_records(input_data)
    input_records = all_input_records
    if min_record_id is not None:
        input_records = [record for record in input_records if record.id >= min_record_id]
    if max_record_id is not None:
        input_records = [record for record in input_records if record.id <= max_record_id]
    input_records = select_records_for_run(input_records, test_mode=test_mode, test_size=test_size, seed=seed)

    cached_results = load_checkpoint(out_dir) if not no_checkpoint_load else {}
    records_for_run = input_records
    if rerun_minus_one:
        records_for_run = select_failed_records_for_rerun(input_records, cached_results)
        logger.info(
            "RerunMinusOne | candidates=%d | selected=%d | output=%s",
            len(input_records),
            len(records_for_run),
            _results_json_path(out_dir).name,
        )
        if not records_for_run:
            logger.info("Done | processed=0 | auto=0 | review=0 | invalid=0 | audit=0 | elapsed=%s", _format_elapsed(pipeline_started_at))
            return

    logger.info(
        "Start | input=%d | test=%s | from_id=%s | model=%s | provider=%s | concurrency=%d | checkpoint_every=%d | parallel_keys=%s | per_key_concurrency=%d | output=%s",
        len(records_for_run),
        test_mode,
        f"{min_record_id if min_record_id is not None else 'start'}..{max_record_id if max_record_id is not None else 'end'}",
        resolved_model["tag"],
        resolved_model["provider"],
        concurrency,
        checkpoint_every,
        parallel_keys,
        per_key_concurrency,
        _results_json_path(out_dir).name,
    )
    logger.info("ModelConfig | tag=%s | name=%s", resolved_model["tag"], resolved_model["model_name"])
    if resolved_model["provider"] == "openrouter":
        key_count = get_openrouter_key_count(api_key)
        logger.info("OpenRouterKeys | active_key=1/%d | total_keys=%d", key_count, key_count)
    elif resolved_model["provider"] == "gemini_api":
        key_count = get_gemini_key_count(api_key)
        logger.info("GeminiKeys | active_key=1/%d | total_keys=%d", key_count, key_count)

    try:
        llm_results = await run_llm_with_checkpoint(
            records_for_run,
            resolved_model,
            temperature,
            out_dir,
            router_cfg,
            api_key,
            max_image_pixels,
            max_output_tokens,
            max_retries,
            retry_delay_seconds,
            max_retry_delay_seconds,
            concurrency,
            checkpoint_every,
            load_checkpoint_file=False,
            parallel_keys=parallel_keys,
            per_key_concurrency=per_key_concurrency,
            cached_results_by_id=cached_results,
            checkpoint_records=all_input_records,
            checkpoint_base_results_by_id=cached_results,
            started_at=pipeline_started_at,
        )
    except QuotaExceededError:
        logger.error("Stopped pipeline due to quota exceeded | elapsed=%s | checkpoint_files=%s,%s", _format_elapsed(pipeline_started_at), _results_path(out_dir).name, _results_json_path(out_dir).name)
        return

    merged_results = _merge_results({result.id: result for result in llm_results}, cached_results)
    completed_inputs = [record for record in all_input_records if record.id in merged_results]
    completed_results = [merged_results[record.id] for record in completed_inputs]
    routed = route_all(completed_inputs, completed_results, router_cfg)
    routed, audit_count = apply_audit_sampling(routed, router_cfg.random_audit_rate, router_cfg.seed)
    write_results(out_dir, routed)

    stats = build_stats(routed, len(input_records) if rerun_minus_one else len(records_for_run))
    logger.info(
        "Done | processed=%d | auto=%d | review=%d | invalid=%d | audit=%d | elapsed=%s",
        stats["processed_samples"],
        stats["auto_accepted_count"],
        stats["need_review_count"],
        stats["label_distribution"].get("invalid", 0),
        audit_count,
        _format_elapsed(pipeline_started_at),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-1 weak annotation pipeline")
    parser.add_argument("--input_data", default=DEFAULT_INPUT_DATA, help=f"Path to input JSON array or JSONL (default: {DEFAULT_INPUT_DATA})")
    parser.add_argument("--config", required=True, help="Path to configs/round1.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--api_key", default=None, help="API key for the selected provider")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=None, help="VLM tag to use: gemma or nemotron")
    parser.add_argument("--test_mode", action="store_true", help="Run on the first 5 filtered records")
    parser.add_argument("--test_size", type=int, default=5, help="Leading records to take in test mode")
    parser.add_argument("--min-record-id", type=int, default=None, help="Keep only rows with id>=N")
    parser.add_argument("--from", dest="min_record_id", type=int, help="Alias of --min-record-id; run records with id>=N")
    parser.add_argument("--to", dest="max_record_id", type=int, help="Alias of --max-record-id; run records with id<=N")
    parser.add_argument("--parallel-keys", action="store_true", help="Send requests across multiple API keys in parallel and stop using keys that hit quota")
    parser.add_argument("--per-key-concurrency", type=int, default=1, help="Concurrency per API key when --parallel-keys is enabled")
    parser.add_argument("--rerun-minus-one", action="store_true", help="Rerun only records with label_llm1 = -1 from the existing checkpoint and update the same output files")
    parser.add_argument("--no-checkpoint-load", action="store_true", help="Ignore previous round1_results.jsonl")
    args = parser.parse_args()

    asyncio.run(
        run_pipeline_async(
            config_path=args.config,
            output_dir=args.output_dir,
            input_data=args.input_data,
            api_key=args.api_key,
            min_record_id=args.min_record_id,
            max_record_id=args.max_record_id,
            no_checkpoint_load=args.no_checkpoint_load,
            rerun_minus_one=args.rerun_minus_one,
            parallel_keys=args.parallel_keys,
            per_key_concurrency=args.per_key_concurrency,
            test_mode=args.test_mode,
            test_size=args.test_size,
            model_tag=args.model,
        )
    )


if __name__ == "__main__":
    main()
