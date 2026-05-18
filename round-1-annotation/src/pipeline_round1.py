from __future__ import annotations

"""Round-1 weak annotation pipeline (LLM-as-a-judge via Gemma API)."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .fusion_router import RouterConfig, apply_audit_sampling, route_all
from .llm_judge import close_async_api_client, judge_single_async, load_async_api_client
from .loaders import load_input_records
from .schemas import InputRecord, LLMJudgeRecord, Round1OutputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT_DATA = "data/bronze/raw_data.json"
RESULTS_FILENAME = "round1_results.jsonl"
DEFAULT_CONCURRENCY = 4


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_router_config(cfg: dict) -> RouterConfig:
    return RouterConfig(
        random_audit_rate=float(cfg.get("random_audit_rate", 0.10)),
        seed=int(cfg.get("seed", 42)),
    )


def _results_path(output_dir: Path) -> Path:
    return output_dir / RESULTS_FILENAME


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
            rec = Round1OutputRecord(**json.loads(line))
            cached[rec.id] = _llm_from_result_record(rec)
        except Exception:
            pass

    if cached:
        logger.info("Resume | cached=%d | file=%s", len(cached), path.name)
    return cached


def write_results(output_dir: Path, records: List[Round1OutputRecord]) -> None:
    path = _results_path(output_dir)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


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


async def run_llm_with_checkpoint(
    records: List[InputRecord],
    model_name: str,
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
) -> List[LLMJudgeRecord]:
    cached = load_checkpoint(output_dir) if load_checkpoint_file else {}
    results_by_id: Dict[int, LLMJudgeRecord] = {record.id: cached[record.id] for record in records if record.id in cached}
    remaining = [record for record in records if record.id not in results_by_id]

    if not remaining:
        return [results_by_id[record.id] for record in records if record.id in results_by_id]

    async_client = load_async_api_client(api_key)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    checkpoint_every = max(1, checkpoint_every)
    completed_since_save = 0
    completed_total = len(results_by_id)

    async def _run_one(record: InputRecord) -> LLMJudgeRecord:
        async with semaphore:
            return await judge_single_async(
                async_client,
                model_name,
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
            result = await finished
            results_by_id[result.id] = result
            completed_total += 1
            completed_since_save += 1

            logger.info(
                "Progress | %d/%d | id=%d | label=%s",
                completed_total,
                len(records),
                result.id,
                result.label_llm1,
            )

            if completed_since_save >= checkpoint_every:
                _save_checkpoint_results(output_dir, records, results_by_id, router_cfg)
                completed_since_save = 0
                logger.info("Checkpoint | saved=%d | file=%s", completed_total, _results_path(output_dir).name)

        if completed_since_save > 0:
            _save_checkpoint_results(output_dir, records, results_by_id, router_cfg)
            logger.info("Checkpoint | saved=%d | file=%s", completed_total, _results_path(output_dir).name)
    finally:
        await close_async_api_client()

    return [results_by_id[record.id] for record in records if record.id in results_by_id]


async def run_pipeline_async(
    config_path: str,
    output_dir: str,
    input_data: str = DEFAULT_INPUT_DATA,
    api_key: Optional[str] = None,
    max_records: Optional[int] = None,
    *,
    min_record_id: Optional[int] = None,
    no_checkpoint_load: bool = False,
    test_mode: bool = False,
    test_size: int = 5,
) -> None:
    cfg = load_config(config_path)
    router_cfg = build_router_config(cfg)
    model_name = cfg.get("llm_model", "gemma-4-31b-it")
    temperature = float(cfg.get("llm_temperature", 0.1))
    max_image_pixels = int(cfg.get("max_image_pixels", 300_000))
    max_output_tokens = int(cfg.get("max_output_tokens", 256))
    max_retries = int(cfg.get("max_retries", 3))
    retry_delay_seconds = int(cfg.get("retry_delay_seconds", 5))
    max_retry_delay_seconds = int(cfg.get("max_retry_delay_seconds", 20))
    concurrency = int(cfg.get("concurrency", DEFAULT_CONCURRENCY))
    checkpoint_every = int(cfg.get("checkpoint_every", 10))
    seed = int(cfg.get("seed", 42))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if no_checkpoint_load:
        _results_path(out_dir).unlink(missing_ok=True)

    input_records = load_input_records(input_data)
    if max_records is not None:
        input_records = input_records[:max_records]
    if min_record_id is not None:
        input_records = [record for record in input_records if record.id >= min_record_id]
    input_records = select_records_for_run(
        input_records,
        test_mode=test_mode,
        test_size=test_size,
        seed=seed,
    )

    logger.info(
        "Start | input=%d | test=%s | concurrency=%d | checkpoint_every=%d | output=%s",
        len(input_records),
        test_mode,
        concurrency,
        checkpoint_every,
        _results_path(out_dir).name,
    )

    llm_results = await run_llm_with_checkpoint(
        input_records,
        model_name,
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
        load_checkpoint_file=not no_checkpoint_load,
    )

    routed = route_all(input_records, llm_results, router_cfg)
    routed, audit_count = apply_audit_sampling(routed, router_cfg.random_audit_rate, router_cfg.seed)
    write_results(out_dir, routed)

    stats = build_stats(routed, len(input_records))
    logger.info(
        "Done | processed=%d | auto=%d | review=%d | invalid=%d | audit=%d",
        stats["processed_samples"],
        stats["auto_accepted_count"],
        stats["need_review_count"],
        stats["label_distribution"].get("invalid", 0),
        audit_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-1 weak annotation pipeline (Gemma API)")
    parser.add_argument(
        "--input_data",
        default=DEFAULT_INPUT_DATA,
        help=f"Path to input JSON array or JSONL (default: {DEFAULT_INPUT_DATA})",
    )
    parser.add_argument("--config", required=True, help="Path to configs/round1.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--api_key", default=None, help="Gemini API key")
    parser.add_argument("--max_records", type=int, default=None, help="Limit to first N records")
    parser.add_argument("--test_mode", action="store_true", help="Run on the first 5 filtered records")
    parser.add_argument("--test_size", type=int, default=5, help="Leading records to take in test mode")
    parser.add_argument("--min-record-id", type=int, default=None, help="Keep only rows with id>=N")
    parser.add_argument("--no-checkpoint-load", action="store_true", help="Ignore previous round1_results.jsonl")
    args = parser.parse_args()

    asyncio.run(
        run_pipeline_async(
            config_path=args.config,
            output_dir=args.output_dir,
            input_data=args.input_data,
            api_key=args.api_key,
            max_records=args.max_records,
            min_record_id=args.min_record_id,
            no_checkpoint_load=args.no_checkpoint_load,
            test_mode=args.test_mode,
            test_size=args.test_size,
        )
    )


if __name__ == "__main__":
    main()
