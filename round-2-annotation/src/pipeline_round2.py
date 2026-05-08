from __future__ import annotations

"""
Round-1 weak annotation pipeline (LLM-as-a-judge via local model inference).

Requires a CUDA GPU. Designed to run on Kaggle (free T4/P100 GPU).

Optional env var for downloading the model from HuggingFace Hub:
    export HF_TOKEN="hf_..."

Usage:
    python -m src.pipeline_round1 \
        --input_data ../data/data-sample.json \
        --config configs/round1.yaml \
        --output_dir outputs/

Outputs written during inference (crash-safe):
    ``.checkpoint_llm.jsonl``
        Raw LLM judge structs; entire file rewritten after every batch (resume).

    ``round1_progress.jsonl``
        One JSON object per finished row (same slim fields as ``round1_all.json``),
        **appended after each batch**. If the session dies, download this file:
        it contains all work completed up to the last batch. The final
        ``round1_all.json`` at the end of a full run still adds audit sampling
        and splits; slim fields are unchanged by audit.

``--no-checkpoint-load``
    Do not read ``.checkpoint_llm.jsonl`` at startup (e.g. Kaggle disk wiped).
    Still writes both files after each batch. Truncates ``round1_progress.jsonl``
    at run start so you do not mix with an old session.

``--min-record-id N``
    Keep only input rows whose ``id >= N``, then run the pipeline on that slice.
    Use this to continue in a **new** session without re-running GPU on ids
    below ``N`` (you already saved those rows in a previous export, or you
    deliberately split the dataset). Does **not** read old ids from the JSON
    file; it just filters whatever is in ``--input_data``.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .fusion_router import RouterConfig, apply_audit_sampling, route_all, route_single
from .llm_judge import judge_batch
from .loaders import load_input_records
from .schemas import InputRecord, LLMJudgeRecord, Round1OutputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_router_config(cfg: dict) -> RouterConfig:
    return RouterConfig(
        random_audit_rate=float(cfg.get("random_audit_rate", 0.10)),
        seed=int(cfg.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / ".checkpoint_llm.jsonl"


def _progress_jsonl_path(output_dir: Path) -> Path:
    return output_dir / "round1_progress.jsonl"


def load_checkpoint(output_dir: Path) -> Dict[int, LLMJudgeRecord]:
    cp = _checkpoint_path(output_dir)
    if not cp.exists():
        return {}
    cached: Dict[int, LLMJudgeRecord] = {}
    for line in cp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = LLMJudgeRecord(**json.loads(line))
            cached[rec.id] = rec
        except Exception:
            pass
    logger.info("Loaded %d cached LLM results from checkpoint", len(cached))
    return cached


def save_checkpoint(output_dir: Path, records: List[LLMJudgeRecord]) -> None:
    cp = _checkpoint_path(output_dir)
    with cp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# LLM runner with checkpoint + progress
# ---------------------------------------------------------------------------

def _iter_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run_llm_with_checkpoint(
    records: List[InputRecord],
    model_name: str,
    temperature: float,
    batch_size: int,
    output_dir: Path,
    router_cfg: RouterConfig,
    input_by_id: Dict[int, InputRecord],
    hf_token: Optional[str] = None,
    device: str = "cuda",
    load_in_4bit: bool = False,
    max_image_pixels: int = 1_048_576,
    *,
    load_checkpoint_file: bool = True,
) -> List[LLMJudgeRecord]:
    if load_checkpoint_file:
        cached = load_checkpoint(output_dir)
    else:
        cached = {}
        logger.info(
            "Checkpoint load disabled (--no-checkpoint-load); "
            "starting inference without resume file."
        )
    remaining = [r for r in records if r.id not in cached]
    all_results: List[LLMJudgeRecord] = [cached[r.id] for r in records if r.id in cached]

    if not remaining:
        logger.info("All %d records found in checkpoint, skipping inference", len(records))
        return all_results

    logger.info(
        "Running LLM on %d records (%d already cached)", len(remaining), len(cached)
    )

    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx, batch in enumerate(_iter_batches(remaining, batch_size), start=1):
        batch_results = judge_batch(
            batch, model_name, temperature, hf_token, device, load_in_4bit,
            max_image_pixels,
        )
        all_results.extend(batch_results)
        save_checkpoint(output_dir, all_results)
        _append_round1_progress_jsonl(
            output_dir, batch_results, input_by_id, router_cfg
        )
        logger.info(
            "Appended %d rows to %s",
            len(batch_results),
            _progress_jsonl_path(output_dir).name,
        )
        logger.info(
            "Batch %d/%d done | records in batch: %d | total completed: %d/%d",
            batch_idx, n_batches, len(batch), len(all_results), len(records),
        )

    return all_results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def build_stats(
    all_records: List[Round1OutputRecord],
    bad_count: int,
    total_samples: int,
) -> dict:
    processed = len(all_records)
    auto_accepted = [r for r in all_records if not r.need_review]
    human_queue = [r for r in all_records if r.need_review]

    label_dist: dict = {"sarcastic": 0, "non_sarcastic": 0, "invalid": 0}
    for r in all_records:
        label_dist[r.round1_label] = label_dist.get(r.round1_label, 0) + 1

    route_dist: dict = {}
    for r in all_records:
        route_dist[r.route_reason] = route_dist.get(r.route_reason, 0) + 1

    has_emoji_dist: dict = {0: 0, 1: 0, "null": 0}
    needs_human_check_dist: dict = {0: 0, 1: 0, "null": 0}
    for r in all_records:
        has_emoji_dist[r.has_emoji if r.has_emoji is not None else "null"] += 1
        needs_human_check_dist[r.needs_human_check if r.needs_human_check is not None else "null"] += 1

    return {
        "total_samples": total_samples,
        "processed_samples": processed,
        "bad_records": bad_count,
        "auto_accepted_count": len(auto_accepted),
        "need_review_count": len(human_queue),
        "auto_accept_rate": round(len(auto_accepted) / processed, 4) if processed else 0,
        "need_review_rate": round(len(human_queue) / processed, 4) if processed else 0,
        "label_distribution": label_dist,
        "route_reason_distribution": route_dist,
        "has_emoji_distribution": {str(k): v for k, v in has_emoji_dist.items()},
        "needs_human_check_distribution": {str(k): v for k, v in needs_human_check_dist.items()},
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _slim_round1_public_dict(rec: Round1OutputRecord) -> dict:
    """
    Final JSON exports only these fields (routing/metadata stripped from disk).
    ``timestamp`` replaces internal ``timestamp_utc`` key name for the artifact.
    """
    return {
        "id": rec.id,
        "text": rec.text,
        "image_path": rec.image_path,
        "label_llm1": rec.label_llm1,
        "has_emoji": rec.has_emoji,
        "needs_human_check": rec.needs_human_check,
        "reasoning": rec.reasoning,
        "timestamp": rec.timestamp_utc,
    }


def _append_round1_progress_jsonl(
    output_dir: Path,
    batch_llm: List[LLMJudgeRecord],
    input_by_id: Dict[int, InputRecord],
    router_cfg: RouterConfig,
) -> None:
    """Append slim public schema: one JSON object per line for this batch."""
    path = _progress_jsonl_path(output_dir)
    with path.open("a", encoding="utf-8") as f:
        for llm_rec in batch_llm:
            inp = input_by_id.get(llm_rec.id)
            if inp is None:
                continue
            override = None
            if llm_rec.image_missing:
                override = "missing_image"
            elif llm_rec.parse_error:
                override = "invalid_json"
            routed = route_single(
                llm_rec, router_cfg, inp.text, inp.image_path, override
            )
            f.write(
                json.dumps(_slim_round1_public_dict(routed), ensure_ascii=False)
                + "\n"
            )


def _write_round1_json(path: Path, records: List[Round1OutputRecord]) -> None:
    """Write Round-1 labeled rows as JSON array with public slim schema."""
    data = [_slim_round1_public_dict(r) for r in records]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_json(path: Path, records: list) -> None:
    """Write arbitrary serializable rows (e.g. bad_records)."""
    data = []
    for rec in records:
        if isinstance(rec, Round1OutputRecord):
            data.append(rec.model_dump())
        else:
            data.append(rec)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_outputs(
    output_dir: Path,
    all_records: List[Round1OutputRecord],
    bad_records: list,
    stats: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    auto_accepted = [r for r in all_records if not r.need_review]
    human_queue = [r for r in all_records if r.need_review]

    _write_round1_json(output_dir / "round1_all.json", all_records)
    _write_round1_json(output_dir / "round1_auto_accepted.json", auto_accepted)
    _write_round1_json(output_dir / "round1_human_queue.json", human_queue)
    _write_json(output_dir / "bad_records.json", bad_records)

    (output_dir / "round1_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Outputs written to %s | all=%d | auto=%d | need_review=%d | bad=%d | "
        "incremental rows in round1_progress.jsonl (during run)",
        output_dir, len(all_records), len(auto_accepted), len(human_queue), len(bad_records),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_data: str,
    config_path: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    max_records: Optional[int] = None,
    *,
    min_record_id: Optional[int] = None,
    no_checkpoint_load: bool = False,
) -> None:
    cfg = load_config(config_path)
    router_cfg = build_router_config(cfg)
    model_name = cfg.get("llm_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    temperature = float(cfg.get("llm_temperature", 0.1))
    batch_size = int(cfg.get("batch_size", 8))
    device = cfg.get("device", "cuda")
    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    max_image_pixels = int(cfg.get("max_image_pixels", 1_048_576))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = _progress_jsonl_path(out_dir)
    if no_checkpoint_load:
        progress_path.unlink(missing_ok=True)
        logger.info(
            "Truncated %s (fresh run with --no-checkpoint-load)",
            progress_path.name,
        )

    logger.info("=== Round-1 Pipeline Start ===")
    logger.info(
        "Model: %s | device: %s | 4bit: %s | max_img_px: %s",
        model_name, device, load_in_4bit,
        max_image_pixels if max_image_pixels > 0 else "unlimited",
    )
    logger.info("RouterConfig: %s", router_cfg)

    input_records = load_input_records(input_data)
    if max_records is not None:
        input_records = input_records[:max_records]
        logger.info("TEST MODE: limiting to first %d records", max_records)
    if min_record_id is not None:
        before = len(input_records)
        input_records = [r for r in input_records if r.id >= min_record_id]
        logger.info(
            "min_record_id=%s: kept %d / %d rows (id >= %s)",
            min_record_id,
            len(input_records),
            before,
            min_record_id,
        )
    total_samples = len(input_records)
    input_by_id = {r.id: r for r in input_records}

    llm_results = run_llm_with_checkpoint(
        input_records,
        model_name,
        temperature,
        batch_size,
        out_dir,
        router_cfg,
        input_by_id,
        hf_token,
        device,
        load_in_4bit,
        max_image_pixels,
        load_checkpoint_file=not no_checkpoint_load,
    )

    routed = route_all(input_records, llm_results, router_cfg)
    routed, _ = apply_audit_sampling(routed, router_cfg.random_audit_rate, router_cfg.seed)

    bad_count = total_samples - len(routed)
    stats = build_stats(routed, bad_count, total_samples)
    write_outputs(out_dir, routed, [], stats)

    logger.info("=== Round-1 Pipeline Complete ===")
    logger.info(
        "auto_accept_rate=%.2f | need_review_rate=%.2f | labels=%s",
        stats["auto_accept_rate"],
        stats["need_review_rate"],
        stats["label_distribution"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-1 weak annotation pipeline (local model inference)"
    )
    parser.add_argument("--input_data", required=True, help="Path to input JSON array or JSONL")
    parser.add_argument("--config", required=True, help="Path to configs/round1.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument(
        "--hf_token", default=None,
        help="HuggingFace token for model download (overrides HF_TOKEN env var)"
    )
    parser.add_argument(
        "--max_records", type=int, default=None,
        help="Limit to first N records (e.g. 5 for a quick smoke-test); omit to run all"
    )
    parser.add_argument(
        "--min-record-id",
        type=int,
        default=None,
        help=(
            "Keep only rows with id>=N from input_data, then run LLM on that slice "
            "(continue a job without re-processing lower ids)."
        ),
    )
    parser.add_argument(
        "--no-checkpoint-load",
        action="store_true",
        help=(
            "Skip reading .checkpoint_llm.jsonl; truncate round1_progress.jsonl at start. "
            "Still saves checkpoint + progress after each batch."
        ),
    )
    args = parser.parse_args()

    run_pipeline(
        input_data=args.input_data,
        config_path=args.config,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        max_records=args.max_records,
        min_record_id=args.min_record_id,
        no_checkpoint_load=args.no_checkpoint_load,
    )


if __name__ == "__main__":
    main()
