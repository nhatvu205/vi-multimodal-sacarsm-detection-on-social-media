from __future__ import annotations
"""
Round-2 Fine-grained annotation pipeline (LLM-as-a-judge via InternVL3.5-8B).
Requires a CUDA GPU. Designed to run on Kaggle.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .fusion_router import RouterConfig, apply_audit_sampling, route_all, route_single
from .llm_judge import judge_batch
from .loaders import load_input_records
from .schemas import InputRecord, LLMJudgeRecord, Round1OutputRecord  # Giữ tên schema cũ nếu chưa đổi
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
        random_audit_rate=float(cfg.get("random_audit_rate", 0.08)),
        seed=int(cfg.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# Checkpoint & Progress
# ---------------------------------------------------------------------------
def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / ".checkpoint_llm_round2.jsonl"


def _progress_jsonl_path(output_dir: Path) -> Path:
    return output_dir / "round2_progress.jsonl"


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
# Main LLM Runner
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
    load_in_4bit: bool = True,
    max_image_pixels: int = 500000,
    *,
    load_checkpoint_file: bool = True,
) -> List[LLMJudgeRecord]:
    
    if load_checkpoint_file:
        cached = load_checkpoint(output_dir)
    else:
        cached = {}
        logger.info("Checkpoint load disabled (--no-checkpoint-load)")

    remaining = [r for r in records if r.id not in cached]
    all_results: List[LLMJudgeRecord] = [cached[r.id] for r in records if r.id in cached]

    if not remaining:
        logger.info("All records found in checkpoint, skipping inference.")
        return all_results

    logger.info("Running LLM on %d new records (%d cached)", len(remaining), len(cached))

    for batch_idx, batch in enumerate(_iter_batches(remaining, batch_size), start=1):
        batch_results = judge_batch(
            records=batch,
            model_name=model_name,
            temperature=temperature,
            hf_token=hf_token,
            max_image_pixels=max_image_pixels,
        )
        all_results.extend(batch_results)
        save_checkpoint(output_dir, all_results)
        _append_round2_progress_jsonl(output_dir, batch_results, input_by_id, router_cfg)

        logger.info(
            f"Batch {batch_idx} done | {len(batch)} records | Total: {len(all_results)}/{len(records)}"
        )

    return all_results


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------
def _slim_round2_public_dict(rec: Round1OutputRecord) -> dict:
    """Bao gồm thêm các field fine-grained của Round 2"""
    return {
        "id": rec.id,
        "text": rec.text,
        "image_path": rec.image_path,
        "label_llm2": rec.label_llm2,
        "T": getattr(rec, "T", None),
        "I": getattr(rec, "I", None),
        "MM": getattr(rec, "MM", None),
        "KI": getattr(rec, "KI", None),
        "has_emoji": rec.has_emoji,
        "needs_human_check": rec.needs_human_check,
        "reasoning": rec.reasoning,
        "timestamp": rec.timestamp_utc,
    }


def _append_round2_progress_jsonl(
    output_dir: Path,
    batch_llm: List[LLMJudgeRecord],
    input_by_id: Dict[int, InputRecord],
    router_cfg: RouterConfig,
) -> None:
    path = _progress_jsonl_path(output_dir)
    with path.open("a", encoding="utf-8") as f:
        for llm_rec in batch_llm:
            inp = input_by_id.get(llm_rec.id)
            if inp is None:
                continue
            override = None
            if getattr(llm_rec, 'image_missing', False):
                override = "missing_image"
            elif getattr(llm_rec, 'parse_error', False):
                override = "invalid_json"

            routed = route_single(llm_rec, router_cfg, inp.text, inp.image_path, override)
            f.write(
                json.dumps(_slim_round2_public_dict(routed), ensure_ascii=False) + "\n"
            )


def _write_round2_json(path: Path, records: List[Round1OutputRecord]) -> None:
    data = [_slim_round2_public_dict(r) for r in records]
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

    _write_round2_json(output_dir / "round2_all.json", all_records)
    _write_round2_json(output_dir / "round2_auto_accepted.json", auto_accepted)
    _write_round2_json(output_dir / "round2_human_queue.json", human_queue)
    
    with (output_dir / "bad_records.json").open("w", encoding="utf-8") as f:
        json.dump(bad_records, f, ensure_ascii=False, indent=2)
    
    (output_dir / "round2_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Round 2 outputs written | all={len(all_records)} | "
        f"auto={len(auto_accepted)} | need_review={len(human_queue)}"
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    input_data: str,
    config_path: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    max_records: Optional[int] = None,
    min_record_id: Optional[int] = None,
    no_checkpoint_load: bool = False,
) -> None:
    prompt_dir = Path(config_path).parent.parent / "prompts" 
    cfg = load_config(config_path)
    router_cfg = build_router_config(cfg)

    model_name = cfg.get("llm_model", "OpenGVLab/InternVL3_5-8B")
    temperature = float(cfg.get("llm_temperature", 0.1))
    batch_size = int(cfg.get("batch_size", 4))          # Nên nhỏ vì model to
    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    max_image_pixels = int(cfg.get("max_image_pixels", 500000))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if no_checkpoint_load:
        _progress_jsonl_path(out_dir).unlink(missing_ok=True)

    logger.info("=== Round-2 Fine-grained Pipeline Start ===")
    logger.info(f"Model: {model_name} | 4bit: {load_in_4bit} | batch_size: {batch_size}")

    input_records = load_input_records(input_data)
    
    if max_records:
        input_records = input_records[:max_records]
    if min_record_id:
        input_records = [r for r in input_records if r.id >= min_record_id]

    total_samples = len(input_records)
    input_by_id = {r.id: r for r in input_records}

    llm_results = run_llm_with_checkpoint(
        records=input_records,
        model_name=model_name,
        temperature=temperature,
        batch_size=batch_size,
        output_dir=out_dir,
        router_cfg=router_cfg,
        input_by_id=input_by_id,
        hf_token=hf_token,
        load_in_4bit=load_in_4bit,
        max_image_pixels=max_image_pixels,
        load_checkpoint_file=not no_checkpoint_load,
    )

    routed = route_all(input_records, llm_results, router_cfg)
    routed, _ = apply_audit_sampling(routed, router_cfg.random_audit_rate, router_cfg.seed)

    bad_count = total_samples - len(routed)
    # TODO: build_stats có thể cần cập nhật thêm field T/I/MM sau
    stats = build_stats(routed, bad_count, total_samples)   # giữ tạm hàm cũ

    write_outputs(out_dir, routed, [], stats)

    logger.info("=== Round-2 Pipeline Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Round-2 Fine-grained Annotation Pipeline")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--min-record-id", type=int, default=None)
    parser.add_argument("--no-checkpoint-load", action="store_true")
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
