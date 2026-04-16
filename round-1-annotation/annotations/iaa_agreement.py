"""
IAA (Inter-Annotator Agreement) module for Round-1 annotation.

Computes agreement metrics between human labels (Label Studio export) and
LLM stage-1 labels, then saves conflict records for review.

Mapping key
-----------
- human_label_samples.json : `inner_id` field on each task
- llm_label_samples.json   : `id` field on each record
Match condition: human.inner_id == llm.id

Label encoding
--------------
- Human "Sarcastic"     -> 1
- Human "Not Sarcastic" -> 0
- LLM   label_llm1 == 1 -> 1
- LLM   label_llm1 == 0 -> 0
- LLM   label_llm1 == "INVALID" -> excluded from metrics

Usage
-----
    python iaa_agreement.py [--llm PATH] [--human PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_llm_labels(path: Path) -> dict[int, dict]:
    """Return dict keyed by id from llm_label_samples.json."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    out: dict[int, dict] = {}
    for r in records:
        rid = r.get("id")
        if rid is not None:
            out[int(rid)] = r
    return out


def _extract_human_label(annotations: list) -> Optional[int]:
    """
    Return 0 or 1 from the first non-cancelled Label Studio annotation.
    Returns None when all annotations are cancelled or have no valid choice.
    """
    for ann in annotations:
        if ann.get("was_cancelled"):
            continue
        for result in ann.get("result", []):
            choices = result.get("value", {}).get("choices", [])
            for choice in choices:
                choice_norm = choice.strip().lower()
                if "not" in choice_norm:
                    return 0
                if "sarcastic" in choice_norm:
                    return 1
    return None


def load_human_labels(path: Path) -> dict[int, dict]:
    """Return dict keyed by inner_id from human_label_samples.json."""
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)
    out: dict[int, dict] = {}
    for task in tasks:
        inner_id = task.get("inner_id")
        if inner_id is None:
            continue
        label = _extract_human_label(task.get("annotations", []))
        out[int(inner_id)] = {
            "label_human": label,
            "text": task.get("data", {}).get("text", ""),
            "task_id": task.get("id"),
            "inner_id": int(inner_id),
        }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """
    Compute Cohen's Kappa for two binary label sequences of equal length.
    Returns nan for empty input; returns 1.0 when expected agreement == 1.
    """
    n = len(y1)
    if n == 0:
        return float("nan")

    po = sum(a == b for a, b in zip(y1, y2)) / n

    p1_neg = sum(1 for x in y1 if x == 0) / n
    p1_pos = sum(1 for x in y1 if x == 1) / n
    p2_neg = sum(1 for x in y2 if x == 0) / n
    p2_pos = sum(1 for x in y2 if x == 1) / n

    pe = p1_neg * p2_neg + p1_pos * p2_pos
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_iaa(
    llm_labels: dict[int, dict],
    human_labels: dict[int, dict],
) -> tuple[Optional[dict], list[dict], list[int], list[int]]:
    """
    Match human and LLM records, compute agreement metrics, collect conflicts.

    Returns
    -------
    stats              : dict of computed metrics (None if no valid pairs)
    conflicts          : records where human label != LLM label
    skipped_no_match   : inner_ids without a corresponding LLM record
    skipped_invalid    : inner_ids excluded due to INVALID or missing label
    """
    pairs: list[dict] = []
    conflicts: list[dict] = []
    skipped_no_match: list[int] = []
    skipped_invalid: list[int] = []

    for inner_id, human in sorted(human_labels.items()):
        llm = llm_labels.get(inner_id)
        if llm is None:
            skipped_no_match.append(inner_id)
            continue

        h_label = human["label_human"]
        l_raw = llm.get("label_llm1")

        if h_label is None or l_raw == "INVALID" or l_raw is None:
            skipped_invalid.append(inner_id)
            continue

        l_label = int(l_raw)
        pairs.append({"id": inner_id, "human": h_label, "llm": l_label})

        if h_label != l_label:
            conflicts.append({
                "id": inner_id,
                "task_id": human["task_id"],
                "text": human["text"],
                "label_human": h_label,
                "label_human_str": "sarcastic" if h_label == 1 else "not_sarcastic",
                "label_llm1": l_label,
                "label_llm1_str": "sarcastic" if l_label == 1 else "not_sarcastic",
                "difficulty": llm.get("difficulty"),
                "route_reason": llm.get("route_reason"),
                "notes": llm.get("notes", ""),
                "reasoning_verdict": (llm.get("reasoning") or {}).get("verdict", ""),
            })

    if not pairs:
        return None, conflicts, skipped_no_match, skipped_invalid

    y_human = [p["human"] for p in pairs]
    y_llm = [p["llm"] for p in pairs]

    n = len(pairs)
    n_agree = sum(h == l for h, l in zip(y_human, y_llm))

    # Confusion matrix (reference = human label)
    tp = sum(1 for h, l in zip(y_human, y_llm) if h == 1 and l == 1)
    tn = sum(1 for h, l in zip(y_human, y_llm) if h == 0 and l == 0)
    fp = sum(1 for h, l in zip(y_human, y_llm) if h == 0 and l == 1)
    fn = sum(1 for h, l in zip(y_human, y_llm) if h == 1 and l == 0)

    kappa = cohen_kappa(y_human, y_llm)

    stats = {
        "n_total_human_tasks": len(human_labels),
        "n_total_llm_records": len(llm_labels),
        "n_matched": n,
        "n_agree": n_agree,
        "n_conflict": len(conflicts),
        "n_skipped_no_match": len(skipped_no_match),
        "n_skipped_invalid": len(skipped_invalid),
        "agreement_pct": round(n_agree / n * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": _interpret_kappa(kappa),
        "confusion_matrix": {
            "TP_human_sarcastic_llm_sarcastic": tp,
            "TN_human_not_sarcastic_llm_not_sarcastic": tn,
            "FP_human_not_sarcastic_llm_sarcastic": fp,
            "FN_human_sarcastic_llm_not_sarcastic": fn,
        },
        "label_distribution": {
            "human": {
                "sarcastic": sum(1 for h in y_human if h == 1),
                "not_sarcastic": sum(1 for h in y_human if h == 0),
            },
            "llm": {
                "sarcastic": sum(1 for l in y_llm if l == 1),
                "not_sarcastic": sum(1 for l in y_llm if l == 0),
            },
        },
    }
    return stats, conflicts, skipped_no_match, skipped_invalid


def _interpret_kappa(k: float) -> str:
    """Return a standard qualitative label for a Cohen's Kappa value."""
    if k != k:  # nan check
        return "undefined"
    if k < 0:
        return "poor (less than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    here = Path(__file__).parent
    p.add_argument("--llm", default=str(here / "llm_label_samples.json"),
                   help="Path to LLM label JSON file")
    p.add_argument("--human", default=str(here / "human_label_samples.json"),
                   help="Path to human label JSON file (Label Studio export)")
    p.add_argument("--out-dir", default=str(here),
                   help="Directory where iaa_report.json and conflict_records.jsonl are written")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    llm_path = Path(args.llm)
    human_path = Path(args.human)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "iaa_report.json"
    conflict_path = out_dir / "conflict_records.jsonl"

    print(f"Loading LLM labels   : {llm_path}")
    llm_labels = load_llm_labels(llm_path)
    print(f"  -> {len(llm_labels)} records")

    print(f"Loading human labels : {human_path}")
    human_labels = load_human_labels(human_path)
    print(f"  -> {len(human_labels)} tasks")

    stats, conflicts, skipped_no_match, skipped_invalid = compute_iaa(llm_labels, human_labels)

    if stats is None:
        print(
            "\nERROR: No matched pairs found.\n"
            "Check that 'inner_id' in the human file matches 'id' in the LLM file."
        )
        sys.exit(1)

    # --- write report ---
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # --- write conflicts ---
    with open(conflict_path, "w", encoding="utf-8") as f:
        for record in conflicts:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- console summary ---
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Matched pairs      : {stats['n_matched']}")
    print(f"  Agreement          : {stats['agreement_pct']} %")
    print(f"  Cohen's Kappa      : {stats['cohen_kappa']}  ({stats['kappa_interpretation']})")
    print(f"  Conflicts          : {stats['n_conflict']}")
    if skipped_no_match:
        print(f"  Skipped (no match) : {len(skipped_no_match)}")
    if skipped_invalid:
        print(f"  Skipped (INVALID)  : {len(skipped_invalid)}")
    print(f"{sep}")
    print(f"\nReport   -> {report_path}")
    print(f"Conflicts-> {conflict_path}  ({len(conflicts)} records)")


if __name__ == "__main__":
    main()
