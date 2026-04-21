"""
IAA (Inter-Annotator Agreement) module for Round-1 annotation.

Computes pairwise and 3-way agreement metrics among two human annotators
(Label Studio exports) and the LLM stage-1 labels, then saves conflict
records for human review.

Annotator sources
-----------------
- ann1  : NLe_label_v4.json   (Label Studio export, annotator 1)
- ann2  : lableling_nghan_v4.json (Label Studio export, annotator 2)
- llm   : round1_all.json     (LLM pipeline output)

Matching key
------------
Records are matched by normalised post **text** (strip-only normalisation).
This is required because human inner_id values (1-50) do not correspond to
LLM post IDs, which are the original dataset IDs (1, 3, 7, ..., 1105).

Label encoding
--------------
- Human "Sarcastic"     -> 1
- Human "Not Sarcastic" -> 0
- LLM   label_llm1 == 1 -> 1
- LLM   label_llm1 == 0 -> 0
- LLM   label_llm1 == "INVALID" or None -> excluded from metrics

Usage
-----
    python iaa_agreement.py [--ann1 PATH] [--ann2 PATH] [--llm PATH]
                            [--ann1-name STR] [--ann2-name STR]
                            [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _extract_human_label(annotations: list) -> Optional[int]:
    """
    Return 0 or 1 from the first non-cancelled Label Studio annotation.
    Returns None when all annotations are cancelled or carry no valid choice.
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


def load_labelstudio_file(path: Path) -> dict[str, dict]:
    """
    Load a Label Studio JSON export and return a dict keyed by normalised text.

    Each value contains:
        label    : int (0 or 1) or None
        inner_id : int
        task_id  : int
        text     : str (raw)
    """
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    out: dict[str, dict] = {}
    for task in tasks:
        raw_text: str = task.get("data", {}).get("text", "")
        key = raw_text.strip()
        if not key:
            continue
        label = _extract_human_label(task.get("annotations", []))
        if key in out:
            warnings.warn(
                f"Duplicate text found in {path.name} (inner_id={task.get('inner_id')}). "
                "Keeping first occurrence.",
                stacklevel=2,
            )
            continue
        out[key] = {
            "label": label,
            "inner_id": task.get("inner_id"),
            "task_id": task.get("id"),
            "text": raw_text,
        }
    return out


def load_llm_file(path: Path) -> dict[str, dict]:
    """
    Load round1_all.json and return a dict keyed by normalised text.

    Each value contains:
        label      : int (0 or 1) or None  (None for INVALID or missing)
        id         : int (original post ID)
        text       : str (raw)
        difficulty : str or None
        route_reason: str or None
        notes      : str
        reasoning  : dict or None
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    out: dict[str, dict] = {}
    for r in records:
        raw_text: str = r.get("text", "")
        key = raw_text.strip()
        if not key:
            continue
        if key in out:
            warnings.warn(
                f"Duplicate text found in {path.name} (id={r.get('id')}). "
                "Keeping first occurrence.",
                stacklevel=2,
            )
            continue
        raw_label = r.get("label_llm1")
        label: Optional[int] = None
        if raw_label not in ("INVALID", None):
            label = int(raw_label)
        out[key] = {
            "label": label,
            "id": r.get("id"),
            "text": raw_text,
            "difficulty": r.get("difficulty"),
            "route_reason": r.get("route_reason"),
            "notes": r.get("notes", ""),
            "reasoning": r.get("reasoning"),
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


def fleiss_kappa(labels_matrix: list[list[int]], n_categories: int = 2) -> float:
    """
    Compute Fleiss' Kappa for a fixed number of raters per item.

    Parameters
    ----------
    labels_matrix : list of lists, shape (n_items, n_raters)
        Each inner list contains the integer labels assigned by each rater.
    n_categories  : int
        Number of distinct label categories (default 2: 0 and 1).

    Returns
    -------
    float — Fleiss' Kappa, or nan if computation is undefined.
    """
    n_items = len(labels_matrix)
    if n_items == 0:
        return float("nan")

    n_raters = len(labels_matrix[0])
    if n_raters < 2:
        return float("nan")

    categories = list(range(n_categories))

    # n_ij[i][j] = number of raters who assigned category j to item i
    n_ij = [[row.count(j) for j in categories] for row in labels_matrix]

    # P_i = extent of agreement for item i
    P_i = []
    for row in n_ij:
        numerator = sum(x * (x - 1) for x in row)
        denominator = n_raters * (n_raters - 1)
        P_i.append(numerator / denominator)

    P_bar = sum(P_i) / n_items

    # p_j = overall proportion of ratings in category j
    total_ratings = n_items * n_raters
    p_j = [sum(n_ij[i][j] for i in range(n_items)) / total_ratings for j in categories]

    P_e = sum(p ** 2 for p in p_j)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def _interpret_kappa(k: float) -> str:
    """Return a standard qualitative label for a kappa value."""
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
# Core IAA computation
# ---------------------------------------------------------------------------

def _pairwise_stats(
    y1: list[int],
    y2: list[int],
    name1: str,
    name2: str,
) -> dict:
    n = len(y1)
    if n == 0:
        return {"annotators": [name1, name2], "n_pairs": 0}

    n_agree = sum(a == b for a, b in zip(y1, y2))
    kappa = cohen_kappa(y1, y2)

    tp = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 0)

    return {
        "annotators": [name1, name2],
        "n_pairs": n,
        "n_agree": n_agree,
        "n_conflict": n - n_agree,
        "agreement_pct": round(n_agree / n * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": _interpret_kappa(kappa),
        "confusion_matrix": {
            f"TP_{name1}_1_{name2}_1": tp,
            f"TN_{name1}_0_{name2}_0": tn,
            f"FP_{name1}_0_{name2}_1": fp,
            f"FN_{name1}_1_{name2}_0": fn,
        },
    }


def compute_iaa(
    ann1: dict[str, dict],
    ann2: dict[str, dict],
    llm: dict[str, dict],
    name1: str = "ann1",
    name2: str = "ann2",
) -> tuple[dict, list[dict]]:
    """
    Match all three annotator dicts by text key and compute agreement metrics.

    Returns
    -------
    report    : dict with pairwise and 3-way stats plus coverage counts
    conflicts : list of records where ann1 and ann2 disagree
    """
    all_texts = set(ann1) | set(ann2) | set(llm)

    # Collect per-text labels; track skips
    triples: list[dict] = []   # texts present in all 3 with valid labels
    skipped: dict[str, list[str]] = {
        "missing_ann1": [],
        "missing_ann2": [],
        "missing_llm": [],
        "invalid_ann1": [],
        "invalid_ann2": [],
        "invalid_llm": [],
    }

    for text_key in sorted(all_texts):
        a1 = ann1.get(text_key)
        a2 = ann2.get(text_key)
        lm = llm.get(text_key)

        if a1 is None:
            skipped["missing_ann1"].append(text_key[:60])
            continue
        if a2 is None:
            skipped["missing_ann2"].append(text_key[:60])
            continue
        if lm is None:
            skipped["missing_llm"].append(text_key[:60])
            continue

        l1 = a1["label"]
        l2 = a2["label"]
        ll = lm["label"]

        if l1 is None:
            skipped["invalid_ann1"].append(text_key[:60])
        if l2 is None:
            skipped["invalid_ann2"].append(text_key[:60])
        if ll is None:
            skipped["invalid_llm"].append(text_key[:60])

        if l1 is None or l2 is None or ll is None:
            continue

        triples.append({
            "text": text_key,
            "label_ann1": l1,
            "label_ann2": l2,
            "label_llm": ll,
            "inner_id_ann1": a1.get("inner_id"),
            "inner_id_ann2": a2.get("inner_id"),
            "llm_id": lm.get("id"),
            "llm_difficulty": lm.get("difficulty"),
            "llm_route_reason": lm.get("route_reason"),
            "llm_notes": lm.get("notes", ""),
            "llm_reasoning_verdict": (lm.get("reasoning") or {}).get("verdict", ""),
        })

    n = len(triples)

    # Pairwise sequences
    y1 = [t["label_ann1"] for t in triples]
    y2 = [t["label_ann2"] for t in triples]
    yl = [t["label_llm"] for t in triples]

    # 3-way full agreement
    n_all_agree = sum(1 for t in triples if t["label_ann1"] == t["label_ann2"] == t["label_llm"])
    fk = fleiss_kappa([[t["label_ann1"], t["label_ann2"], t["label_llm"]] for t in triples])

    # Conflicts: ann1 != ann2
    conflicts = [
        {
            "text": t["text"],
            "inner_id_ann1": t["inner_id_ann1"],
            "inner_id_ann2": t["inner_id_ann2"],
            "llm_id": t["llm_id"],
            f"label_{name1}": t["label_ann1"],
            f"label_{name1}_str": "sarcastic" if t["label_ann1"] == 1 else "not_sarcastic",
            f"label_{name2}": t["label_ann2"],
            f"label_{name2}_str": "sarcastic" if t["label_ann2"] == 1 else "not_sarcastic",
            "label_llm": t["label_llm"],
            "label_llm_str": "sarcastic" if t["label_llm"] == 1 else "not_sarcastic",
            "llm_difficulty": t["llm_difficulty"],
            "llm_route_reason": t["llm_route_reason"],
            "llm_reasoning_verdict": t["llm_reasoning_verdict"],
        }
        for t in triples
        if t["label_ann1"] != t["label_ann2"]
    ]

    # Coverage / skip summary
    coverage = {
        "n_texts_union": len(all_texts),
        "n_texts_all_three_present": sum(
            1 for txt in all_texts if txt in ann1 and txt in ann2 and txt in llm
        ),
        "n_valid_triples": n,
        "n_skipped_missing_ann1": len(skipped["missing_ann1"]),
        "n_skipped_missing_ann2": len(skipped["missing_ann2"]),
        "n_skipped_missing_llm": len(skipped["missing_llm"]),
        "n_skipped_invalid_ann1": len(skipped["invalid_ann1"]),
        "n_skipped_invalid_ann2": len(skipped["invalid_ann2"]),
        "n_skipped_invalid_llm": len(skipped["invalid_llm"]),
    }

    report = {
        "coverage": coverage,
        "pairwise": {
            f"{name1}_vs_{name2}": _pairwise_stats(y1, y2, name1, name2),
            f"{name1}_vs_llm": _pairwise_stats(y1, yl, name1, "llm"),
            f"{name2}_vs_llm": _pairwise_stats(y2, yl, name2, "llm"),
        },
        "three_way": {
            "n_valid_triples": n,
            "n_all_agree": n_all_agree,
            "n_any_conflict": n - n_all_agree,
            "all_agree_pct": round(n_all_agree / n * 100, 2) if n else float("nan"),
            "fleiss_kappa": round(fk, 4),
            "fleiss_kappa_interpretation": _interpret_kappa(fk),
            "label_distribution": {
                name1: {
                    "sarcastic": sum(1 for v in y1 if v == 1),
                    "not_sarcastic": sum(1 for v in y1 if v == 0),
                },
                name2: {
                    "sarcastic": sum(1 for v in y2 if v == 1),
                    "not_sarcastic": sum(1 for v in y2 if v == 0),
                },
                "llm": {
                    "sarcastic": sum(1 for v in yl if v == 1),
                    "not_sarcastic": sum(1 for v in yl if v == 0),
                },
            },
        },
        "human_conflict_records": conflicts,
    }

    return report, conflicts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--ann1",
        default=str(here / "NLe_label_v4.json"),
        help="Path to annotator-1 Label Studio JSON export",
    )
    p.add_argument(
        "--ann2",
        default=str(here / "lableling_nghan_v4.json"),
        help="Path to annotator-2 Label Studio JSON export",
    )
    p.add_argument(
        "--llm",
        default=str(here / "round1_all.json"),
        help="Path to LLM pipeline output JSON (round1_all.json)",
    )
    p.add_argument(
        "--ann1-name",
        default="NLe",
        help="Display name for annotator 1 (used in report keys)",
    )
    p.add_argument(
        "--ann2-name",
        default="Nghan",
        help="Display name for annotator 2 (used in report keys)",
    )
    p.add_argument(
        "--out-dir",
        default=str(here),
        help="Directory where iaa_report.json and conflict_records.json are written",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    ann1_path = Path(args.ann1)
    ann2_path = Path(args.ann2)
    llm_path = Path(args.llm)
    out_dir = Path(args.out_dir)
    name1: str = args.ann1_name
    name2: str = args.ann2_name
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "iaa_report.json"
    conflict_path = out_dir / "conflict_records.json"

    print(f"Loading annotator-1  : {ann1_path}  ({name1})")
    ann1 = load_labelstudio_file(ann1_path)
    print(f"  -> {len(ann1)} records")

    print(f"Loading annotator-2  : {ann2_path}  ({name2})")
    ann2 = load_labelstudio_file(ann2_path)
    print(f"  -> {len(ann2)} records")

    print(f"Loading LLM labels   : {llm_path}")
    llm = load_llm_file(llm_path)
    print(f"  -> {len(llm)} records")

    report, conflicts = compute_iaa(ann1, ann2, llm, name1=name1, name2=name2)

    cov = report["coverage"]
    three = report["three_way"]
    pw = report["pairwise"]

    if cov["n_valid_triples"] == 0:
        print(
            "\nERROR: No valid triples found (all 3 annotators with non-null labels).\n"
            "Verify that all three files label the same posts (matched by text)."
        )
        sys.exit(1)

    # Strip inline conflict list from report before saving (saved separately)
    report_to_save = {k: v for k, v in report.items() if k != "human_conflict_records"}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, ensure_ascii=False, indent=2)

    with open(conflict_path, "w", encoding="utf-8") as f:
        json.dump(conflicts, f, ensure_ascii=False, indent=2)

    # --- console summary ---
    sep = "=" * 60
    pw12 = pw[f"{name1}_vs_{name2}"]
    pw1l = pw[f"{name1}_vs_llm"]
    pw2l = pw[f"{name2}_vs_llm"]

    print(f"\n{sep}")
    print(f"  Coverage")
    print(f"    Union of texts     : {cov['n_texts_union']}")
    print(f"    Present in all 3   : {cov['n_texts_all_three_present']}")
    print(f"    Valid triples      : {cov['n_valid_triples']}")
    if cov["n_skipped_invalid_llm"]:
        print(f"    Skipped (LLM INVALID) : {cov['n_skipped_invalid_llm']}")

    print(f"\n  Pairwise IAA")
    print(f"    {name1} vs {name2:<12} : {pw12['agreement_pct']} %  "
          f"kappa={pw12['cohen_kappa']}  ({pw12['kappa_interpretation']})")
    print(f"    {name1} vs LLM{'':<10} : {pw1l['agreement_pct']} %  "
          f"kappa={pw1l['cohen_kappa']}  ({pw1l['kappa_interpretation']})")
    print(f"    {name2} vs LLM{'':<10} : {pw2l['agreement_pct']} %  "
          f"kappa={pw2l['cohen_kappa']}  ({pw2l['kappa_interpretation']})")

    print(f"\n  3-Way Agreement (Fleiss' Kappa)")
    print(f"    All 3 agree        : {three['n_all_agree']} / {three['n_valid_triples']}"
          f"  ({three['all_agree_pct']} %)")
    print(f"    Fleiss' Kappa      : {three['fleiss_kappa']}  ({three['fleiss_kappa_interpretation']})")

    print(f"\n  Human Conflicts ({name1} != {name2})  : {len(conflicts)}")
    print(f"{sep}")
    print(f"\nReport    -> {report_path}")
    print(f"Conflicts -> {conflict_path}  ({len(conflicts)} records)")


if __name__ == "__main__":
    main()
