"""
preprocess_text.py
------------------
Text preprocessing for dataset JSON:
  - Clean duplicated lines inside one sample text
  - Normalize whitespace
  - Drop records with null/empty text or image_path
  - Drop records with missing image file
  - Drop exact duplicate text (after cleaning)
  - Reassign continuous IDs

Run:
    python preprocessing/preprocess_text.py \
        --input data/final-data/data.json \
        --output data/final-data/data.clean.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

DEFAULT_INPUT = Path("data/final-data/data.json")
DEFAULT_OUTPUT = Path("data/final-data/data.clean.json")

RE_HANDLE = re.compile(r"^@[A-Za-z0-9_.]{2,32}$")
RE_USERNAME_LITERAL = re.compile(r"^username$", re.I)
RE_ASCII_HANDLE = re.compile(r"^[A-Za-z0-9_.]{4,32}$")
RE_REL_TIME = re.compile(
    r"^(?:\d{1,2}\s*(?:giờ|phút|ngày|tuần|tháng|năm)\s*(?:trước)?|\d{1,2}[:h]\d{0,2}|just\s*now|now)$",
    re.I,
)
RE_UI_TOKEN = re.compile(
    r"^(?:see\s*more|xem\s*thêm|ẩn\s*bớt|dịch|trả\s*lời|reply|like|thích|bình\s*luận|comment|chia\s*sẻ)$",
    re.I,
)


def normalize_spaces(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def try_collapse_full_repetition(text: str) -> str:
    """If text == unit repeated N times, keep only one unit."""
    s = text.strip()
    n = len(s)
    if n < 2:
        return s
    for k in range(1, n // 2 + 1):
        if n % k == 0:
            t = n // k
            unit = s[:k]
            if t >= 2 and unit * t == s:
                return unit.strip()
    return s


def remove_duplicated_lines(text: str) -> Tuple[str, bool]:
    """Keep first occurrence of each line, preserve order."""
    raw_lines = text.splitlines()
    seen = set()
    out = []
    changed = False

    for line in raw_lines:
        line_norm = normalize_spaces(line)
        if not line_norm:
            continue
        key = line_norm.lower()
        if key in seen:
            changed = True
            continue
        seen.add(key)
        out.append(line_norm)

    cleaned = "\n".join(out).strip()
    return cleaned, changed


def strip_crawl_artifacts(text: str) -> Tuple[str, bool]:
    """
    Remove high-confidence crawl artifacts from top lines:
      - username / @handle
      - relative time line
      - UI tokens (See more, Xem thêm, ...)
      - username <-> time pair in first two lines
    """
    lines = [normalize_spaces(x) for x in (text or "").splitlines() if normalize_spaces(x)]
    if not lines:
        return "", False

    drop_idx: set[int] = set()

    first = lines[0]
    second = lines[1] if len(lines) > 1 else ""

    if (
        RE_USERNAME_LITERAL.match(first)
        or RE_HANDLE.match(first)
        or RE_REL_TIME.match(first)
        or RE_UI_TOKEN.match(first)
    ):
        drop_idx.add(0)

    if len(lines) >= 2 and RE_ASCII_HANDLE.match(first) and RE_REL_TIME.match(second):
        drop_idx.update({0, 1})
    if len(lines) >= 2 and RE_REL_TIME.match(first) and RE_ASCII_HANDLE.match(second):
        drop_idx.update({0, 1})

    # Also drop literal "username" if it appears near the top block.
    for i, ln in enumerate(lines[:4]):
        if RE_USERNAME_LITERAL.match(ln):
            drop_idx.add(i)

    if not drop_idx:
        return "\n".join(lines).strip(), False

    cleaned = "\n".join(ln for i, ln in enumerate(lines) if i not in drop_idx).strip()
    if not cleaned:
        cleaned = "\n".join(lines).strip()
        return cleaned, False
    return cleaned, True


def preprocess_text_value(text: str) -> Tuple[str, bool]:
    """Main text cleaning routine."""
    before = text or ""
    x = normalize_spaces(before)
    x, crawl_changed = strip_crawl_artifacts(x)
    x = try_collapse_full_repetition(x)
    x, line_changed = remove_duplicated_lines(x)
    x = normalize_spaces(x)
    changed = (x != before.strip()) or line_changed or crawl_changed
    return x, changed


def check_null(records: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for rec in records:
        text = (rec.get("text") or "").strip()
        img = (rec.get("image_path") or "").strip()
        if not text or not img:
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped


def check_image_exists(records: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for rec in records:
        img_path = rec.get("image_path", "")
        if img_path and Path(img_path).exists():
            kept.append(rec)
        else:
            dropped.append(rec)
    return kept, dropped


def dedup_exact(records: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = set()
    kept, dropped = [], []
    for rec in records:
        key = re.sub(r"\s+", " ", (rec.get("text") or "").strip().lower())
        if key in seen:
            dropped.append(rec)
        else:
            seen.add(key)
            kept.append(rec)
    return kept, dropped


def reassign_ids(records: list[dict]) -> list[dict]:
    for new_id, rec in enumerate(records, start=1):
        rec["id"] = new_id
    return records


def preprocess(
    input_path: Path,
    output_path: Path,
    clean_ocr: bool = False,
    dry_run: bool = False,
) -> None:
    with open(input_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    total_start = len(records)

    # 1) text clean
    text_changed = 0
    ocr_changed = 0
    for rec in records:
        clean_text, changed = preprocess_text_value(rec.get("text") or "")
        rec["text"] = clean_text
        if changed:
            text_changed += 1

        if clean_ocr:
            clean_ocr_text, changed_ocr = preprocess_text_value(rec.get("ocr_text") or "")
            rec["ocr_text"] = clean_ocr_text
            if changed_ocr:
                ocr_changed += 1

    # 2) null check
    records, dropped_null = check_null(records)

    # 3) image exists
    records, dropped_noimg = check_image_exists(records)

    # 4) exact dedup by cleaned text
    records, dropped_dup = dedup_exact(records)

    # 5) reassign ids
    records = reassign_ids(records)

    print(f"Input              : {input_path}")
    print(f"Total start        : {total_start}")
    print(f"Text cleaned       : {text_changed}")
    if clean_ocr:
        print(f"OCR cleaned        : {ocr_changed}")
    print(f"Drop null text/img : {len(dropped_null)}")
    print(f"Drop missing image : {len(dropped_noimg)}")
    print(f"Drop exact dup     : {len(dropped_dup)}")
    print(f"Total final        : {len(records)}")

    if dry_run:
        print("[DRY-RUN] no file written")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess text in dataset JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-ocr", action="store_true", help="Apply same cleaning to ocr_text")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path=args.input,
        output_path=args.output,
        clean_ocr=args.clean_ocr,
        dry_run=args.dry_run,
    )
