"""
build_part2_dataset.py
----------------------
Xu ly du lieu crawl bo sung trong data_collection/part-2 thanh:
  - data/raw-data/data-p2.json
  - anh moi copy vao data/images/ voi ten post{id:05d}.{ext}

Dau ra giong schema data/raw-data/raw_data.json:
  - id
  - text
  - image_path
  - source
  - ocr_text

Quy uoc:
  - ID moi tiep noi max ID hien co trong raw_data.json
  - Chi lay image dau tien cho moi record de khop schema hien tai
  - Threads dedup theo post_link trong part-2
  - OCR tam thoi de rong
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DATA_PATH = Path("data/raw-data/raw_data.json")
OUTPUT_JSON_PATH = Path("data/raw-data/data-p2.json")
OUTPUT_IMG_DIR = Path("data/images")

FB_JSON_PATH = Path("data_collection/part-2/facebook_p2/data.json")
FB_BASE_DIR = FB_JSON_PATH.parent

THREADS_BASE_DIR = Path("data_collection/part-2/thread_p2")


def clean_facebook_text(text: str) -> str | None:
    if not text or not text.strip():
        return None

    text = text.strip()

    lines = text.split("\n")
    n = len(lines)
    for length in range(n // 2, 0, -1):
        if lines[-length:] == lines[-2 * length : -length]:
            lines = lines[:-length]
            break
    text = "\n".join(lines).strip()

    half = len(text) // 2
    if half > 0 and text[:half].strip() == text[half:].strip():
        text = text[:half].strip()

    return text if text else None


def clean_threads_text(text: str) -> str | None:
    if not text or not text.strip():
        return None

    text = str(text).strip()

    text = re.sub(
        r"^((thịnh hành|thread đầu tiên)[\s\n]*)+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if "tiết lộ nội dung" in text.lower():
        return None

    parts = re.split(r"đang trả lời @[\w._]+\s*", text, flags=re.IGNORECASE, maxsplit=1)
    if len(parts) == 2:
        before, after = parts[0].strip(), parts[1].strip()
        text = before if before else after

    text = re.sub(r"@[\w._]+", "username", text)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    bad_indices: set[int] = set()

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if "đã đăng lại" in line_lower:
            bad_indices.add(i)
            if i + 1 < len(lines):
                bad_indices.add(i + 1)
            continue

        if line_lower == "dịch":
            bad_indices.add(i)
            continue

        if line_lower.endswith(" dịch"):
            lines[i] = line[:-5].strip()
            line = lines[i]
            line_lower = line.lower()

        time_pattern = r"^\d+\s+(giây|phút|giờ|ngày|tuần|tháng|năm)"
        date_pattern = r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"

        if (
            re.match(time_pattern, line_lower)
            or re.match(date_pattern, line)
            or "vừa xong" in line_lower
        ):
            bad_indices.add(i)
            if i > 0:
                bad_indices.add(i - 1)
            if i > 1 and len(lines[i - 2].split()) <= 5:
                bad_indices.add(i - 2)
            continue

        if re.match(r"^[\d.,kKmM]+$", line) or line == "/":
            bad_indices.add(i)

    lines = [ln for i, ln in enumerate(lines) if i not in bad_indices]

    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^#\d+:\s*$", lines[i]) and i + 1 < len(lines):
            i += 1
            continue
        cleaned.append(lines[i])
        i += 1

    deduped: list[str] = []
    for ln in cleaned:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)

    n = len(deduped)
    for length in range(n // 2, 0, -1):
        if deduped[-length:] == deduped[-2 * length : -length]:
            deduped = deduped[:-length]
            break

    result = "\n".join(deduped).strip()
    return result if result else None


def next_start_id() -> int:
    with open(RAW_DATA_PATH, encoding="utf-8") as f:
        raw_records: list[dict] = json.load(f)
    return (max(rec["id"] for rec in raw_records) + 1) if raw_records else 1


def copy_image(src_path: Path, rec_id: int) -> str:
    suffix = src_path.suffix.lower()
    dest_name = f"post{rec_id:05d}{suffix}"
    dest_path = OUTPUT_IMG_DIR / dest_name
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return f"data/images/{dest_name}"


def build_threads_records(start_id: int) -> tuple[list[dict], int, dict[str, int]]:
    records: list[dict] = []
    next_id = start_id
    stats = {"raw": 0, "duplicate_link": 0, "missing_image": 0, "empty_text": 0, "kept": 0}
    seen_links: set[str] = set()

    for session_dir in sorted(THREADS_BASE_DIR.iterdir()):
        if not session_dir.is_dir():
            continue

        json_files = sorted(session_dir.glob("*.json"))
        if not json_files:
            continue

        with open(json_files[0], encoding="utf-8") as f:
            session_records = json.load(f)

        if not isinstance(session_records, list):
            session_records = [session_records]

        for record in session_records:
            stats["raw"] += 1

            post_link = record.get("post_link", "")
            if post_link:
                if post_link in seen_links:
                    stats["duplicate_link"] += 1
                    continue
                seen_links.add(post_link)

            image_local = record.get("image_local") or []
            if isinstance(image_local, str):
                image_local = [image_local]

            if not image_local:
                stats["missing_image"] += 1
                continue

            first_image = session_dir / "images" / Path(image_local[0]).name
            if not first_image.exists():
                stats["missing_image"] += 1
                continue

            cleaned_text = clean_threads_text(record.get("text", ""))
            if not cleaned_text:
                stats["empty_text"] += 1
                continue

            new_record = {
                "id": next_id,
                "text": cleaned_text,
                "image_path": copy_image(first_image, next_id),
                "source": "threads",
                "ocr_text": "",
            }
            records.append(new_record)
            next_id += 1
            stats["kept"] += 1

    return records, next_id, stats


def build_facebook_records(start_id: int) -> tuple[list[dict], int, dict[str, int]]:
    with open(FB_JSON_PATH, encoding="utf-8") as f:
        fb_records: list[dict] = json.load(f)

    records: list[dict] = []
    next_id = start_id
    stats = {"raw": 0, "missing_image": 0, "empty_text": 0, "kept": 0}

    for record in fb_records:
        stats["raw"] += 1

        src_path = FB_BASE_DIR / (record.get("image_path") or "")
        if not src_path.exists():
            stats["missing_image"] += 1
            continue

        cleaned_text = clean_facebook_text(record.get("text", ""))
        if not cleaned_text:
            stats["empty_text"] += 1
            continue

        new_record = {
            "id": next_id,
            "text": cleaned_text,
            "image_path": copy_image(src_path, next_id),
            "source": "facebook",
            "ocr_text": "",
        }
        records.append(new_record)
        next_id += 1
        stats["kept"] += 1

    return records, next_id, stats


def main() -> None:
    start_id = next_start_id()
    print(f"Bat dau gan ID tu: {start_id}")

    thread_records, next_id, thread_stats = build_threads_records(start_id)
    fb_records, next_id, fb_stats = build_facebook_records(next_id)

    all_records = thread_records + fb_records

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print("\n=== THREADS ===")
    for key, value in thread_stats.items():
        print(f"{key:>14}: {value}")

    print("\n=== FACEBOOK ===")
    for key, value in fb_stats.items():
        print(f"{key:>14}: {value}")

    print("\n=== OUTPUT ===")
    print(f"records moi    : {len(all_records)}")
    print(f"threads        : {len(thread_records)}")
    print(f"facebook       : {len(fb_records)}")
    print(f"id cuoi        : {next_id - 1}")
    print(f"json           : {OUTPUT_JSON_PATH}")
    print(f"images folder  : {OUTPUT_IMG_DIR}")


if __name__ == "__main__":
    main()
