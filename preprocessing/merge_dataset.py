"""
merge_dataset.py
----------------
Gộp data Threads và Facebook thành:
  - merged_dataset/images/   : ảnh đổi tên theo global ID (không collision)
  - merged_dataset/data.json : JSON với các trường id, text, image_path, source

Nguồn dữ liệu:
  - Threads : data_collection/merged_threads/threads_data.json
              ảnh thực tế tại data_collection/merged_threads/images/
  - Facebook: data_collection/facebook_output/facebook_data.json
              ảnh thực tế tại data_collection/facebook_output/images/

Chạy incremental:
  - Đọc data.json hiện có để xác định global_id tiếp theo và high-water mark threads
  - Chỉ xử lý threads records chưa được merge (id > threads_watermark)
  - Không đụng vào facebook records đã merge
  - Ảnh mới đặt tên post{id:05d}.{ext} bắt đầu từ ID tiếp theo → không collision

Ghi chú:
  - Ảnh Threads được resolve bằng basename của image_path trong threads_data.json
  - image_path là relative path tính từ thư mục chạy script
  - Facebook text xóa nội dung bị lặp (artifact scraper)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import shutil

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ===================================================================
# CẤU HÌNH
# ===================================================================
THREADS_JSON = Path("data_collection/merged_threads/threads_data.json")
THREADS_IMG_DIR = Path("data_collection/merged_threads/images")

FB_JSON = Path("data_collection/facebook_output/facebook_data.json")
FB_IMG_BASE = Path("data_collection/facebook_output")

OUTPUT_DIR = Path("merged_dataset")
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_JSON = OUTPUT_DIR / "data.json"


# ===================================================================
# TEXT CLEANING — FACEBOOK
# Facebook text đã tương đối sạch, chỉ cần xử lý nội dung bị lặp
# ===================================================================

def clean_facebook_text(text: str) -> str | None:
    if not text or not text.strip():
        return None

    text = text.strip()

    # Xóa khối lặp theo dòng (scraper thường ghi caption 2 lần)
    lines = text.split("\n")
    n = len(lines)
    for length in range(n // 2, 0, -1):
        if lines[-length:] == lines[-2 * length: -length]:
            lines = lines[:-length]
            break
    text = "\n".join(lines).strip()

    # Xóa lặp nguyên dòng duy nhất: "câu A câu A"
    half = len(text) // 2
    if half > 0 and text[:half].strip() == text[half:].strip():
        text = text[:half].strip()

    return text if text else None


# ===================================================================
# MAIN
# ===================================================================

def merge_dataset() -> None:
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Bước 0: Load dữ liệu hiện có — incremental mode
    # ------------------------------------------------------------------
    existing_records: list[dict] = []
    global_id = 1
    threads_watermark = 0  # số threads records đã merge vào data.json

    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, encoding="utf-8") as f:
                existing_records = json.load(f)
            if existing_records:
                global_id = max(r["id"] for r in existing_records) + 1
            threads_watermark = sum(
                1 for r in existing_records if r.get("source") == "threads"
            )
            print(
                f"[INCREMENTAL] Đã có {len(existing_records)} records "
                f"(threads={threads_watermark}), tiếp tục từ ID={global_id}"
            )
        except Exception as exc:
            print(f"[WARN] Không đọc được data.json cũ ({exc}), bắt đầu từ đầu")
            existing_records = []
            global_id = 1
            threads_watermark = 0

    new_records: list[dict] = []
    stats = {"threads_ok": 0, "threads_skip": 0, "threads_skip_old": 0}

    # ------------------------------------------------------------------
    # Nguồn: Threads mới (chưa merge)
    # ------------------------------------------------------------------
    print("--- Threads ---")
    if not THREADS_JSON.exists():
        print(f"  [WARN] Không tìm thấy {THREADS_JSON}, bỏ qua Threads.")
    else:
        with open(THREADS_JSON, encoding="utf-8") as f:
            threads_records: list[dict] = json.load(f)

        for rec in threads_records:
            rec_id = rec.get("id", 0)

            # Bỏ qua records đã được merge vào data.json (theo thứ tự ID)
            if rec_id <= threads_watermark:
                stats["threads_skip_old"] += 1
                continue

            text = (rec.get("text") or "").strip()
            if not text:
                stats["threads_skip"] += 1
                continue

            # Resolve ảnh bằng basename của image_path trong threads_data.json
            raw_path = rec.get("image_path") or ""
            if not raw_path:
                stats["threads_skip"] += 1
                continue

            basename = Path(raw_path).name
            src_path = THREADS_IMG_DIR / basename
            if not src_path.exists():
                stats["threads_skip"] += 1
                continue

            suffix = src_path.suffix.lower()
            new_name = f"post{global_id:05d}{suffix}"
            dest = OUTPUT_IMG_DIR / new_name
            if not dest.exists():
                shutil.copy2(src_path, dest)

            new_records.append({
                "id": global_id,
                "text": text,
                "image_path": f"{OUTPUT_IMG_DIR}/{new_name}".replace("\\", "/"),
                "source": "threads",
            })
            global_id += 1
            stats["threads_ok"] += 1

    print(f"  Threads records da co (skip): {stats['threads_skip_old']}")
    print(f"  Threads moi them            : {stats['threads_ok']}")
    print(f"  Threads bo qua (loi)        : {stats['threads_skip']}")

    # ------------------------------------------------------------------
    # Lưu JSON — append records mới vào sau records cũ
    # ------------------------------------------------------------------
    all_records = existing_records + new_records
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n=== KET QUA ===")
    print(f"Records cu       : {len(existing_records)}")
    print(f"Records moi them : {len(new_records)}")
    print(f"Tong records     : {len(all_records)}")
    print(f"Anh da copy      : {len(list(OUTPUT_IMG_DIR.iterdir()))}")
    print(f"JSON luu tai     : {OUTPUT_JSON}")
    print(f"Anh luu tai      : {OUTPUT_IMG_DIR}")


if __name__ == "__main__":
    merge_dataset()
