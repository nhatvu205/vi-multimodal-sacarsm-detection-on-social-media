"""
merge_dataset.py
----------------
Gộp data Threads và Facebook thành:
  - merged_dataset/images/  : ảnh đổi tên theo global ID (không collision)
  - merged_dataset/data.json : JSON với các trường id, text, image_path, source

Nguồn dữ liệu:
  - Threads : data_collection/threads/threads_data.json
              ảnh thực tế tại data_collection/threads/images/
  - Facebook: data_collection/facebook/facebook_data.json
              ảnh thực tế tại data_collection/facebook/images/

Ghi chú:
  - Ảnh Threads được resolve bằng basename (bỏ qua path cũ trong JSON)
  - Ảnh đổi tên post{id:05d}.{ext} để tránh conflict giữa 2 nguồn
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
THREADS_JSON = Path("data_collection/threads/threads_data.json")
THREADS_IMG_DIR = Path("data_collection/threads/images")

FB_JSON = Path("data_collection/facebook/facebook_data.json")
FB_IMG_BASE = Path("data_collection/facebook")  # images_path là relative từ đây

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

    output_records: list[dict] = []
    global_id = 1
    stats = {"threads_ok": 0, "threads_skip": 0, "fb_ok": 0, "fb_skip": 0}

    # ------------------------------------------------------------------
    # Nguồn 1: Threads
    # ------------------------------------------------------------------
    print("--- Threads ---")
    with open(THREADS_JSON, encoding="utf-8") as f:
        threads_records: list[dict] = json.load(f)

    for rec in threads_records:
        text = (rec.get("text") or "").strip()
        if not text:
            stats["threads_skip"] += 1
            continue

        # Resolve ảnh bằng basename — bỏ qua path cũ trong image_path
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

        output_records.append({
            "id": global_id,
            "text": text,
            "image_path": f"{OUTPUT_IMG_DIR}/{new_name}".replace("\\", "/"),
            "source": "threads",
        })
        global_id += 1
        stats["threads_ok"] += 1

    print(f"  Giữ lại : {stats['threads_ok']}")
    print(f"  Bỏ qua  : {stats['threads_skip']}")

    # ------------------------------------------------------------------
    # Nguồn 2: Facebook
    # ------------------------------------------------------------------
    print("--- Facebook ---")
    with open(FB_JSON, encoding="utf-8") as f:
        fb_records: list[dict] = json.load(f)

    for rec in fb_records:
        images_path = rec.get("images_path", "")
        if not images_path:
            stats["fb_skip"] += 1
            continue

        src_path = FB_IMG_BASE / images_path
        if not src_path.exists():
            stats["fb_skip"] += 1
            continue

        text = clean_facebook_text(rec.get("text", ""))
        if not text:
            stats["fb_skip"] += 1
            continue

        suffix = src_path.suffix.lower()
        new_name = f"post{global_id:05d}{suffix}"
        dest = OUTPUT_IMG_DIR / new_name
        if not dest.exists():
            shutil.copy2(src_path, dest)

        output_records.append({
            "id": global_id,
            "text": text,
            "image_path": f"{OUTPUT_IMG_DIR}/{new_name}".replace("\\", "/"),
            "source": "facebook",
        })
        global_id += 1
        stats["fb_ok"] += 1

    print(f"  Giữ lại : {stats['fb_ok']}")
    print(f"  Bỏ qua  : {stats['fb_skip']}")

    # ------------------------------------------------------------------
    # Lưu JSON
    # ------------------------------------------------------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    total = stats["threads_ok"] + stats["fb_ok"]
    print(f"\n=== KẾT QUẢ ===")
    print(f"Tổng records   : {total}  (threads={stats['threads_ok']}, facebook={stats['fb_ok']})")
    print(f"Ảnh đã copy    : {len(list(OUTPUT_IMG_DIR.iterdir()))}")
    print(f"JSON lưu tại   : {OUTPUT_JSON}")
    print(f"Ảnh lưu tại    : {OUTPUT_IMG_DIR}")


if __name__ == "__main__":
    merge_dataset()
