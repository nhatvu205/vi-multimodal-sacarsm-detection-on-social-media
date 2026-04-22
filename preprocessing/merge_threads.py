"""
merge_threads.py
----------------
Gộp data Threads mới từ output/ vào:
  - merged_threads/images/          : ảnh đổi tên theo global ID (không collision)
  - merged_threads/threads_data.json: JSON sạch, append vào file hiện có

Chạy incremental: chỉ xử lý session mới (post_link chưa tồn tại trong JSON cũ).
Global ID tiếp nối từ max ID hiện có + 1 để tên ảnh không bị collision.

Đảm bảo:
  - Dedup theo post_link
  - Chỉ giữ record có ảnh thực tế tồn tại trên disk
  - image_path / image_paths là relative path tính từ thư mục chạy script
  - Text cleaning per-record (KHÔNG áp dụng cross-record để tránh truncate sai)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

# Fix encoding cho Windows terminal không hỗ trợ Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ===================================================================
# CẤU HÌNH
# ===================================================================
THREADS_INPUT_DIR = Path("output")                          # folder chứa các session mới
OUTPUT_DIR = Path("data_collection/merged_threads")
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_JSON_PATH = OUTPUT_DIR / "threads_data.json"


# ===================================================================
# TEXT CLEANING (per-record, không cross-record)
# ===================================================================

def clean_text(text: str) -> str | None:
    """
    Làm sạch text crawl thô:
    - Xóa username dòng đầu, timestamp, số tương tác, nội dung lặp
    - Trả về None nếu sau khi clean không còn nội dung
    """
    if not text or not text.strip():
        return None

    text = str(text).strip()

    # Xóa "Thịnh hành" / "Thread đầu tiên" ở đầu
    text = re.sub(
        r"^((thịnh hành|thread đầu tiên)[\s\n]*)+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Loại bỏ record có "Tiết lộ nội dung"
    if "tiết lộ nội dung" in text.lower():
        return None

    # Xử lý "Đang trả lời @..." — chỉ giữ phần trước hoặc sau
    parts = re.split(
        r"đang trả lời @[\w._]+\s*", text, flags=re.IGNORECASE, maxsplit=1
    )
    if len(parts) == 2:
        before, after = parts[0].strip(), parts[1].strip()
        text = before if before else after

    # Thay @username → "username"
    text = re.sub(r"@[\w._]+", "username", text)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    bad_indices: set[int] = set()

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Xóa "đã đăng lại" + dòng username kế tiếp
        if "đã đăng lại" in line_lower:
            bad_indices.add(i)
            if i + 1 < len(lines):
                bad_indices.add(i + 1)
            continue

        # Xóa dòng "Dịch"
        if line_lower == "dịch":
            bad_indices.add(i)
            continue

        # Cắt " Dịch" ở cuối dòng
        if line_lower.endswith(" dịch"):
            lines[i] = line[:-5].strip()
            line = lines[i]
            line_lower = line.lower()

        # Xóa dòng timestamp
        time_pattern = r"^\d+\s+(giây|phút|giờ|ngày|tuần|tháng|năm)"
        date_pattern = r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"

        if (
            re.match(time_pattern, line_lower)
            or re.match(date_pattern, line)
            or "vừa xong" in line_lower
        ):
            bad_indices.add(i)
            # Dòng trước timestamp thường là username hoặc topic tag
            if i > 0:
                bad_indices.add(i - 1)
            if i > 1 and len(lines[i - 2].split()) <= 5:
                bad_indices.add(i - 2)
            continue

        # Xóa số tương tác và dấu "/"
        if re.match(r"^[\d.,kKmM]+$", line) or line == "/":
            bad_indices.add(i)

    lines = [ln for i, ln in enumerate(lines) if i not in bad_indices]

    # Xóa pattern "#1: " (slide indicator) ngay trước dòng nội dung
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^#\d+:\s*$", lines[i]) and i + 1 < len(lines):
            i += 1
            continue
        cleaned.append(lines[i])
        i += 1

    # Xóa dòng liền kề bị lặp
    deduped: list[str] = []
    for ln in cleaned:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)

    # Xóa khối văn bản bị lặp ở cuối (slide carousel lặp caption)
    n = len(deduped)
    for length in range(n // 2, 0, -1):
        if deduped[-length:] == deduped[-2 * length : -length]:
            deduped = deduped[:-length]
            break

    result = "\n".join(deduped).strip()
    return result if result else None


# ===================================================================
# MAIN
# ===================================================================

def find_image_on_disk(session_dir: Path, image_local_entry: str) -> Path | None:
    """
    Resolve đường dẫn ảnh thực tế từ session_dir.
    image_local_entry có dạng: "output/threads_search_.../images/postXXXX_imgYY.ext"
    Tìm file theo basename trong thư mục images/ của session.
    """
    basename = Path(image_local_entry).name
    candidate = session_dir / "images" / basename
    return candidate if candidate.exists() else None


def merge_threads() -> None:
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Bước 0: Load dữ liệu hiện có để chạy incremental
    # ------------------------------------------------------------------
    existing_records: list[dict] = []
    seen_links: set[str] = set()
    global_id = 1

    if OUTPUT_JSON_PATH.exists():
        try:
            with open(OUTPUT_JSON_PATH, encoding="utf-8") as f:
                existing_records = json.load(f)
            for r in existing_records:
                if r.get("post_link"):
                    seen_links.add(r["post_link"])
            if existing_records:
                global_id = max(r["id"] for r in existing_records) + 1
            print(f"[INCREMENTAL] Đã có {len(existing_records)} records, tiếp tục từ ID={global_id}")
        except Exception as exc:
            print(f"[WARN] Không đọc được JSON cũ ({exc}), bắt đầu từ đầu")
            existing_records = []
            seen_links = set()
            global_id = 1

    # ------------------------------------------------------------------
    # Bước 1: Thu thập records từ các session trong output/
    # ------------------------------------------------------------------
    raw_records: list[dict] = []
    session_dirs: dict = {}  # post_link → session_dir (để resolve ảnh sau dedup)

    for session_dir in sorted(THREADS_INPUT_DIR.iterdir()):
        if not session_dir.is_dir():
            continue

        json_files = list(session_dir.glob("*.json"))
        if not json_files:
            print(f"[SKIP] Không có JSON trong {session_dir.name}")
            continue

        json_path = json_files[0]
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[ERROR] Đọc {json_path}: {exc}")
            continue

        if not isinstance(data, list):
            data = [data]

        for record in data:
            post_link = record.get("post_link", "")
            if post_link and post_link not in session_dirs:
                session_dirs[post_link] = session_dir
            raw_records.append(record)

        print(f"[OK] {session_dir.name}: {len(data)} records")

    print(f"\nTổng records từ output/ (chưa lọc): {len(raw_records)}")

    # ------------------------------------------------------------------
    # Bước 2: Dedup theo post_link — bỏ qua link đã có trong JSON cũ
    # ------------------------------------------------------------------
    deduped: list[dict] = []
    for record in raw_records:
        link = record.get("post_link", "")
        if not link or link in seen_links:
            continue
        seen_links.add(link)
        deduped.append(record)

    print(f"Sau dedup (kể cả records cũ): {len(deduped)} records mới cần xử lý")

    # ------------------------------------------------------------------
    # Bước 3: Lọc, clean text, resolve & copy ảnh, gán global ID
    # ------------------------------------------------------------------
    output_records: list[dict] = []
    skip_no_image = 0
    skip_empty_text = 0

    for record in deduped:
        post_link = record.get("post_link", "")
        session_dir = session_dirs.get(post_link)
        if session_dir is None:
            continue

        image_local_list: list[str] = record.get("image_local") or []
        if isinstance(image_local_list, str):
            image_local_list = [image_local_list]

        # Resolve ảnh thực tế trên disk
        resolved_src: list[Path] = []
        for entry in image_local_list:
            src = find_image_on_disk(session_dir, entry)
            if src is not None:
                resolved_src.append(src)

        # Chỉ giữ record có ít nhất 1 ảnh tồn tại
        if not resolved_src:
            skip_no_image += 1
            continue

        # Clean text
        raw_text = record.get("text", "")
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            skip_empty_text += 1
            continue

        # Copy ảnh → OUTPUT_IMG_DIR với tên mới (tránh collision)
        new_image_paths: list[str] = []
        for img_idx, src_path in enumerate(resolved_src, start=1):
            suffix = src_path.suffix.lower()
            new_name = f"post{global_id:05d}_img{img_idx:02d}{suffix}"
            dest_path = OUTPUT_IMG_DIR / new_name
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
            # Relative path tính từ thư mục chạy script
            new_image_paths.append(str(OUTPUT_IMG_DIR / new_name).replace("\\", "/"))

        output_records.append(
            {
                "id": global_id,
                "text": cleaned_text,
                "image_path": new_image_paths[0],
                "image_paths": new_image_paths if len(new_image_paths) > 1 else None,
                # metadata
                "post_link": post_link,
                "username": record.get("username", ""),
                "keyword": record.get("keyword", ""),
                "mode": record.get("mode", ""),
                "scraped_at": record.get("scraped_at", ""),
            }
        )
        global_id += 1

    # ------------------------------------------------------------------
    # Bước 4: Append records mới vào JSON hiện có
    # ------------------------------------------------------------------
    all_records = existing_records + output_records
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n=== KẾT QUẢ ===")
    print(f"Records cũ       : {len(existing_records)}")
    print(f"Records mới thêm : {len(output_records)}")
    print(f"Tổng records     : {len(all_records)}")
    print(f"Bỏ (thiếu ảnh)   : {skip_no_image}")
    print(f"Bỏ (text rỗng)   : {skip_empty_text}")
    print(f"Ảnh đã copy      : {len(list(OUTPUT_IMG_DIR.iterdir()))}")
    print(f"JSON lưu tại     : {OUTPUT_JSON_PATH}")
    print(f"Ảnh lưu tại      : {OUTPUT_IMG_DIR}")


if __name__ == "__main__":
    merge_threads()
