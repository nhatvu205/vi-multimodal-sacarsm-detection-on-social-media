"""
rename_images.py
----------------
Doi ten file anh trong merged_dataset/images/ cho khop voi ID record
trong merged_dataset/data.json.

Sau khi preprocess_text.py reassign ID, image_path van tro den ten file
cu (vd: record id=3976 nhung anh la post03989.jpg). Script nay dong bo
lai: post{id:05d}.{ext}.

Thuat toan 2-pass de tranh conflict:
  Pass 1: doi ten tat ca file nguon thanh tmp_{id:05d}.{ext}
  Pass 2: doi ten tmp_ thanh ten chinh thuc post{id:05d}.{ext}
Sau do cap nhat image_path trong data.json va luu lai.

Chay:
    python preprocessing/rename_images.py [--data PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DATA = Path("merged_dataset/data.json")


def rename_images(data_path: Path, dry_run: bool = False) -> None:
    print(f"Doc du lieu tu: {data_path}")
    with open(data_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    img_dir = data_path.parent / "images"
    stats = {"no_change": 0, "renamed": 0, "missing": 0, "error": 0, "recovered": 0}

    # ------------------------------------------------------------------
    # Buoc 0: Phat hien va xu ly cac file tmp_ con sot tu lan chay truoc
    # (Recovery: neu pass 2 that bai lan truoc, tmp_ van con tren disk)
    # ------------------------------------------------------------------
    leftover_tmps: dict[int, tuple[Path, str]] = {}  # id -> (tmp_path, suffix)
    for tmp_file in img_dir.glob("tmp_*.???*"):
        # Lay id tu ten file "tmp_XXXXX.ext"
        stem = tmp_file.stem  # "tmp_XXXXX"
        try:
            old_id = int(stem[4:])
            leftover_tmps[old_id] = (tmp_file, tmp_file.suffix.lower())
        except ValueError:
            pass

    if leftover_tmps:
        print(f"[RECOVERY] Tim thay {len(leftover_tmps)} file tmp_ con sot, xu ly truoc...")

    # ------------------------------------------------------------------
    # Xay dung ban do id -> record index (de cap nhat image_path sau)
    # ------------------------------------------------------------------
    id_to_idx: dict[int, int] = {rec["id"]: i for i, rec in enumerate(records)}

    # ------------------------------------------------------------------
    # Buoc 1: Xay dung ke hoach doi ten chinh
    # rename_plan: list of (src_path, final_path, record_index, new_image_path_str)
    # ------------------------------------------------------------------
    rename_plan: list[tuple[Path, Path, int, str]] = []

    for idx, rec in enumerate(records):
        rec_id: int = rec["id"]
        current_img: str = rec.get("image_path", "")
        suffix: str

        if not current_img:
            # Kiem tra xem co file tmp_ cho id nay khong (recovery)
            if rec_id in leftover_tmps:
                tmp_path, suffix = leftover_tmps.pop(rec_id)
                new_name = f"post{rec_id:05d}{suffix}"
                final_path = img_dir / new_name
                new_img_str = f"{img_dir}/{new_name}".replace("\\", "/")
                rename_plan.append((tmp_path, final_path, idx, new_img_str))
                stats["recovered"] += 1
            else:
                stats["missing"] += 1
            continue

        src_path = Path(current_img)
        suffix = src_path.suffix.lower()
        new_name = f"post{rec_id:05d}{suffix}"
        final_path = img_dir / new_name
        new_img_str = f"{img_dir}/{new_name}".replace("\\", "/")

        # Da dung ten dung
        if src_path.name == new_name:
            stats["no_change"] += 1
            records[idx]["image_path"] = new_img_str
            continue

        # File nguon khong ton tai — kiem tra tmp_ thay the
        if not src_path.exists():
            if rec_id in leftover_tmps:
                tmp_path, _ = leftover_tmps.pop(rec_id)
                rename_plan.append((tmp_path, final_path, idx, new_img_str))
                stats["recovered"] += 1
            else:
                print(f"  [MISS] {src_path.name} khong ton tai (id={rec_id})")
                stats["missing"] += 1
            continue

        rename_plan.append((src_path, final_path, idx, new_img_str))
        stats["renamed"] += 1

    print(f"\nKe hoach doi ten:")
    print(f"  Giu nguyen (da dung ten)  : {stats['no_change']}")
    print(f"  Can doi ten (moi)         : {stats['renamed']}")
    print(f"  Tu file tmp_ (recovery)   : {stats['recovered']}")
    print(f"  File anh khong tim thay   : {stats['missing']}")

    if dry_run:
        for src, final, _, _ in rename_plan[:10]:
            print(f"  [DRY] {src.name} -> {final.name}")
        if len(rename_plan) > 10:
            print(f"  ... va {len(rename_plan) - 10} file khac")
        print("\n[DRY-RUN] Khong thuc hien doi ten.")
        return

    if not rename_plan:
        print("\nTat ca file anh da dung ten dung.")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Da luu (cap nhat path format): {data_path}")
        return

    # ------------------------------------------------------------------
    # Pass 1: doi ten tat ca nguon -> tmp_{id:05d}.ext
    # (bo qua file da la tmp_, chung se xu ly thang o pass 2)
    # ------------------------------------------------------------------
    print("\nPass 1: doi ten sang tmp_...")
    tmp_plan: list[tuple[Path, Path, int, str]] = []
    for src_path, final_path, rec_idx, new_img_str in rename_plan:
        # Neu nguon da la file tmp_ (tu recovery), giu nguyen cho pass 2
        if src_path.name.startswith("tmp_"):
            tmp_plan.append((src_path, final_path, rec_idx, new_img_str))
            continue

        rec_id = records[rec_idx]["id"]
        suffix = final_path.suffix
        tmp_name = f"tmp_{rec_id:05d}{suffix}"
        tmp_path = img_dir / tmp_name
        try:
            # replace() de ghi de neu tmp_ cu ton tai tu lan chay truoc
            src_path.replace(tmp_path)
            tmp_plan.append((tmp_path, final_path, rec_idx, new_img_str))
        except Exception as exc:
            print(f"  [ERROR] {src_path.name} -> {tmp_name}: {exc}")
            stats["error"] += 1

    # ------------------------------------------------------------------
    # Pass 2: doi ten tmp_ -> ten chinh thuc post{id:05d}.ext
    # ------------------------------------------------------------------
    print("Pass 2: doi ten tu tmp_ sang ten chinh thuc...")
    for tmp_path, final_path, rec_idx, new_img_str in tmp_plan:
        try:
            # replace() ghi de orphan images (anh cua record dup da xoa)
            tmp_path.replace(final_path)
            records[rec_idx]["image_path"] = new_img_str
        except Exception as exc:
            print(f"  [ERROR] {tmp_path.name} -> {final_path.name}: {exc}")
            stats["error"] += 1

    # ------------------------------------------------------------------
    # Luu data.json voi image_path da cap nhat
    # ------------------------------------------------------------------
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    total_done = stats["renamed"] + stats["recovered"] - stats["error"]
    print(f"\n=== KET QUA ===")
    print(f"Giu nguyen    : {stats['no_change']}")
    print(f"Da doi ten    : {total_done}")
    print(f"  - Moi       : {stats['renamed'] - max(0, stats['error'])}")
    print(f"  - Recovery  : {stats['recovered']}")
    print(f"Loi           : {stats['error']}")
    print(f"Thieu file    : {stats['missing']}")
    print(f"Da luu        : {data_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dong bo ten anh voi ID record")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"File data.json (mac dinh: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi bao cao ke hoach, khong doi ten that",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rename_images(data_path=args.data, dry_run=args.dry_run)
