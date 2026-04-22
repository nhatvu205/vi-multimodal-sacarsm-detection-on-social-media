"""
preprocess_text.py
------------------
Preprocessing cuoi cung cho merged_dataset/data.json:
  - Kiem tra va loai bo records co text null / rong
  - Kiem tra va loai bo records co image_path null / file khong ton tai
  - Loai bo duplicate theo text chinh xac (exact match)
  - Phat hien near-duplicate bang MinHash (tuy chon)
  - Reassign ID lien tuc sau khi da loc
  - Luu file sach vao OUTPUT_JSON (mac dinh ghi de file goc)

Chay:
    python preprocessing/preprocess_text.py [--input PATH] [--output PATH]
                                            [--no-minhash] [--threshold 0.85]
                                            [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_INPUT = Path("merged_dataset/data.json")
DEFAULT_OUTPUT = Path("merged_dataset/data.json")  # ghi de file goc

# ------------------------------------------------------------------
# Tham so MinHash
# ------------------------------------------------------------------
MINHASH_THRESHOLD = 0.85  # Jaccard similarity de coi la near-dup
SHINGLE_SIZE = 3           # ki tu / word shingle


# ===================================================================
# UTILITY
# ===================================================================

def normalize_text(text: str) -> str:
    """Chuan hoa Unicode NFC, lower, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Tao tap word-shingles do dai k tu text."""
    words = text.split()
    if len(words) < k:
        return {text}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ===================================================================
# KIEM TRA NULL / MISSING
# ===================================================================

def check_null(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Tra ve (kept, dropped).
    Drop neu text rong hoac image_path rong.
    """
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
    """
    Tra ve (kept, dropped).
    Drop neu file anh khong ton tai tren disk.
    Chay tu thu muc goc cua repo (noi goi script).
    """
    kept, dropped = [], []
    for rec in records:
        img_path = rec.get("image_path", "")
        if img_path and Path(img_path).exists():
            kept.append(rec)
        else:
            dropped.append(rec)
    return kept, dropped


# ===================================================================
# DEDUP EXACT
# ===================================================================

def dedup_exact(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Loai bo ban sao co text chinh xac (sau normalize).
    Giu lan xuat hien dau tien.
    """
    seen: set[str] = set()
    kept, dropped = [], []
    for rec in records:
        key = normalize_text(rec.get("text", ""))
        if key in seen:
            dropped.append(rec)
        else:
            seen.add(key)
            kept.append(rec)
    return kept, dropped


# ===================================================================
# NEAR-DUPLICATE (MINHASH)
# ===================================================================

def _hash_shingle(s: str) -> int:
    return hash(s) & 0xFFFFFFFF


def minhash_signature(shingles: set[str], num_hashes: int = 128) -> list[int]:
    sig = [0xFFFFFFFF] * num_hashes
    for sh in shingles:
        h = _hash_shingle(sh)
        for i in range(num_hashes):
            # bitmix don gian
            val = (h ^ (i * 2654435761)) & 0xFFFFFFFF
            if val < sig[i]:
                sig[i] = val
    return sig


def jaccard_estimate(sig1: list[int], sig2: list[int]) -> float:
    matches = sum(a == b for a, b in zip(sig1, sig2))
    return matches / len(sig1)


def dedup_near(
    records: list[dict],
    threshold: float = MINHASH_THRESHOLD,
    num_hashes: int = 128,
) -> tuple[list[dict], list[dict]]:
    """
    Loai bo near-duplicate bang MinHash (O(N^2) voi pairwise check).
    Phu hop cho dataset nho-trung binh (~chuc nghin records).
    Giu ban dau tien trong cluster.
    """
    norms = [normalize_text(rec.get("text", "")) for rec in records]
    sigs = [
        minhash_signature(get_shingles(n), num_hashes) for n in norms
    ]

    dropped_idx: set[int] = set()
    for i in range(len(records)):
        if i in dropped_idx:
            continue
        for j in range(i + 1, len(records)):
            if j in dropped_idx:
                continue
            sim = jaccard_estimate(sigs[i], sigs[j])
            if sim >= threshold:
                dropped_idx.add(j)

    kept = [rec for idx, rec in enumerate(records) if idx not in dropped_idx]
    dropped = [rec for idx, rec in enumerate(records) if idx in dropped_idx]
    return kept, dropped


# ===================================================================
# REASSIGN ID
# ===================================================================

def reassign_ids(records: list[dict]) -> list[dict]:
    """Gan lai ID lien tuc 1..N sau khi da loc."""
    for new_id, rec in enumerate(records, start=1):
        rec["id"] = new_id
    return records


# ===================================================================
# MAIN
# ===================================================================

def preprocess(
    input_path: Path,
    output_path: Path,
    use_minhash: bool = True,
    minhash_threshold: float = MINHASH_THRESHOLD,
    dry_run: bool = False,
) -> None:
    print(f"Doc du lieu tu: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    total_start = len(records)
    print(f"Tong ban dau   : {total_start} records")

    # --- Kiem tra null text / image_path ---
    records, dropped_null = check_null(records)
    print(f"\n[NULL CHECK]")
    print(f"  Drop (null text/img): {len(dropped_null)}")
    print(f"  Con lai             : {len(records)}")

    # --- Kiem tra file anh ton tai ---
    records, dropped_nofile = check_image_exists(records)
    print(f"\n[IMAGE EXISTS CHECK]")
    print(f"  Drop (file khong co): {len(dropped_nofile)}")
    print(f"  Con lai             : {len(records)}")

    # --- Exact duplicate ---
    records, dropped_exact = dedup_exact(records)
    print(f"\n[EXACT DEDUP]")
    print(f"  Drop (exact dup)    : {len(dropped_exact)}")
    print(f"  Con lai             : {len(records)}")

    # --- Near-duplicate (MinHash) ---
    dropped_near: list[dict] = []
    if use_minhash:
        print(f"\n[NEAR-DEDUP MinHash] nguong={minhash_threshold}, dang tinh toan...")
        records, dropped_near = dedup_near(records, threshold=minhash_threshold)
        print(f"  Drop (near-dup)     : {len(dropped_near)}")
        print(f"  Con lai             : {len(records)}")

    # --- Reassign ID ---
    records = reassign_ids(records)

    total_drop = len(dropped_null) + len(dropped_nofile) + len(dropped_exact) + len(dropped_near)
    print(f"\n=== TONG KET ===")
    print(f"Ban dau        : {total_start}")
    print(f"Tong drop      : {total_drop}")
    print(f"  - Null        : {len(dropped_null)}")
    print(f"  - Anh mat     : {len(dropped_nofile)}")
    print(f"  - Exact dup   : {len(dropped_exact)}")
    print(f"  - Near dup    : {len(dropped_near)}")
    print(f"Con lai (sach) : {len(records)}")

    if dry_run:
        print("\n[DRY-RUN] Khong luu file.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess text data.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"File JSON dau vao (mac dinh: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File JSON dau ra (mac dinh: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-minhash",
        action="store_true",
        help="Tat phat hien near-duplicate bang MinHash",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MINHASH_THRESHOLD,
        help=f"Nguong Jaccard similarity cho near-dup (mac dinh: {MINHASH_THRESHOLD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi bao cao, khong ghi file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path=args.input,
        output_path=args.output,
        use_minhash=not args.no_minhash,
        minhash_threshold=args.threshold,
        dry_run=args.dry_run,
    )
