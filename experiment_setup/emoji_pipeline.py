"""
emoji_pipeline.py
-----------------
Pipeline xử lý emoji cho toàn bộ dataset (cleaned_dataset.json).
Tích hợp emoji_processor.py → sinh ra các biến thể text cho ablation.

Kịch bản ablation (text field):
  A0: text gốc (không xử lý gì)
  A1: text + process emoticon ASCII + slang (không có text_sentiment → no conflict)
  A2: text + process emoticon ASCII + slang + conflict boost (cần text_sentiment)
  A3: text stripped emoji hoàn toàn (baseline không có emoji/emoticon)

Output:
  processed_dataset.json   — mỗi sample bổ sung các field mới
  feature_matrix.csv       — feature vector phẳng cho classifier
  processing_report.json   — thống kê pipeline

Usage:
  python emoji_pipeline.py [--input cleaned_dataset.json] [--output_dir ./output]
"""

import json
import csv
import argparse
import re
import unicodedata
import sys
import os
from pathlib import Path
from dataclasses import asdict

# Import từ emoji_processor.py cùng thư mục
sys.path.insert(0, str(Path(__file__).parent))
from emoji_processor import (
    process,
    get_feature_vector,
    extract_emojis,
    ProcessedText,
)


# ─────────────────────────────────────────────
#  HÀM HỖ TRỢ
# ─────────────────────────────────────────────

def strip_all_emoji_and_emoticon(text: str) -> str:
    """
    Kịch bản A3: loại bỏ hoàn toàn emoji Unicode và emoticon ASCII.
    Dùng làm baseline text-only thuần túy.
    """
    # Loại emoji Unicode
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm")
        and ord(ch) < 0x1F000  # loại supplementary symbols
    )
    # Loại emoji range phổ biến
    cleaned = re.sub(
        r"[\U0001F300-\U0001FAFF"   # Misc symbols, emoticons, transport, etc.
        r"\U00002600-\U000027BF"    # Misc symbols
        r"\U0000FE00-\U0000FE0F"    # Variation selectors
        r"\U0001F900-\U0001F9FF"    # Supplemental symbols
        r"\u200d"                  # Zero-width joiner
        r"]+",
        " ", cleaned
    )
    # Loại emoticon ASCII phổ biến
    emoticon_patterns = [
        r"[:=;xX8B]-?[)D\]\[(pPoO0|]+",
        r"[;:]-?['\"]-?[\(\)]",
        r">\s*[:<]-?\s*\)",
        r"\^[-_]?\^",
        r"T[-_]?T",
        r"Q[-_]?Q",
        r"=[)(\]]+",
        r"-[-_]+-",
        r"[oO][._][oO]",
    ]
    for pat in emoticon_patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    # Chuẩn hoá khoảng trắng thừa
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def process_sample(sample: dict, text_sentiment: float | None = None) -> dict:
    """
    Xử lý một sample, trả về sample bổ sung các field:
      - text_A0: text gốc (không đổi)
      - text_A1: emoticon + slang processed (no conflict)
      - text_A2: emoticon + slang + conflict boost (nếu có text_sentiment)
      - text_A3: stripped (no emoji/emoticon)
      - emoji_features: dict feature từ emoji_processor
      - sarcasm_score, sarcasm_score_with_conflict, conflict_boost
    """
    text = sample.get("text", "")

    # A0: giữ nguyên
    text_A0 = text

    # A1: xử lý emoticon + slang, không conflict
    result_A1: ProcessedText = process(text, text_sentiment=None)
    text_A1 = result_A1.processed

    # A2: xử lý với conflict (nếu có text_sentiment)
    if text_sentiment is not None:
        result_A2: ProcessedText = process(text, text_sentiment=text_sentiment)
        text_A2 = result_A2.processed
        sarcasm_final = result_A2.sarcasm_score_with_conflict
        conflict_boost = result_A2.conflict_boost
    else:
        text_A2 = text_A1
        sarcasm_final = result_A1.sarcasm_score
        conflict_boost = 0.0

    # A3: stripped hoàn toàn
    text_A3 = strip_all_emoji_and_emoticon(text)

    # Feature vector (từ A1 result — text gốc + features)
    features = get_feature_vector(result_A1)
    # Override với conflict-aware scores nếu có
    features["sarcasm_score_final"] = sarcasm_final
    features["conflict_boost"] = conflict_boost
    features["has_conflict"] = int(conflict_boost > 0)

    return {
        **sample,
        "text_A0": text_A0,
        "text_A1": text_A1,
        "text_A2": text_A2,
        "text_A3": text_A3,
        "emoji_features": {
            **result_A1.emoji_features,
            "sarcasm_score": result_A1.sarcasm_score,
            "sarcasm_score_with_conflict": sarcasm_final,
            "conflict_boost": conflict_boost,
            "sentiment_polarity": result_A1.sentiment_polarity,
        },
        "feature_vector": features,
    }


# ─────────────────────────────────────────────
#  PIPELINE CHÍNH
# ─────────────────────────────────────────────

def run_pipeline(input_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Đọc dataset: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"      → {len(dataset)} samples")

    # Thống kê
    stats = {
        "total": len(dataset),
        "has_emoji": 0,
        "has_emoticon": 0,
        "has_sarcasm_emoji": 0,
        "has_conflict": 0,
        "label_distribution": {"mm_label": {}, "text_label": {}, "image_label": {}},
    }

    print("[2/4] Xử lý emoji pipeline...")
    processed = []
    for i, sample in enumerate(dataset):
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{len(dataset)}...")

        # NOTE: text_sentiment = None vì chưa có PhoBERT inference
        # Khi tích hợp PhoBERT, truyền score thực vào đây
        result = process_sample(sample, text_sentiment=None)
        processed.append(result)

        # Cập nhật thống kê
        ef = result["emoji_features"]
        if ef.get("emoji_count", 0) > 0:
            stats["has_emoji"] += 1
        if result["feature_vector"].get("emoticon_count", 0) > 0:
            stats["has_emoticon"] += 1
        if ef.get("has_sarcasm_emoji"):
            stats["has_sarcasm_emoji"] += 1
        if ef.get("conflict_boost", 0) > 0:
            stats["has_conflict"] += 1

    # Label distribution
    for label_key in ["mm_label", "text_label", "image_label"]:
        from collections import Counter
        dist = Counter(str(s.get(label_key, "?")) for s in dataset)
        stats["label_distribution"][label_key] = dict(dist)

    print("[3/4] Lưu output...")

    # 1. processed_dataset.json
    out_json = output_dir / "processed_dataset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"      → {out_json}")

    # 2. feature_matrix.csv
    out_csv = output_dir / "feature_matrix.csv"
    if processed:
        fv_keys = list(processed[0]["feature_vector"].keys())
        meta_keys = ["id", "mm_label", "text_label", "image_label", "source"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=meta_keys + fv_keys)
            writer.writeheader()
            for s in processed:
                row = {k: s.get(k, "") for k in meta_keys}
                row.update(s["feature_vector"])
                writer.writerow(row)
    print(f"      → {out_csv}")

    # 3. processing_report.json
    out_report = output_dir / "processing_report.json"
    stats["emoji_coverage_pct"] = round(stats["has_emoji"] / stats["total"] * 100, 2)
    stats["emoticon_coverage_pct"] = round(stats["has_emoticon"] / stats["total"] * 100, 2)
    stats["sarcasm_emoji_pct"] = round(stats["has_sarcasm_emoji"] / stats["total"] * 100, 2)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"      → {out_report}")

    print("[4/4] Xong! Tóm tắt:")
    print(f"      Total samples   : {stats['total']}")
    print(f"      Có emoji Unicode: {stats['has_emoji']} ({stats['emoji_coverage_pct']}%)")
    print(f"      Có emoticon ASCII: {stats['has_emoticon']} ({stats['emoticon_coverage_pct']}%)")
    print(f"      Có sarcasm emoji: {stats['has_sarcasm_emoji']} ({stats['sarcasm_emoji_pct']}%)")
    print(f"      Ablation texts  : text_A0, text_A1, text_A2, text_A3 → sẵn sàng")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emoji processing pipeline")
    parser.add_argument("--input",      default="cleaned_dataset.json")
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir)
