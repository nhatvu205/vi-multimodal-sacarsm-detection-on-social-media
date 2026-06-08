"""
Trích xuất mẫu 10% từ data/final-data/data.json để human validation,
đồng thời giữ phân phối theo tổ hợp nhãn (mm_label, text_label, image_label).

Mặc định script sẽ:
- đọc `data/final-data/data.json`
- lấy mẫu 10% theo stratified sampling trên combo nhãn
- đảm bảo mọi combo có trong dữ liệu gốc đều xuất hiện trong mẫu
- lưu ra `data/final-data/human_validation_10pct.json`

Run:
    python3 preprocessing/extract_human_validation_sample.py

Hoặc:
    python3 preprocessing/extract_human_validation_sample.py \
        --input data/final-data/data.json \
        --output data/final-data/human_validation_10pct.json \
        --ratio 0.1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("data/final-data/data.json")
DEFAULT_OUTPUT = Path("data/final-data/human_validation_10pct.json")
LABEL_FIELDS = ("mm_label", "text_label", "image_label")


def make_combo(record: dict) -> tuple[int, int, int]:
    return tuple(int(record[field]) for field in LABEL_FIELDS)


def combo_to_name(combo: tuple[int, int, int]) -> str:
    mm_label, text_label, image_label = combo
    return f"mm={mm_label}|text={text_label}|image={image_label}"


def compute_targets(group_sizes: dict[tuple[int, int, int], int], ratio: float) -> dict[tuple[int, int, int], int]:
    total_size = sum(group_sizes.values())
    target_total = round(total_size * ratio)
    combo_count = len(group_sizes)

    if total_size == 0:
        return {combo: 0 for combo in group_sizes}

    if ratio > 0:
        target_total = max(combo_count, target_total)
    target_total = min(total_size, target_total)

    raw_targets = {combo: size * ratio for combo, size in group_sizes.items()}
    targets = {}
    remainders = []

    for combo, raw_target in raw_targets.items():
        base = math.floor(raw_target)
        if ratio > 0 and group_sizes[combo] > 0:
            base = max(1, base)
        base = min(group_sizes[combo], base)
        targets[combo] = base
        remainders.append((raw_target - math.floor(raw_target), combo))

    current_total = sum(targets.values())

    if current_total < target_total:
        for _, combo in sorted(remainders, reverse=True):
            if current_total >= target_total:
                break
            if targets[combo] < group_sizes[combo]:
                targets[combo] += 1
                current_total += 1

    elif current_total > target_total:
        for _, combo in sorted(remainders):
            if current_total <= target_total:
                break
            if targets[combo] > 1:
                targets[combo] -= 1
                current_total -= 1

    return targets


def sample_records(records: list[dict], ratio: float, seed: int) -> tuple[list[dict], dict[tuple[int, int, int], int], dict[tuple[int, int, int], int]]:
    grouped_records: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[make_combo(record)].append(record)

    group_sizes = {combo: len(items) for combo, items in grouped_records.items()}
    targets = compute_targets(group_sizes, ratio)

    rng = random.Random(seed)
    sampled_records: list[dict] = []

    for combo, items in grouped_records.items():
        shuffled_items = items[:]
        rng.shuffle(shuffled_items)
        for record in shuffled_items[: targets[combo]]:
            sampled_record = dict(record)
            sampled_record["label_combo"] = combo_to_name(combo)
            sampled_records.append(sampled_record)

    sampled_records.sort(key=lambda item: item.get("id", 0))
    return sampled_records, group_sizes, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Lấy mẫu stratified 10% theo combo nhãn để human validation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Đường dẫn file JSON đầu vào.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Đường dẫn file JSON đầu ra.")
    parser.add_argument("--ratio", type=float, default=0.10, help="Tỷ lệ lấy mẫu, mặc định 0.10.")
    parser.add_argument("--seed", type=int, default=42, help="Seed random để tái lập kết quả.")
    args = parser.parse_args()

    if not 0 < args.ratio <= 1:
        raise ValueError("--ratio phải nằm trong khoảng (0, 1].")

    with open(args.input, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    sampled_records, group_sizes, targets = sample_records(records, args.ratio, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled_records, f, ensure_ascii=False, indent=2)

    total_records = len(records)
    total_sampled = len(sampled_records)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Total : {total_records}")
    print(f"Sample: {total_sampled} ({total_sampled / total_records:.2%})")
    print("\nPhân phối theo combo nhãn:")

    for combo in sorted(group_sizes):
        original_count = group_sizes[combo]
        sampled_count = targets[combo]
        print(
            f"- {combo_to_name(combo)} | "
            f"goc={original_count} ({original_count / total_records:.2%}) | "
            f"mau={sampled_count} ({sampled_count / total_sampled:.2%})"
        )


if __name__ == "__main__":
    main()
