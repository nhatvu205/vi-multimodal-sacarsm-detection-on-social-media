"""
Convert ocr_images.jsonl to ocr_images.json and add ocr_text field to data.json.

Usage:
    python add_ocr_text.py

Both files are expected to be in the same directory as this script (merged_dataset/).
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
JSONL_PATH = SCRIPT_DIR / "ocr_images.jsonl"
OCR_JSON_PATH = SCRIPT_DIR / "ocr_images.json"
DATA_JSON_PATH = SCRIPT_DIR / "data.json"


def convert_jsonl_to_json() -> list[dict]:
    records = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    with open(OCR_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(records)} records -> {OCR_JSON_PATH}")
    return records


def add_ocr_text_to_data(ocr_records: list[dict]) -> None:
    ocr_map: dict[str, str] = {}
    for rec in ocr_records:
        filename = rec.get("filename", "")
        text = rec.get("text", "") if rec.get("status") == "ok" else ""
        if filename:
            ocr_map[filename] = text

    with open(DATA_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    matched = 0
    for record in data:
        image_path = record.get("image_path", "")
        filename = os.path.basename(image_path)
        ocr_text = ocr_map.get(filename, "")
        record["ocr_text"] = ocr_text
        if ocr_text:
            matched += 1

    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Added ocr_text to {len(data)} records ({matched} with non-empty OCR text)")


if __name__ == "__main__":
    ocr_records = convert_jsonl_to_json()
    add_ocr_text_to_data(ocr_records)
