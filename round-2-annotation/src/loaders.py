from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .schemas import InputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)

def load_ocr_lookup(ocr_path: str) -> Dict[str, str]:
    """Load ocr_images.json → dict {filename: text}."""
    p = Path(ocr_path)
    if not p.exists():
        logger.warning("OCR file not found: %s", ocr_path)
        return {}
    items = json.loads(p.read_text(encoding="utf-8"))
    return {
        item["filename"]: item["text"]
        for item in items
        if item.get("status") == "ok" and item.get("text")
    }
def load_input_records(path: str) -> List[InputRecord]:
    """Load input records from a JSON array file or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")

    raw_text = p.read_text(encoding="utf-8").strip()
    records: List[InputRecord] = []

    if raw_text.startswith("["):
        items = json.loads(raw_text)
        for item in tqdm(items, desc="Loading input records", unit="rec", leave=False):
            records.append(InputRecord(**item))
    else:
        lines = [l for l in raw_text.splitlines() if l.strip()]
        for line_no, line in tqdm(
            enumerate(lines, start=1),
            total=len(lines),
            desc="Loading input records",
            unit="rec",
            leave=False,
        ):
            try:
                item = json.loads(line)
                records.append(InputRecord(**item))
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_no, path, exc)

    logger.info("Loaded %d input records from %s", len(records), path)
    # Inject OCR nếu có
    if ocr_path:
        ocr_lookup = load_ocr_lookup(ocr_path)
        injected = 0
        for rec in records:
            if rec.ocr_text:          # record đã có sẵn OCR thì bỏ qua
                continue
            # Lấy filename từ image_path
            fname = Path(rec.image_path).name if rec.image_path else None
            if fname and fname in ocr_lookup:
                # Pydantic model cần dùng model_copy để update
                records[records.index(rec)] = rec.model_copy(
                    update={"ocr_text": ocr_lookup[fname]}
                )
                injected += 1
        logger.info("Injected OCR text for %d/%d records", injected, len(records))

    return records
