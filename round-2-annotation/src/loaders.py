from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from .schemas import InputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)


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
    return records
