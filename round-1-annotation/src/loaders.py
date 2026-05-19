from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schemas import InputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)


def _resolve_input_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [p]
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend([
            repo_root / path,
            Path.cwd() / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Input data file not found: {path}. Tried: {tried}")


def load_input_records(path: str) -> List[InputRecord]:
    resolved_path = _resolve_input_path(path)
    raw_text = resolved_path.read_text(encoding="utf-8").strip()
    records: List[InputRecord] = []

    if raw_text.startswith("["):
        items = json.loads(raw_text)
        for item in items:
            records.append(InputRecord(**item))
    else:
        lines = [l for l in raw_text.splitlines() if l.strip()]
        for line_no, line in enumerate(lines, start=1):
            try:
                item = json.loads(line)
                records.append(InputRecord(**item))
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_no, resolved_path, exc)

    logger.info("Loaded %d input records from %s", len(records), resolved_path)
    return records
