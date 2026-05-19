from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import InputRecord
from .utils_logging import get_logger

logger = get_logger(__name__)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [p]
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend([repo_root / path, Path.cwd() / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"File not found: {path}. Tried: {tried}")


def load_ocr_lookup(ocr_path: str) -> Dict[str, str]:
    resolved = _resolve_path(ocr_path)
    items = json.loads(resolved.read_text(encoding="utf-8"))
    lookup: Dict[str, str] = {}

    if isinstance(items, dict):
        for k, v in items.items():
            if isinstance(v, str) and v.strip():
                lookup[k] = v.strip()
        return lookup

    for item in items:
        filename = item.get("filename") or item.get("image") or item.get("image_path")
        text = item.get("text") or item.get("ocr_text")
        status = item.get("status")
        if filename and text and (status in (None, "ok", "success")):
            lookup[Path(filename).name] = str(text).strip()
    return lookup


def load_input_records(path: str, ocr_path: Optional[str] = None) -> List[InputRecord]:
    resolved_path = _resolve_path(path)
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
                records.append(InputRecord(**json.loads(line)))
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_no, resolved_path, exc)

    if ocr_path:
        lookup = load_ocr_lookup(ocr_path)
        injected = 0
        updated: List[InputRecord] = []
        for rec in records:
            if rec.ocr_text:
                updated.append(rec)
                continue
            fname = Path(rec.image_path).name if rec.image_path else None
            if fname and fname in lookup:
                updated.append(rec.model_copy(update={"ocr_text": lookup[fname]}))
                injected += 1
            else:
                updated.append(rec)
        records = updated
        logger.info("Injected OCR text for %d/%d records", injected, len(records))

    logger.info("Loaded %d input records from %s", len(records), resolved_path)
    return records
