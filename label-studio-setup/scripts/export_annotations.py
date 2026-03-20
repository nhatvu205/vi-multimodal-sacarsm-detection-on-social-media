"""
export_annotations.py
---------------------
Exports all annotated tasks from a Label Studio project to a timestamped
JSON file inside data/exports/.

Usage:
    python scripts/export_annotations.py
    python scripts/export_annotations.py --only-annotated

Required environment variables (set them in .env):
    LS_URL      - Label Studio base URL, e.g. http://localhost:8080
    LS_API_KEY  - Your API key from http://localhost:8080/user/account
    PROJECT_ID  - ID of the project to export from

Output:
    data/exports/annotations_<YYYYMMDD_HHMMSS>.json

Export format (Label Studio JSON):
    Each entry in the output array is one task, e.g.:
    {
      "id": 1,
      "data": { "image": "...", "text": "..." },
      "annotations": [
        {
          "result": [
            {
              "from_name": "sentiment",
              "to_name": "image",
              "type": "choices",
              "value": { "choices": ["Positive"] }
            },
            ...
          ],
          "completed_by": { "email": "annotator@example.com" },
          "created_at": "2025-01-01T00:00:00.000000Z"
        }
      ]
    }
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

load_dotenv(Path(__file__).parent.parent / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LS_API_KEY", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")

EXPORTS_DIR = Path(__file__).parent.parent / "data" / "exports"


def main():
    parser = argparse.ArgumentParser(description="Export annotations from Label Studio.")
    parser.add_argument(
        "--only-annotated",
        action="store_true",
        help="Skip tasks that have no annotations yet.",
    )
    args = parser.parse_args()

    if not LS_API_KEY:
        print(
            "ERROR: LS_API_KEY is not set in .env.\n"
            "  Retrieve it from http://localhost:8080/user/account"
        )
        sys.exit(1)

    if not PROJECT_ID:
        print("ERROR: PROJECT_ID is not set in .env.")
        sys.exit(1)

    print(f"Connecting to Label Studio at {LS_URL} ...")
    ls = Client(url=LS_URL, api_key=LS_API_KEY)

    try:
        ls.check_connection()
    except Exception as exc:
        print(f"ERROR: Cannot reach Label Studio. Details: {exc}")
        sys.exit(1)

    project = ls.get_project(int(PROJECT_ID))
    print(f"Exporting tasks from project #{PROJECT_ID} ...")

    tasks = project.export_tasks(export_type="JSON")

    if args.only_annotated:
        before = len(tasks)
        tasks = [t for t in tasks if t.get("annotations")]
        print(f"  Filtered to annotated only: {before} -> {len(tasks)} task(s)")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EXPORTS_DIR / f"annotations_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(tasks, fh, indent=2, ensure_ascii=False)

    annotated_count = sum(1 for t in tasks if t.get("annotations"))
    print(
        f"\nExport complete.\n"
        f"  Total tasks exported  : {len(tasks)}\n"
        f"  Tasks with annotations: {annotated_count}\n"
        f"  Output file           : {output_path}"
    )


if __name__ == "__main__":
    main()
