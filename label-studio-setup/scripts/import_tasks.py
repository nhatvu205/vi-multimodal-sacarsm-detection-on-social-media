"""
import_tasks.py
--------------
Imports labeling tasks from a JSON file into a Label Studio project.

Usage:
    python scripts/import_tasks.py
    python scripts/import_tasks.py --file path/to/your_tasks.json

Required environment variables (set them in .env):
    LS_URL      - Label Studio base URL, e.g. http://localhost:8080
    LS_API_KEY  - Your API key from http://localhost:8080/user/account
    PROJECT_ID  - ID of the project to import tasks into

Task JSON format:
    Each task must have a "data" object with at least two fields:
      "image" - local file URL, e.g. /data/local-files/?d=images/photo.jpg
      "text"  - text content associated with the image

    Example:
        [
          {
            "data": {
              "image": "/data/local-files/?d=images/photo.jpg",
              "text": "Caption or context for this image."
            }
          }
        ]

    The image file must exist at data/images/photo.jpg on the host.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

load_dotenv(Path(__file__).parent.parent / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LS_API_KEY", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")

DEFAULT_TASKS_FILE = Path(__file__).parent.parent / "sample_data" / "tasks.json"


def main():
    parser = argparse.ArgumentParser(description="Import tasks into Label Studio.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_TASKS_FILE,
        help="Path to the tasks JSON file (default: sample_data/tasks.json)",
    )
    args = parser.parse_args()

    if not LS_API_KEY:
        print(
            "ERROR: LS_API_KEY is not set in .env.\n"
            "  Retrieve it from http://localhost:8080/user/account"
        )
        sys.exit(1)

    if not PROJECT_ID:
        print(
            "ERROR: PROJECT_ID is not set in .env.\n"
            "  Run:  python scripts/create_project.py\n"
            "  Then copy the printed Project ID into your .env file."
        )
        sys.exit(1)

    tasks_file: Path = args.file
    if not tasks_file.exists():
        print(f"ERROR: Tasks file not found: {tasks_file}")
        sys.exit(1)

    with tasks_file.open(encoding="utf-8") as fh:
        tasks = json.load(fh)

    if not isinstance(tasks, list) or len(tasks) == 0:
        print("ERROR: Tasks file must contain a non-empty JSON array.")
        sys.exit(1)

    print(f"Connecting to Label Studio at {LS_URL} ...")
    ls = Client(url=LS_URL, api_key=LS_API_KEY)

    try:
        ls.check_connection()
    except Exception as exc:
        print(f"ERROR: Cannot reach Label Studio. Details: {exc}")
        sys.exit(1)

    project = ls.get_project(int(PROJECT_ID))
    print(f"Importing {len(tasks)} task(s) into project #{PROJECT_ID} ...")
    project.import_tasks(tasks)

    print(
        f"\nImport complete.\n"
        f"  Imported : {len(tasks)} task(s)\n"
        f"  View     : {LS_URL}/projects/{PROJECT_ID}/data\n\n"
        f"Annotators can now open the project and start labeling."
    )


if __name__ == "__main__":
    main()
