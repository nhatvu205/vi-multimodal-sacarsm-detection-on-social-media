"""
create_project.py
-----------------
Creates a new Label Studio project using the labeling config in
config/labeling_config.xml and prints the project ID.

Usage:
    python scripts/create_project.py

Required environment variables (set them in .env):
    LS_URL      - Label Studio base URL, e.g. http://localhost:8080
    LS_API_KEY  - Your API key from http://localhost:8080/user/account

Optional:
    PROJECT_TITLE - Name for the new project (default: "Multimodal Annotation")
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

load_dotenv(Path(__file__).parent.parent / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LS_API_KEY", "")
PROJECT_TITLE = os.getenv("PROJECT_TITLE", "Multimodal Annotation")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "labeling_config.xml"


def main():
    if not LS_API_KEY:
        print(
            "ERROR: LS_API_KEY is not set.\n"
            "  1. Open http://localhost:8080/user/account in your browser.\n"
            "  2. Copy the API token shown there.\n"
            "  3. Paste it as LS_API_KEY in your .env file."
        )
        sys.exit(1)

    if not CONFIG_PATH.exists():
        print(f"ERROR: Labeling config not found at {CONFIG_PATH}")
        sys.exit(1)

    label_config = CONFIG_PATH.read_text(encoding="utf-8")

    print(f"Connecting to Label Studio at {LS_URL} ...")
    ls = Client(url=LS_URL, api_key=LS_API_KEY)

    try:
        ls.check_connection()
    except Exception as exc:
        print(
            f"ERROR: Cannot reach Label Studio.\n"
            f"  Details: {exc}\n"
            f"  Make sure the containers are running:  docker compose up -d"
        )
        sys.exit(1)

    project = ls.start_project(
        title=PROJECT_TITLE,
        label_config=label_config,
    )

    print(
        f"\nProject created successfully!\n"
        f"  Title      : {project.params['title']}\n"
        f"  Project ID : {project.id}\n"
        f"  URL        : {LS_URL}/projects/{project.id}/\n\n"
        f"Add PROJECT_ID={project.id} to your .env file, then run:\n"
        f"  python scripts/import_tasks.py"
    )


if __name__ == "__main__":
    main()
