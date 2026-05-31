from __future__ import annotations

from pathlib import Path


class ModelAdapter:
    def __init__(self, config: dict, run_dir: Path):
        self.config = config
        self.run_dir = run_dir

    @property
    def model_name(self) -> str:
        return self.config['model']['key']

    @property
    def supports_training(self) -> bool:
        return False

    def train(self, train_records: list[dict], dev_records: list[dict], scenario: str) -> None:
        return None

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None) -> list[dict]:
        raise NotImplementedError

    def release(self) -> None:
        return None
