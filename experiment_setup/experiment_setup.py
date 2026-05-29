"""
experiment_setup.py
-------------------
Setup chung cho toàn bộ thực nghiệm Multimodal Sarcasm Detection.

Chức năng:
  1. Tạo train/val/test split cố định (seed=42) — dùng chung cho tất cả model
  2. EmojiDataset: torch Dataset load processed_dataset.json
  3. Giao diện model base (ModelBase) — mỗi người implement cho model của mình
  4. evaluate(): tính Accuracy, F1 (macro + weighted), AUC, Confusion matrix
  5. run_ablation(): chạy đủ 4 kịch bản (A0–A3) cho một model
  6. save_results(): lưu kết quả vào results/<model_name>/

Usage (thực nghiệm):
  from experiment_setup import EmojiDataset, evaluate, run_ablation, make_splits

  train_ids, val_ids, test_ids = make_splits("output/processed_dataset.json")
  # Implement ModelBase → truyền vào run_ablation(...)
"""

import json
import csv
import random
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ── Optional: torch / sklearn (chỉ import khi cần) ──────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch chưa cài. Dataset class vẫn dùng được ở chế độ plain list.")

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, confusion_matrix,
        classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[INFO] scikit-learn chưa cài. Dùng: pip install scikit-learn")


# ─────────────────────────────────────────────
#  1. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

SPLIT_RATIO = (0.7, 0.15, 0.15)   # train / val / test
RANDOM_SEED = 42


def make_splits(
    processed_path: str = "output/processed_dataset.json",
    split_ratio: tuple = SPLIT_RATIO,
    seed: int = RANDOM_SEED,
    save_dir: str = "output",
) -> tuple[list[int], list[int], list[int]]:
    """
    Tạo split cố định theo ID — dùng chung cho tất cả model.
    Lưu split vào split_ids.json để tái sử dụng.

    Returns:
        (train_ids, val_ids, test_ids) — list của sample['id']
    """
    split_file = Path(save_dir) / "split_ids.json"

    # Nếu đã có split → load lại (đảm bảo tất cả model dùng cùng split)
    if split_file.exists():
        with open(split_file, encoding="utf-8") as f:
            splits = json.load(f)
        print(f"[Split] Load split từ {split_file}: "
              f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits["train"], splits["val"], splits["test"]

    with open(processed_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # Stratify theo mm_label
    label0 = [s["id"] for s in dataset if s.get("mm_label") == 0]
    label1 = [s["id"] for s in dataset if s.get("mm_label") == 1]

    rng = random.Random(seed)
    rng.shuffle(label0)
    rng.shuffle(label1)

    def _split(ids):
        n = len(ids)
        n_train = int(n * split_ratio[0])
        n_val   = int(n * split_ratio[1])
        return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]

    tr0, v0, te0 = _split(label0)
    tr1, v1, te1 = _split(label1)

    train_ids = tr0 + tr1
    val_ids   = v0  + v1
    test_ids  = te0 + te1

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    print(f"[Split] Tạo split mới → {split_file}")
    print(f"        train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────
#  2. DATASET CLASS
# ─────────────────────────────────────────────

class EmojiDataset:
    """
    Dataset wrapper cho processed_dataset.json.
    Hỗ trợ 4 kịch bản ablation: text_field = "text_A0" | "text_A1" | "text_A2" | "text_A3"

    Nếu PyTorch available, kế thừa torch.utils.data.Dataset.
    Nếu không, vẫn dùng được như list (plain Python).

    Args:
        data:        list[dict] — đã lọc theo split IDs
        text_field:  "text_A0" | "text_A1" | "text_A2" | "text_A3"
        label_field: "mm_label" | "text_label" | "image_label"
        image_dir:   thư mục chứa ảnh (nếu dùng multimodal)
        include_features: bao gồm feature_vector (cho fusion)
    """

    ABLATION_SCENARIOS = {
        "A0": "text_A0",   # text gốc
        "A1": "text_A1",   # emoticon + slang processed
        "A2": "text_A2",   # + conflict boost
        "A3": "text_A3",   # stripped (no emoji/emoticon)
    }

    def __init__(
        self,
        data: list[dict],
        text_field: str = "text_A1",
        label_field: str = "mm_label",
        image_dir: str = "data/images",
        include_features: bool = False,
    ):
        self.data = data
        self.text_field = text_field
        self.label_field = label_field
        self.image_dir = image_dir
        self.include_features = include_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        item = {
            "id":        sample["id"],
            "text":      sample.get(self.text_field, sample.get("text", "")),
            "label":     sample.get(self.label_field, 0),
            "image_path": str(Path(self.image_dir) / Path(sample.get("image_path", "")).name),
            "ocr_text":  sample.get("ocr_text", ""),
        }
        if self.include_features:
            item["features"] = sample.get("feature_vector", {})
        return item

    def get_texts(self) -> list[str]:
        return [self.data[i].get(self.text_field, "") for i in range(len(self.data))]

    def get_labels(self) -> list[int]:
        return [self.data[i].get(self.label_field, 0) for i in range(len(self.data))]

    @classmethod
    def from_processed_file(
        cls,
        processed_path: str,
        split_ids: list[int],
        **kwargs,
    ) -> "EmojiDataset":
        with open(processed_path, encoding="utf-8") as f:
            all_data = json.load(f)
        id_set = set(split_ids)
        filtered = [s for s in all_data if s["id"] in id_set]
        return cls(filtered, **kwargs)


# Nếu có torch: tạo TorchDataset kế thừa Dataset
if TORCH_AVAILABLE:
    class TorchEmojiDataset(EmojiDataset, Dataset):
        """
        Torch-compatible version.
        Dùng với DataLoader:
          ds = TorchEmojiDataset.from_processed_file(...)
          loader = DataLoader(ds, batch_size=32, shuffle=True)
        """
        pass


# ─────────────────────────────────────────────
#  3. MODEL BASE INTERFACE
# ─────────────────────────────────────────────

class ModelBase:
    """
    Giao diện chung cho tất cả model thực nghiệm.
    Mỗi người implement cho model của mình.

    Required methods:
        predict(texts, image_paths=None) → list[int]  (0 hoặc 1)
        predict_proba(texts, image_paths=None) → list[float]  (prob of class 1)

    Optional:
        train(train_dataset, val_dataset) → None
        save(path) / load(path)
    """

    name: str = "base"

    def predict(self, texts: list[str], image_paths: list[str] | None = None) -> list[int]:
        """Trả về nhãn dự đoán (0/1) cho danh sách texts."""
        raise NotImplementedError

    def predict_proba(self, texts: list[str], image_paths: list[str] | None = None) -> list[float]:
        """Trả về xác suất class=1 cho danh sách texts."""
        raise NotImplementedError

    def train(self, train_ds: EmojiDataset, val_ds: EmojiDataset) -> None:
        """Optional: fine-tune model."""
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# ─────────────────────────────────────────────
#  4. EVALUATE
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    model_name:    str
    scenario:      str          # "A0" | "A1" | "A2" | "A3"
    split:         str          # "val" | "test"
    accuracy:      float = 0.0
    f1_macro:      float = 0.0
    f1_weighted:   float = 0.0
    auc:           float = 0.0
    confusion:     list  = field(default_factory=list)
    report:        str   = ""
    n_samples:     int   = 0


def evaluate(
    model: ModelBase,
    dataset: EmojiDataset,
    scenario: str = "A1",
    split: str = "test",
) -> EvalResult:
    """
    Đánh giá model trên một dataset + kịch bản ablation.

    Args:
        model:    ModelBase instance
        dataset:  EmojiDataset (đã set đúng text_field)
        scenario: "A0" | "A1" | "A2" | "A3"
        split:    "val" | "test"

    Returns:
        EvalResult
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("Cần scikit-learn: pip install scikit-learn")

    texts  = dataset.get_texts()
    labels = dataset.get_labels()

    # Lấy image paths nếu cần
    image_paths = [dataset[i]["image_path"] for i in range(len(dataset))]

    preds = model.predict(texts, image_paths=image_paths)
    try:
        probas = model.predict_proba(texts, image_paths=image_paths)
        auc = roc_auc_score(labels, probas)
    except (NotImplementedError, Exception):
        probas = None
        auc = 0.0

    acc  = accuracy_score(labels, preds)
    f1m  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1w  = f1_score(labels, preds, average="weighted", zero_division=0)
    cm   = confusion_matrix(labels, preds).tolist()
    rep  = classification_report(labels, preds, zero_division=0)

    return EvalResult(
        model_name  = model.name,
        scenario    = scenario,
        split       = split,
        accuracy    = round(acc,  4),
        f1_macro    = round(f1m,  4),
        f1_weighted = round(f1w,  4),
        auc         = round(auc,  4),
        confusion   = cm,
        report      = rep,
        n_samples   = len(labels),
    )


# ─────────────────────────────────────────────
#  5. RUN ABLATION
# ─────────────────────────────────────────────

def run_ablation(
    model: ModelBase,
    processed_path: str,
    split_ids: dict,                       # {"train": [...], "val": [...], "test": [...]}
    label_field: str = "mm_label",
    image_dir:   str = "data/images",
    scenarios:   list[str] | None = None,  # None = tất cả [A0, A1, A2, A3]
    save_dir:    str = "results",
) -> list[EvalResult]:
    """
    Chạy đủ 4 kịch bản ablation cho một model trên tập test.

    Args:
        model:          ModelBase instance (đã train xong hoặc pretrained)
        processed_path: đường dẫn đến processed_dataset.json
        split_ids:      dict với keys "train", "val", "test"
        label_field:    nhãn dự đoán
        image_dir:      thư mục ảnh
        scenarios:      list kịch bản, mặc định ["A0","A1","A2","A3"]
        save_dir:       thư mục lưu kết quả

    Returns:
        list[EvalResult] — một kết quả cho mỗi kịch bản
    """
    if scenarios is None:
        scenarios = ["A0", "A1", "A2", "A3"]

    results = []

    for sc in scenarios:
        text_field = EmojiDataset.ABLATION_SCENARIOS[sc]
        print(f"  [Ablation {sc}] text_field={text_field} ...")

        test_ds = EmojiDataset.from_processed_file(
            processed_path,
            split_ids=split_ids["test"],
            text_field=text_field,
            label_field=label_field,
            image_dir=image_dir,
        )

        result = evaluate(model, test_ds, scenario=sc, split="test")
        results.append(result)

        print(f"         Acc={result.accuracy:.4f}  F1_macro={result.f1_macro:.4f}  AUC={result.auc:.4f}")

    save_results(results, model.name, save_dir)
    return results


# ─────────────────────────────────────────────
#  6. SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(results: list[EvalResult], model_name: str, save_dir: str = "results") -> None:
    """
    Lưu kết quả ablation:
      results/<model_name>/ablation_results.json
      results/<model_name>/ablation_summary.csv
    """
    out_dir = Path(save_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON chi tiết
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # CSV tóm tắt
    csv_path = out_dir / "ablation_summary.csv"
    fieldnames = ["model_name", "scenario", "split", "accuracy", "f1_macro", "f1_weighted", "auc", "n_samples"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})

    print(f"  [Saved] {json_path}")
    print(f"  [Saved] {csv_path}")


# ─────────────────────────────────────────────
#  7. LOAD SPLIT IDS HELPER
# ─────────────────────────────────────────────

def load_split_ids(save_dir: str = "output") -> dict:
    """Load split_ids.json đã tạo bởi make_splits()."""
    path = Path(save_dir) / "split_ids.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có split_ids.json. Chạy make_splits() trước."
        )
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  8. DEMO / TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: tạo split và load dataset
    print("=== experiment_setup.py — demo ===")
    print()

    # Bước 1: tạo split
    train_ids, val_ids, test_ids = make_splits(
        processed_path="output/processed_dataset.json",
        save_dir="output",
    )

    # Bước 2: load dataset cho kịch bản A1
    test_ds = EmojiDataset.from_processed_file(
        "output/processed_dataset.json",
        split_ids=test_ids,
        text_field="text_A1",
        label_field="mm_label",
    )
    print(f"\nTest dataset (A1): {len(test_ds)} samples")
    print("Sample[0]:", test_ds[0]["text"][:100], "...")
    print("Label:", test_ds[0]["label"])

    # Bước 3: demo dummy model
    class DummyMajorityModel(ModelBase):
        name = "dummy_majority"
        def predict(self, texts, image_paths=None):
            return [1] * len(texts)  # luôn dự đoán class 1
        def predict_proba(self, texts, image_paths=None):
            return [0.6] * len(texts)

    print("\n--- Đánh giá DummyMajorityModel ---")
    model = DummyMajorityModel()
    result = evaluate(model, test_ds, scenario="A1", split="test")
    print(f"Accuracy: {result.accuracy}")
    print(f"F1 macro: {result.f1_macro}")
    print(f"AUC:      {result.auc}")
    print()
    print("✅ Setup chạy thành công. Implement ModelBase cho từng model thực nghiệm.")
