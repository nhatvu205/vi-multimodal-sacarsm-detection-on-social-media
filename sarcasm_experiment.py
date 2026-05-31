"""
=============================================================================
  SARCASM DETECTION — FULL EXPERIMENT PIPELINE
  Input : image (PIL) + text (str)
  Output: T (0/1) — text-only label
          I (0/1) — image-only label
          MM (0/1) — multimodal label
=============================================================================

Usage examples
--------------
# Run a single model group
python sarcasm_experiment.py --group text    --ablation all
python sarcasm_experiment.py --group image   --ablation all
python sarcasm_experiment.py --group multi   --ablation all

# Run everything
python sarcasm_experiment.py --group all --ablation all

# Resume from a checkpoint
python sarcasm_experiment.py --group text --checkpoint checkpoints/phobert_s1.pt

Ablation scenarios (4 kịch bản)
---------------------------------
  s1 : raw text + raw image  (no preprocessing, no emoji removal)
  s2 : preprocessed text     + raw image
  s3 : raw text              + preprocessed image
  s4 : preprocessed text     + preprocessed image

For image-only models only s1 is used (1 kịch bản).

Directory layout expected
--------------------------
data/
  train.csv   — columns: text, image_path, label
  val.csv
  test.csv
images/       — image files referenced by image_path

Checkpoint format
-----------------
checkpoints/<model_name>_<scenario>.pt
  contains: {"epoch": int, "state_dict": ..., "optimizer": ...,
             "best_f1": float, "config": dict}
=============================================================================
"""

import os
import re
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# ─── Logging (early — needed by ViSoLex block below) ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log"),
    ],
)
log = logging.getLogger(__name__)

# ─── Emoji processor (local module) ─────────────────────────────────────────
from emoji_processor import process as emoji_process, get_feature_vector, ProcessedText

# ─── ViSoLex (HuggingFace: visolex) ─────────────────────────────────────────
try:
    from visolex.modules.lexical_normalizer import LexicalNormalizer as _VisoLexNorm
    _visolex_instance: Optional[_VisoLexNorm] = None

    def _get_visolex() -> _VisoLexNorm:
        """Lazy singleton — chỉ load model 1 lần."""
        global _visolex_instance
        if _visolex_instance is None:
            log.info("Loading ViSoLex normalizer...")
            _visolex_instance = _VisoLexNorm()
        return _visolex_instance

    def _visolex_normalize(text: str) -> str:
        """Chuẩn hoá từ vựng tiếng Việt bằng ViSoLex.
        Trả về chuỗi đã chuẩn hoá; nếu model fail thì trả nguyên text."""
        try:
            norm = _get_visolex()
            # ViSoLex trả list[str] cho từng token → join lại
            result = norm.normalize(text)
            if isinstance(result, list):
                return " ".join(result)
            return str(result)
        except Exception as exc:
            log.warning(f"ViSoLex normalize failed: {exc}. Using raw text.")
            return text

    VISOLEX_AVAILABLE = True
except ImportError:
    VISOLEX_AVAILABLE = False
    log.warning(
        "ViSoLex không tìm thấy. Cài bằng: pip install visolex\n"
        "  → Scenario s2/s4 sẽ bỏ qua bước chuẩn hoá từ vựng."
    )

    def _visolex_normalize(text: str) -> str:  # noqa: F811
        return text

# ─── Optional imports ────────────────────────────────────────────────────────
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    HFCLIP_AVAILABLE = True
except ImportError:
    HFCLIP_AVAILABLE = False

try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM as QwenBase
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    "data_dir": "data",
    "image_dir": "images",
    "checkpoint_dir": "checkpoints",
    "output_dir": "results",
    "batch_size": 16,
    "max_len": 128,
    "epochs": 5,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224,
    "num_workers": 2,
}

Path(CFG["checkpoint_dir"]).mkdir(exist_ok=True)
Path(CFG["output_dir"]).mkdir(exist_ok=True)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# =============================================================================
#  PREPROCESSING
# =============================================================================

def preprocess_text(
    text: str,
    text_sentiment: Optional[float] = None,
) -> "tuple[str, ProcessedText]":
    """
    Preprocessing pipeline cho tiếng Việt:
      1. emoji_processor.process() — emoticon ASCII → emoji, từ lóng, conflict detection
      2. ViSoLex — chuẩn hoá từ vựng (teen code, viết tắt, v.v.)
      3. Normalize whitespace

    Emoji Unicode được GIỮ NGUYÊN (theo chiến lược trong emoji_processor):
    VLM (LLaVA, Qwen-VL) tự tokenize và hiểu được; BERT-based models sẽ
    nhận được nhãn cảm xúc thay thế (vd: [rất vui]) từ bước emoticon.

    Args:
        text:            Văn bản thô.
        text_sentiment:  Score cảm xúc từ model ngoài (PhoBERT, v.v.)
                         dùng cho conflict detection. None = bỏ qua bước này.

    Returns:
        (processed_str, ProcessedText)  — chuỗi đã xử lý + object features đầy đủ
    """
    text = str(text)

    # Bước 1: emoji_processor — emoticon ASCII + slang + conflict
    emoji_result: ProcessedText = emoji_process(
        text, text_sentiment=text_sentiment
    )
    processed = emoji_result.processed  # emoji Unicode vẫn còn nguyên

    # Bước 2: ViSoLex — chuẩn hoá từ vựng tiếng Việt
    processed = _visolex_normalize(processed)

    # Bước 3: Normalize whitespace
    processed = re.sub(r"\s+", " ", processed).strip()

    return processed, emoji_result


def preprocess_image(image: Image.Image, size: int = CFG["image_size"]) -> Image.Image:
    """Resize + convert to RGB."""
    return image.convert("RGB").resize((size, size))


def get_ablation_inputs(row, scenario: str) -> "tuple[str, Image.Image, Optional[ProcessedText]]":
    """
    Trả về (text, pil_image, emoji_result|None) theo kịch bản ablation.

    Kịch bản:
      s1 — raw text        + raw image      (không preprocessing nào)
      s2 — processed text  + raw image      (ViSoLex + emoji_processor)
      s3 — raw text        + resized image  (chỉ resize/RGB ảnh)
      s4 — processed text  + resized image  (full preprocessing)

    emoji_result chỉ được trả về ở s2/s4 để SarcasmDataset lưu features.
    Ở s1/s3 trả None (raw text, không cần features).

    Notes:
      - text_sentiment trong s2/s4 được bỏ qua (None) ở đây vì PhoBERT
        chưa chạy tại thời điểm preprocessing dataset. Nếu muốn dùng
        conflict detection với sentiment thực, hãy truyền text_sentiment
        từ một model đã inference trước vào preprocess_text() riêng.
    """
    raw_text  = str(row["text"])
    img_path  = Path(CFG["image_dir"]) / row["image_path"]
    raw_image = Image.open(img_path).convert("RGB")

    if scenario == "s1":
        return raw_text, raw_image, None
    elif scenario == "s2":
        proc_text, emoji_result = preprocess_text(raw_text)
        return proc_text, raw_image, emoji_result
    elif scenario == "s3":
        prep_image = preprocess_image(raw_image)
        return raw_text, prep_image, None
    elif scenario == "s4":
        proc_text, emoji_result = preprocess_text(raw_text)
        prep_image = preprocess_image(raw_image)
        return proc_text, prep_image, emoji_result
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


ABLATION_SCENARIOS = {
    "text":  ["s1", "s2", "s3", "s4"],
    "image": ["s1"],               # Image-only: only 1 scenario
    "multi": ["s1", "s2", "s3", "s4"],
}

# =============================================================================
#  DATASET
# =============================================================================

class SarcasmDataset(Dataset):
    """Generic dataset for text+image sarcasm detection."""

    def __init__(self, csv_path: str, scenario: str = "s1",
                 text_transform=None, image_transform=None,
                 mode: str = "mm"):
        """
        mode: 'text' | 'image' | 'mm'
        """
        self.df = pd.read_csv(csv_path)
        self.scenario = scenario
        self.text_transform = text_transform   # tokenizer callable
        self.image_transform = image_transform # torchvision transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])

        # get_ablation_inputs now returns (text, image, emoji_result|None)
        text, image, emoji_result = get_ablation_inputs(row, self.scenario)

        sample = {"label": torch.tensor(label, dtype=torch.long)}

        # ── Store emoji feature vector when preprocessing was applied ────────
        if emoji_result is not None:
            fv = get_feature_vector(emoji_result)
            sample["emoji_features"] = torch.tensor(
                [
                    fv["emoticon_count"],
                    fv["emoticon_max_intensity"],
                    fv["emoticon_polarity_sum"],
                    fv["emoticon_high_intensity"],
                    fv["viet_slang_count"],
                    fv["emoji_count"],
                    fv["unique_emoji_count"],
                    fv["sarcasm_emoji_count"],
                    fv["has_sarcasm_emoji"],
                    fv["sarcasm_score"],
                    fv["conflict_boost"],
                    fv["has_conflict"],
                    fv["sarcasm_score_final"],
                    fv["sentiment_polarity"],
                ],
                dtype=torch.float32,
            )  # shape [14]
        else:
            sample["emoji_features"] = torch.zeros(14, dtype=torch.float32)

        if self.mode in ("text", "mm") and self.text_transform:
            enc = self.text_transform(
                text,
                max_length=CFG["max_len"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            sample["input_ids"]      = enc["input_ids"].squeeze(0)
            sample["attention_mask"] = enc["attention_mask"].squeeze(0)
            if "token_type_ids" in enc:
                sample["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if self.mode in ("image", "mm") and self.image_transform:
            sample["pixel_values"] = self.image_transform(image)

        if self.mode == "mm" and self.text_transform is None:
            # For LLaVA / Qwen: store raw text + PIL image
            sample["raw_text"] = text
            sample["raw_image"] = image

        return sample


# =============================================================================
#  UTILITIES
# =============================================================================

def save_checkpoint(path: str, model, optimizer, epoch: int,
                    best_f1: float, config: dict):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_f1,
        "config": config,
    }, path)
    log.info(f"  ✔ Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location=CFG["device"])
    model.load_state_dict(ckpt["state_dict"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    log.info(f"  ✔ Checkpoint loaded ← {path}  (epoch {ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f})")
    return ckpt


def compute_metrics(labels, preds, probs=None):
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="weighted", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec  = recall_score(labels, preds, average="weighted", zero_division=0)
    auc  = roc_auc_score(labels, probs) if probs is not None else None
    
    return {
        "accuracy": acc, 
        "f1": f1, 
        "precision": prec, 
        "recall": rec, 
        "auc": auc
    }


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        labels = batch["label"].to(device)
        logits = model(batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, return_details: bool = False):
    """
    Đánh giá model trên loader.

    Args:
        return_details: nếu True, trả thêm (all_labels, all_preds, all_probs,
                        all_indices) để dùng cho export_errors().
    """
    model.eval()
    all_labels, all_preds, all_probs, all_indices = [], [], [], []
    global_idx = 0
    for batch in loader:
        labels = batch["label"].to(device)
        logits = model(batch, device)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        bsz    = labels.shape[0]
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_indices.extend(range(global_idx, global_idx + bsz))
        global_idx += bsz
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    if return_details:
        return metrics, all_labels, all_preds, all_probs, all_indices
    return metrics


def export_errors(
    csv_path: str,
    all_labels: list,
    all_preds: list,
    all_probs: list,
    all_indices: list,
    model_name: str,
    scenario: str,
    split: str = "test",
) -> str:
    """
    Xuất ra file JSON các record bị predict sai trên tập `split`.

    Mỗi record trong JSON gồm:
      - index        : vị trí trong tập dữ liệu gốc
      - text         : văn bản gốc
      - image_path   : đường dẫn ảnh
      - true_label   : nhãn thực
      - pred_label   : nhãn model dự đoán
      - prob_sarcasm : xác suất lớp 1 (sarcastic)
      - error_type   : "FP" (false positive) hoặc "FN" (false negative)

    File được lưu tại:
      results/errors_<model_name>_<scenario>_<split>_<timestamp>.json

    Returns:
        Đường dẫn file vừa lưu.
    """
    df = pd.read_csv(csv_path)
    errors = []
    for idx, label, pred, prob in zip(all_indices, all_labels, all_preds, all_probs):
        if int(label) == int(pred):
            continue  # predict đúng → bỏ qua
        row = df.iloc[idx]
        errors.append({
            "index":         int(idx),
            "text":          str(row.get("text", "")),
            "image_path":    str(row.get("image_path", "")),
            "true_label":    int(label),
            "pred_label":    int(pred),
            "prob_sarcasm":  round(float(prob), 4),
            "error_type":    "FP" if (int(pred) == 1 and int(label) == 0) else "FN",
        })
    # Sắp xếp: FP theo prob cao nhất, FN theo prob thấp nhất — dễ phân tích nhất
    errors.sort(key=lambda x: x["prob_sarcasm"], reverse=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        f"{CFG['output_dir']}/errors_{model_name}_{scenario}_{split}_{timestamp}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model":      model_name,
                "scenario":   scenario,
                "split":      split,
                "n_errors":   len(errors),
                "n_total":    len(all_labels),
                "error_rate": round(len(errors) / max(len(all_labels), 1), 4),
                "records":    errors,
            },
            f, indent=2, ensure_ascii=False,
        )
    log.info(
        f"  ⚠ Errors exported ({len(errors)}/{len(all_labels)} wrong) → {out_path}"
    )
    return out_path


def run_training_loop(model_name, model, train_loader, val_loader,
                      test_loader, scenario, device, ckpt_path):
    """Generic fine-tuning loop with checkpointing."""
    optimizer = AdamW(model.parameters(), lr=CFG["lr"])
    total_steps = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # Resume if checkpoint exists
    start_epoch = 0
    best_f1 = 0.0
    if Path(ckpt_path).exists():
        ckpt = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]

    model.to(device)
    for epoch in range(start_epoch, CFG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        log.info(f"  [{model_name}|{scenario}] Epoch {epoch+1}/{CFG['epochs']} "
                 f"loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_f1,
                            {"model": model_name, "scenario": scenario})

    # Final evaluation on test set — bao gồm export errors
    load_checkpoint(ckpt_path, model)
    test_metrics, all_labels, all_preds, all_probs, all_indices = evaluate(
        model, test_loader, device, return_details=True
    )
    log.info(f"  [{model_name}|{scenario}] TEST → {test_metrics}")
    export_errors(
        csv_path    = f"{CFG['data_dir']}/test.csv",
        all_labels  = all_labels,
        all_preds   = all_preds,
        all_probs   = all_probs,
        all_indices = all_indices,
        model_name  = model_name,
        scenario    = scenario,
        split       = "test",
    )
    return test_metrics


# =============================================================================
#  TEXT-ONLY MODELS
# =============================================================================

class TransformerClassifier(nn.Module):
    """Generic BERT-style text classifier."""

    def __init__(self, model_name_or_path: str, num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size  = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, batch, device):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids.to(device)
        out = self.encoder(**kwargs)
        pooled = out.last_hidden_state[:, 0, :]   # [CLS]
        return self.classifier(self.dropout(pooled))


TEXT_MODELS = {
    "roberta-base": "roberta-base",
    "phobert-base": "vinai/phobert-base",
    "mbert":        "bert-base-multilingual-cased",
}


def run_text_model(model_key: str, scenarios=None):
    if scenarios is None:
        scenarios = ABLATION_SCENARIOS["text"]
    model_path = TEXT_MODELS[model_key]
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    results    = {}

    for sc in scenarios:
        log.info(f"\n{'='*60}\n  TEXT | {model_key} | Scenario {sc}\n{'='*60}")

        def make_loader(split):
            ds = SarcasmDataset(
                f"{CFG['data_dir']}/{split}.csv",
                scenario=sc,
                text_transform=tokenizer,
                mode="text",
            )
            return DataLoader(ds, batch_size=CFG["batch_size"],
                              shuffle=(split == "train"),
                              num_workers=CFG["num_workers"])

        train_loader = make_loader("train")
        val_loader   = make_loader("val")
        test_loader  = make_loader("test")

        model    = TransformerClassifier(model_path)
        ckpt_path = f"{CFG['checkpoint_dir']}/{model_key}_{sc}.pt"

        metrics = run_training_loop(
            model_key, model, train_loader, val_loader,
            test_loader, sc, CFG["device"], ckpt_path
        )
        results[sc] = metrics

    return results


# =============================================================================
#  IMAGE-ONLY MODELS
# =============================================================================

class ViTClassifier(nn.Module):
    """
    Generic ViT-based classifier using HuggingFace ViT backbone.
    Works for both ViT-B/32 and CLIP ViT-L/14 (vision encoder).
    """

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import ViTModel, ViTFeatureExtractor
        self.vit = ViTModel.from_pretrained(model_name)
        hidden   = self.vit.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, batch, device):
        pixel_values = batch["pixel_values"].to(device)
        out  = self.vit(pixel_values=pixel_values)
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class CLIPImageClassifier(nn.Module):
    """Use CLIP's vision encoder for image-only classification."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14",
                 num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import CLIPVisionModel
        self.vision = CLIPVisionModel.from_pretrained(clip_model_name)
        hidden = self.vision.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, batch, device):
        pixel_values = batch["pixel_values"].to(device)
        out  = self.vision(pixel_values=pixel_values)
        pooled = out.pooler_output          # [B, hidden]
        return self.classifier(self.dropout(pooled))


IMAGE_MODELS = {
    "vit-b32":        "google/vit-base-patch32-224-in21k",
    "clip-vitl14":    "openai/clip-vit-large-patch14",
}


def get_image_transform(model_key: str):
    """Return a torchvision-style transform for the given vision model."""
    import torchvision.transforms as T
    if "clip" in model_key:
        from transformers import CLIPProcessor
        proc = CLIPProcessor.from_pretrained(IMAGE_MODELS[model_key])
        def transform(img):
            return proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return transform
    else:
        from transformers import ViTFeatureExtractor
        fe = ViTFeatureExtractor.from_pretrained(IMAGE_MODELS[model_key])
        def transform(img):
            return fe(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return transform


def run_image_model(model_key: str, scenarios=None):
    if scenarios is None:
        scenarios = ABLATION_SCENARIOS["image"]  # only s1
    results = {}
    img_transform = get_image_transform(model_key)

    for sc in scenarios:
        log.info(f"\n{'='*60}\n  IMAGE | {model_key} | Scenario {sc}\n{'='*60}")

        def make_loader(split):
            ds = SarcasmDataset(
                f"{CFG['data_dir']}/{split}.csv",
                scenario=sc,
                image_transform=img_transform,
                mode="image",
            )
            return DataLoader(ds, batch_size=CFG["batch_size"],
                              shuffle=(split == "train"),
                              num_workers=CFG["num_workers"])

        train_loader = make_loader("train")
        val_loader   = make_loader("val")
        test_loader  = make_loader("test")

        if model_key == "vit-b32":
            model = ViTClassifier(IMAGE_MODELS[model_key])
        else:
            model = CLIPImageClassifier(IMAGE_MODELS[model_key])

        ckpt_path = f"{CFG['checkpoint_dir']}/{model_key}_{sc}.pt"
        metrics = run_training_loop(
            model_key, model, train_loader, val_loader,
            test_loader, sc, CFG["device"], ckpt_path
        )
        results[sc] = metrics

    return results


# =============================================================================
#  MULTIMODAL MODELS
# =============================================================================

# ── 1. MMSD3 (ZHCMOONWIND/MMSD3.0) ───────────────────────────────────────────

class MMSD3Classifier(nn.Module):
    """
    Wrapper cho kiến trúc MMSD3.0 (Multi-Image Benchmark for Real-World
    Multimodal Sarcasm Detection — ZHCMOONWIND/MMSD3.0).

    Cách tiếp cận:
      - Text encoder : RoBERTa-base
      - Image encoder: CLIP ViT-L/14 (vision tower)
      - Fusion       : cross-attention (text query, image key/value) + concat
      - Classifier   : 2-layer MLP → binary

    Backbone CLIP bị đóng băng; chỉ fine-tune cross-attention + MLP để
    phù hợp với tài nguyên GPU hạn chế.

    Reference: https://github.com/ZHCMOONWIND/MMSD3.0
    """

    CLIP_MODEL  = "openai/clip-vit-large-patch14"
    TEXT_MODEL  = "roberta-base"

    def __init__(self, proj_dim: int = 512, num_heads: int = 8,
                 num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import CLIPVisionModel

        # ── Encoders ──────────────────────────────────────────────────────────
        self.text_enc  = AutoModel.from_pretrained(self.TEXT_MODEL)
        self.image_enc = CLIPVisionModel.from_pretrained(self.CLIP_MODEL)

        # Freeze image encoder (CLIP đã rất mạnh, tiết kiệm VRAM)
        for p in self.image_enc.parameters():
            p.requires_grad = False

        t_dim = self.text_enc.config.hidden_size          # 768
        i_dim = self.image_enc.config.hidden_size         # 1024 for ViT-L

        # ── Projection to common dim ──────────────────────────────────────────
        self.text_proj  = nn.Linear(t_dim, proj_dim)
        self.image_proj = nn.Linear(i_dim, proj_dim)

        # ── Cross-attention: text attends to image ─────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(proj_dim)

        # ── Classifier ────────────────────────────────────────────────────────
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, batch, device):
        # ── Text ──────────────────────────────────────────────────────────────
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        t_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            t_kwargs["token_type_ids"] = token_type_ids.to(device)

        t_out  = self.text_enc(**t_kwargs)
        # Sequence output (all tokens) for cross-attention query
        t_seq  = self.text_proj(t_out.last_hidden_state)      # [B, L, proj_dim]
        t_cls  = t_seq[:, 0, :]                                # [B, proj_dim]

        # ── Image ─────────────────────────────────────────────────────────────
        pixel_values = batch["pixel_values"].to(device)
        i_out  = self.image_enc(pixel_values=pixel_values)
        # Patch sequence as key/value
        i_seq  = self.image_proj(i_out.last_hidden_state)      # [B, P, proj_dim]

        # ── Cross-attention: text CLS queries image patches ───────────────────
        # query: [B, 1, proj_dim]  key/value: [B, P, proj_dim]
        t_query   = t_cls.unsqueeze(1)
        attn_out, _ = self.cross_attn(t_query, i_seq, i_seq)   # [B, 1, proj_dim]
        attn_out  = self.cross_norm(attn_out.squeeze(1))        # [B, proj_dim]

        # ── Fusion + classify ─────────────────────────────────────────────────
        fused = torch.cat([t_cls, attn_out], dim=-1)            # [B, proj_dim*2]
        return self.classifier(self.dropout(fused))


# ── 2. Multi-view CLIP ────────────────────────────────────────────────────────

class MultiViewCLIPClassifier(nn.Module):
    """
    Multi-view CLIP: encode text và ảnh từ nhiều "view" (augmentation hoặc
    multi-crop), sau đó aggregate bằng attention pooling.

    Kiến trúc:
      - Backbone: CLIP (openai/clip-vit-base-patch32) — đóng băng
      - Image views: center crop + 2 random crops (3 views mặc định)
      - Text views: original + version sau preprocessing (2 views)
      - Aggregation: learnable attention pooling trên tất cả views
      - Classifier : Linear → binary

    Điều này giúp model robust hơn với ảnh meme có nhiều vùng ý nghĩa
    và text có nhiều cách diễn đạt khác nhau.
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, n_image_views: int = 3, proj_dim: int = 256,
                 num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        import torchvision.transforms as T

        self.n_image_views = n_image_views
        self.clip      = CLIPModel.from_pretrained(self.CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL)

        # Freeze CLIP backbone
        for p in self.clip.parameters():
            p.requires_grad = False

        embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/32

        # Learnable attention pooling over views
        self.view_attn = nn.Linear(embed_dim, 1)

        # Projection + classifier
        self.text_proj  = nn.Linear(embed_dim, proj_dim)
        self.image_proj = nn.Linear(embed_dim, proj_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim * 2, num_labels)

        # Image augmentation pipeline for extra views
        self._augment = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def _encode_image_views(self, pil_images: list, device) -> torch.Tensor:
        """
        Encode N views per image; return attention-pooled embedding.
        pil_images: list of PIL.Image, length B
        Returns: [B, embed_dim]
        """
        proc = self.processor
        all_view_embeds = []

        for v in range(self.n_image_views):
            if v == 0:
                # View 0: standard CLIP preprocessing (center crop)
                inputs = proc(images=pil_images, return_tensors="pt", padding=True)
                pv = inputs["pixel_values"].to(device)
            else:
                # View v: random augmentation
                tensors = torch.stack([self._augment(img) for img in pil_images])
                pv = tensors.to(device)

            with torch.no_grad():
                img_feat = self.clip.get_image_features(pixel_values=pv)  # [B, E]
            all_view_embeds.append(img_feat)

        # Stack: [B, n_views, E]
        stacked = torch.stack(all_view_embeds, dim=1)
        # Attention pooling
        attn_w = torch.softmax(self.view_attn(stacked), dim=1)  # [B, n_views, 1]
        pooled = (stacked * attn_w).sum(dim=1)                   # [B, E]
        return pooled

    def _encode_text_views(self, texts: list, device) -> torch.Tensor:
        """
        Encode text. texts đã được tokenize nếu gọi từ DataLoader
        bình thường — ở đây ta dùng raw_text từ batch.

        Chỉ 1 view (không augment text để tránh mất nghĩa châm biếm).
        Returns: [B, embed_dim]
        """
        proc = self.processor
        inputs = proc(text=texts, return_tensors="pt",
                      padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            txt_feat = self.clip.get_text_features(**inputs)     # [B, E]
        return txt_feat

    def forward(self, batch, device):
        texts  = batch["raw_text"]   # list[str]
        images = batch["raw_image"]  # list[PIL.Image]

        img_emb = self._encode_image_views(images, device)       # [B, E]
        txt_emb = self._encode_text_views(texts, device)         # [B, E]

        img_proj = self.image_proj(img_emb)                      # [B, proj_dim]
        txt_proj = self.text_proj(txt_emb)                       # [B, proj_dim]

        fused = torch.cat([txt_proj, img_proj], dim=-1)          # [B, proj_dim*2]
        return self.classifier(self.dropout(fused))


# ── Multimodal runner ─────────────────────────────────────────────────────────

MULTI_MODELS = {
    "mmsd3":          MMSD3Classifier,
    "multiview-clip": MultiViewCLIPClassifier,
    "llava":          LLaVAClassifier,
    "qwen-vl":        QwenVLClassifier,
}

# None → model xử lý text/image nội bộ (raw_mode)
# str  → path HuggingFace để khởi tạo tokenizer / image processor ngoài
MULTI_TEXT_MODELS = {
    "mmsd3":          "roberta-base",
    "multiview-clip": None,   # CLIP processor tích hợp trong model
    "llava":          None,   # raw text
    "qwen-vl":        None,   # raw text
}

MULTI_IMAGE_MODELS = {
    "mmsd3":          "openai/clip-vit-large-patch14",
    "multiview-clip": None,   # CLIP processor tích hợp trong model
    "llava":          None,
    "qwen-vl":        None,
}


def run_multi_model(model_key: str, scenarios=None):
    if scenarios is None:
        scenarios = ABLATION_SCENARIOS["multi"]
    results = {}

    for sc in scenarios:
        log.info(f"\n{'='*60}\n  MULTI | {model_key} | Scenario {sc}\n{'='*60}")

        raw_mode = model_key in ("llava", "qwen-vl", "multiview-clip")

        if raw_mode:
            # No tokenizer / image transform; model handles it internally
            def make_loader(split):
                ds = SarcasmDataset(
                    f"{CFG['data_dir']}/{split}.csv",
                    scenario=sc,
                    mode="mm",
                )
                loader = DataLoader(
                    ds,
                    batch_size=4 if "llava" in model_key else 8,
                    shuffle=(split == "train"),
                    num_workers=0,
                    collate_fn=lambda b: {
                        "raw_text":     [x["raw_text"]  for x in b],
                        "raw_image":    [x["raw_image"] for x in b],
                        "label":        torch.stack([x["label"] for x in b]),
                        "emoji_features": torch.stack([x["emoji_features"] for x in b]),
                    },
                )
                return loader
        else:
            text_model_path  = MULTI_TEXT_MODELS[model_key]
            image_model_path = MULTI_IMAGE_MODELS[model_key]

            # mmsd3: RoBERTa tokenizer + CLIP image processor (ViT-L/14)
            tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            from transformers import CLIPProcessor as _CP
            _clip_proc = _CP.from_pretrained(image_model_path)
            image_transform = (
                lambda img: _clip_proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            )

            def make_loader(split):
                ds = SarcasmDataset(
                    f"{CFG['data_dir']}/{split}.csv",
                    scenario=sc,
                    text_transform=tokenizer,
                    image_transform=image_transform,
                    mode="mm",
                )
                return DataLoader(ds, batch_size=CFG["batch_size"],
                                  shuffle=(split == "train"),
                                  num_workers=CFG["num_workers"])

        train_loader = make_loader("train")
        val_loader   = make_loader("val")
        test_loader  = make_loader("test")

        model     = MULTI_MODELS[model_key]()
        ckpt_path = f"{CFG['checkpoint_dir']}/{model_key}_{sc}.pt"

        metrics = run_training_loop(
            model_key, model, train_loader, val_loader,
            test_loader, sc, CFG["device"], ckpt_path
        )
        results[sc] = metrics

    return results


# =============================================================================
#  RESULTS AGGREGATION
# =============================================================================

def save_results(all_results: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = f"{CFG['output_dir']}/results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"\n📊 All results saved → {out_path}")

    # Print summary table (Đã tăng chiều dài từ 70 lên 90 để vừa 2 cột mới)
    print("\n" + "="*90)
    print(f"{'Model':<20} {'Scenario':<8} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print("-"*90)
    for group, models in all_results.items():
        for model_name, scenarios in models.items():
            for sc, metrics in scenarios.items():
                auc = f"{metrics['auc']:.4f}" if metrics['auc'] else "  N/A  "
                print(f"{model_name:<20} {sc:<8} "
                      f"{metrics['f1']:>8.4f} "
                      f"{metrics['accuracy']:>8.4f} "
                      f"{metrics['precision']:>8.4f} "
                      f"{metrics['recall']:>8.4f} "
                      f"{auc:>8}")
    print("="*90)

# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sarcasm Detection Experiment Pipeline")
    parser.add_argument("--group", choices=["text", "image", "multi", "all"],
                        default="all", help="Model group to run")
    parser.add_argument("--ablation", choices=["s1", "s2", "s3", "s4", "all"],
                        default="all", help="Which ablation scenario(s) to run")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model keys to run (e.g. phobert-base llava)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    parser.add_argument("--data_dir", type=str, default=CFG["data_dir"])
    parser.add_argument("--image_dir", type=str, default=CFG["image_dir"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Update global config from args
    CFG.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "data_dir": args.data_dir,
        "image_dir": args.image_dir,
    })

    # Determine scenarios
    if args.ablation == "all":
        scenarios_text  = ABLATION_SCENARIOS["text"]
        scenarios_image = ABLATION_SCENARIOS["image"]
        scenarios_multi = ABLATION_SCENARIOS["multi"]
    else:
        scenarios_text  = [args.ablation]
        scenarios_image = [args.ablation] if args.ablation == "s1" else []
        scenarios_multi = [args.ablation]

    all_results = {"text": {}, "image": {}, "multi": {}}

    # ── TEXT GROUP ────────────────────────────────────────────────────────────
    if args.group in ("text", "all"):
        models_to_run = args.models if args.models else list(TEXT_MODELS.keys())
        for mk in models_to_run:
            if mk not in TEXT_MODELS:
                continue
            log.info(f"\n▶ Starting TEXT model: {mk}")
            all_results["text"][mk] = run_text_model(mk, scenarios_text)

    # ── IMAGE GROUP ───────────────────────────────────────────────────────────
    if args.group in ("image", "all"):
        models_to_run = args.models if args.models else list(IMAGE_MODELS.keys())
        for mk in models_to_run:
            if mk not in IMAGE_MODELS:
                continue
            log.info(f"\n▶ Starting IMAGE model: {mk}")
            all_results["image"][mk] = run_image_model(mk, scenarios_image)

    # ── MULTIMODAL GROUP ──────────────────────────────────────────────────────
    if args.group in ("multi", "all"):
        models_to_run = args.models if args.models else list(MULTI_MODELS.keys())
        for mk in models_to_run:
            if mk not in MULTI_MODELS:
                continue
            log.info(f"\n▶ Starting MULTI model: {mk}")
            all_results["multi"][mk] = run_multi_model(mk, scenarios_multi)

    save_results(all_results)


if __name__ == "__main__":
    main()
