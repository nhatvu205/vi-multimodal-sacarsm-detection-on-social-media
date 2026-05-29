# run_phobert.py

from experiment_setup import ModelBase, run_ablation, load_split_ids, EmojiDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ─────────────────────────────────────────────
# 1. DEFINE MODEL 
# ─────────────────────────────────────────────

class PhoBERTModel(ModelBase):
    name = "phobert"

    def __init__(self, model_name="vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[PhoBERT] Device: {self.device}")

    def train(self, train_ds, val_ds, epochs=5, lr=2e-5, batch_size=32):
        from torch.optim import AdamW
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        texts  = train_ds.get_texts()
        labels = train_ds.get_labels()

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct = 0, 0

            for i in range(0, len(texts), batch_size):
                batch_texts  = texts[i:i+batch_size]
                batch_labels = torch.tensor(labels[i:i+batch_size]).to(self.device)

                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt",
                    truncation=True, max_length=256, padding=True
                ).to(self.device)

                outputs = self.model(**inputs, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                correct += (outputs.logits.argmax(-1) == batch_labels).sum().item()

            train_acc = correct / len(texts)

            # Validation sau mỗi epoch
            val_acc = self._validate(val_ds)
            print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss:.3f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    def _validate(self, val_ds):
        self.model.eval()
        texts  = val_ds.get_texts()
        labels = val_ds.get_labels()
        preds  = self.predict(texts)
        correct = sum(p == l for p, l in zip(preds, labels))
        return correct / len(labels)

    def predict(self, texts, image_paths=None):
        return [int(p > 0.5) for p in self.predict_proba(texts)]

    def predict_proba(self, texts, image_paths=None):
        self.model.eval()
        probs = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=256
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0][1].item()
            probs.append(prob)
        return probs


# ─────────────────────────────────────────────
# 2. LOAD SPLIT
# ─────────────────────────────────────────────

split_ids = load_split_ids("output")

# ─────────────────────────────────────────────
# 3. LOAD TRAIN / VAL DATASET (dùng A1 để train)
# ─────────────────────────────────────────────

train_ds = EmojiDataset.from_processed_file(
    "output/processed_dataset.json",
    split_ids["train"],
    text_field="text_A1",
    label_field="mm_label",
)
val_ds = EmojiDataset.from_processed_file(
    "output/processed_dataset.json",
    split_ids["val"],
    text_field="text_A1",
    label_field="mm_label",
)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────

model = PhoBERTModel()
model.train(train_ds, val_ds, epochs=5, lr=2e-5, batch_size=32)

# ─────────────────────────────────────────────
# 5. ABLATION (test trên A0, A1, A2, A3)
# ─────────────────────────────────────────────

run_ablation(
    model,
    "output/processed_dataset.json",
    split_ids,
    label_field="mm_label",
    save_dir="results",
)