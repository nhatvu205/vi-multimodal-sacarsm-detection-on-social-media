# run_phobert.py
from experiment_setup import ModelBase, run_ablation, load_split_ids
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PhoBERTModel(ModelBase):
    name = "phobert"

    def __init__(self, model_name="vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, texts, image_paths=None):
        return [int(p > 0.5) for p in self.predict_proba(texts)]

    def predict_proba(self, texts, image_paths=None):
        self.model.eval()
        probs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0][1].item()
            probs.append(prob)
        return probs

split_ids = load_split_ids("output")
model = PhoBERTModel()
run_ablation(model, "output/processed_dataset.json", split_ids, save_dir="results")