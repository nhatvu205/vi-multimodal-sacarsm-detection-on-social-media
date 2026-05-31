from __future__ import annotations

from ..data import load_image
from .base import ModelAdapter


class TextClassifierAdapter(ModelAdapter):
    @property
    def supports_training(self) -> bool:
        return True

    def _setup(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = self._resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['pretrained_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model']['pretrained_name'],
            num_labels=2,
        ).to(self.device)

    def _resolve_device(self):
        import torch

        requested = self.config.get('training', {}).get('device', 'cuda')
        if requested == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _build_batches(self, records: list[dict], batch_size: int):
        for start in range(0, len(records), batch_size):
            yield records[start:start + batch_size]

    def train(self, train_records: list[dict], dev_records: list[dict], scenario: str) -> None:
        self._setup()
        import torch
        from sklearn.metrics import f1_score
        from transformers import get_linear_schedule_with_warmup

        cfg = self.config['training']
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg['learning_rate']),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        total_steps = max(1, (len(train_records) + cfg['batch_size'] - 1) // cfg['batch_size']) * int(cfg['epochs'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(total_steps * float(cfg.get('warmup_ratio', 0.0))),
            total_steps,
        )
        best_f1 = -1.0
        patience = 0
        ckpt_path = self.run_dir / self.model_name / scenario / 'checkpoint.pt'
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for _epoch in range(int(cfg['epochs'])):
            self.model.train()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                inputs = self.tokenizer(
                    [x['text'] for x in batch],
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.get('max_length', 256)),
                    return_tensors='pt',
                )
                labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                outputs = self.model(**inputs, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            dev_predictions = self.predict(dev_records, scenario)
            dev_labels = [p['label'] for p in dev_predictions]
            dev_preds = [p['prediction'] for p in dev_predictions]
            dev_f1 = float(f1_score(dev_labels, dev_preds, average='weighted', zero_division=0))
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                patience = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
        import torch

        cfg = self.config['training']
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self.tokenizer(
                    [x['text'] for x in batch],
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.get('max_length', 256)),
                    return_tensors='pt',
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                for item, pred, prob in zip(batch, preds, probs):
                    results.append({
                        'id': item['id'],
                        'label': item['label'],
                        'prediction': int(pred),
                        'probability': float(prob),
                        'raw_output': str(int(pred)),
                    })
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer


class _CLIPVisionBinaryClassifier:
    def __init__(self, model_name: str):
        import torch.nn as nn
        from transformers import CLIPVisionModel

        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.1)

    def to(self, device):
        self.backbone.to(device)
        self.classifier.to(device)
        self.dropout.to(device)
        return self

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.classifier.parameters()) + list(self.dropout.parameters())

    def train(self):
        self.backbone.train()
        self.classifier.train()

    def eval(self):
        self.backbone.eval()
        self.classifier.eval()

    def state_dict(self):
        return {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
        }

    def load_state_dict(self, state):
        self.backbone.load_state_dict(state['backbone'])
        self.classifier.load_state_dict(state['classifier'])

    def __call__(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))


class ImageClassifierAdapter(ModelAdapter):
    @property
    def supports_training(self) -> bool:
        return True

    def _resolve_device(self):
        import torch

        requested = self.config.get('training', {}).get('device', 'cuda')
        if requested == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _setup(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.device = self._resolve_device()
        model_name = self.config['model']['pretrained_name']
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if self.config['model'].get('vision_backbone') == 'clip':
            self.model = _CLIPVisionBinaryClassifier(model_name).to(self.device)
            self._clip_mode = True
        else:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            self._clip_mode = False

    def _build_batches(self, records: list[dict], batch_size: int):
        for start in range(0, len(records), batch_size):
            yield records[start:start + batch_size]

    def _encode_images(self, batch: list[dict], scenario: str):
        images = [load_image(item, scenario, self.config) for item in batch]
        encoded = self.processor(images=images, return_tensors='pt')
        return encoded['pixel_values']

    def train(self, train_records: list[dict], dev_records: list[dict], scenario: str) -> None:
        self._setup()
        import torch
        from sklearn.metrics import f1_score

        cfg = self.config['training']
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg['learning_rate']),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        criterion = torch.nn.CrossEntropyLoss()
        best_score = -1.0
        patience = 0
        ckpt_path = self.run_dir / self.model_name / scenario / 'checkpoint.pt'
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for _epoch in range(int(cfg['epochs'])):
            self.model.train()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                pixel_values = self._encode_images(batch, scenario).to(self.device)
                labels = torch.tensor([x['label'] for x in batch], dtype=torch.long, device=self.device)
                logits = self.model(pixel_values) if self._clip_mode else self.model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            dev_predictions = self.predict(dev_records, scenario)
            dev_labels = [p['label'] for p in dev_predictions]
            dev_preds = [p['prediction'] for p in dev_predictions]
            dev_score = float(f1_score(dev_labels, dev_preds, average='weighted', zero_division=0))
            if dev_score > best_score:
                best_score = dev_score
                patience = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
        import torch

        cfg = self.config['training']
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                pixel_values = self._encode_images(batch, scenario).to(self.device)
                logits = self.model(pixel_values) if self._clip_mode else self.model(pixel_values=pixel_values).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                for item, pred, prob in zip(batch, preds, probs):
                    results.append({
                        'id': item['id'],
                        'label': item['label'],
                        'prediction': int(pred),
                        'probability': float(prob),
                        'raw_output': str(int(pred)),
                    })
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
