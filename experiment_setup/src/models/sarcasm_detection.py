from __future__ import annotations

from pathlib import Path

from ..data import load_image
from .base import ModelAdapter


class _SarcasmDetectionBaseAdapter(ModelAdapter):
    @property
    def supports_training(self) -> bool:
        return True

    def _resolve_device(self):
        import torch

        requested = self.config.get('training', {}).get('device', 'cuda')
        if requested == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _build_batches(self, records: list[dict], batch_size: int):
        for start in range(0, len(records), batch_size):
            yield records[start:start + batch_size]

    def _label_from_record(self, record: dict) -> int:
        labels = record.get('labels', {})
        mm = int(labels.get('mm_label', 0))
        text = int(labels.get('text_label', 0))
        image = int(labels.get('image_label', 0))

        if mm == 0:
            return 0
        if text == 1 and image == 0:
            return 2
        if text == 0 and image == 1:
            return 3
        return 1

    def _checkpoint_path(self, scenario: str, name: str = 'checkpoint.pt') -> Path:
        return self.run_dir / self.model_name / scenario / name

    def _f1_score(self, labels: list[int], preds: list[int]) -> float:
        from sklearn.metrics import f1_score

        return float(f1_score(labels, preds, average='weighted', zero_division=0))

    def _encode_texts(self, batch: list[dict]):
        cfg = self.config['training']
        texts = [item['text'] for item in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(cfg.get('max_length', 256)),
            return_tensors='pt',
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _encode_images(self, batch: list[dict], scenario: str):
        images = [load_image(item, scenario, self.config) for item in batch]
        encoded = self.image_processor(images=images, return_tensors='pt')
        return encoded['pixel_values'].to(self.device)

    def _targets_tensor(self, batch: list[dict], label_builder) -> 'torch.Tensor':
        return self.torch.tensor([label_builder(item) for item in batch], dtype=self.torch.long, device=self.device)

    def release(self) -> None:
        for attr in (
            'model',
            'tokenizer',
            'image_processor',
            'phase1',
            'phase2',
            'gating',
            'full_model',
            'model1',
            'model2',
            'hierarchical_model',
        ):
            if hasattr(self, attr):
                delattr(self, attr)


class _Approach1Model:
    def __init__(self, text_model_name: str, image_model_name: str, hidden_dim: int, num_classes: int, dropout: float):
        import torch.nn as nn
        from transformers import AutoModel, CLIPModel

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_model = CLIPModel.from_pretrained(image_model_name)
        text_hidden = self.text_encoder.config.hidden_size
        image_hidden = self.image_model.config.vision_config.hidden_size
        self.fusion_proj = nn.Linear(text_hidden + image_hidden, hidden_dim)
        self.fcn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def to(self, device):
        self.text_encoder.to(device)
        self.image_model.to(device)
        self.fusion_proj.to(device)
        self.fcn.to(device)
        return self

    def parameters(self):
        return list(self.text_encoder.parameters()) + list(self.image_model.parameters()) + list(self.fusion_proj.parameters()) + list(self.fcn.parameters())

    def train(self):
        self.text_encoder.train()
        self.image_model.train()
        self.fusion_proj.train()
        self.fcn.train()

    def eval(self):
        self.text_encoder.eval()
        self.image_model.eval()
        self.fusion_proj.eval()
        self.fcn.eval()

    def state_dict(self):
        return {
            'text_encoder': self.text_encoder.state_dict(),
            'image_model': self.image_model.state_dict(),
            'fusion_proj': self.fusion_proj.state_dict(),
            'fcn': self.fcn.state_dict(),
        }

    def load_state_dict(self, state):
        self.text_encoder.load_state_dict(state['text_encoder'])
        self.image_model.load_state_dict(state['image_model'])
        self.fusion_proj.load_state_dict(state['fusion_proj'])
        self.fcn.load_state_dict(state['fcn'])

    def __call__(self, input_ids, attention_mask, pixel_values):
        import torch

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0, :]
        vision_out = self.image_model.vision_model(pixel_values=pixel_values)
        image_emb = vision_out.pooler_output
        fused = self.fusion_proj(torch.cat([text_emb, image_emb], dim=-1))
        return self.fcn(fused)


class SarcasmDetectionMultimodalFusionAdapter(_SarcasmDetectionBaseAdapter):
    def _setup(self):
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer

        model_cfg = self.config['model']
        self.torch = torch
        self.device = self._resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg['pretrained_name'])
        self.image_processor = AutoImageProcessor.from_pretrained(model_cfg['vision_pretrained_name'])
        self.model = _Approach1Model(
            text_model_name=model_cfg['pretrained_name'],
            image_model_name=model_cfg['vision_pretrained_name'],
            hidden_dim=int(model_cfg.get('hidden_dim', 512)),
            num_classes=int(model_cfg.get('num_labels', 4)),
            dropout=float(model_cfg.get('dropout', 0.3)),
        ).to(self.device)

    def train(self, train_records: list[dict], dev_records: list[dict], scenario: str) -> None:
        self._setup()
        cfg = self.config['training']
        criterion = self.torch.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg['learning_rate']),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        best_score = -1.0
        patience = 0
        ckpt_path = self._checkpoint_path(scenario)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in range(int(cfg['epochs'])):
            self.model.train()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                labels = self._targets_tensor(batch, self._label_from_record)
                logits = self.model(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            dev_predictions = self.predict(dev_records, scenario)
            dev_labels = [item['label'] for item in dev_predictions]
            dev_preds = [item['prediction'] for item in dev_predictions]
            dev_score = self._f1_score(dev_labels, dev_preds)
            if dev_score > best_score:
                best_score = dev_score
                patience = 0
                self.torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        if ckpt_path.exists():
            state = self.torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
            ckpt_path = self._checkpoint_path(scenario)
            if ckpt_path.exists():
                state = self.torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state)

        cfg = self.config['training']
        self.model.eval()
        results = []
        split = records[0].get('split', 'unknown') if records else 'unknown'
        total = len(records)

        with self.torch.no_grad():
            processed = 0
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                logits = self.model(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                probs = self.torch.softmax(logits, dim=-1)
                confs, preds = probs.max(dim=-1)
                for item, pred, conf in zip(batch, preds.detach().cpu().tolist(), confs.detach().cpu().tolist()):
                    processed += 1
                    results.append({
                        'id': item['id'],
                        'label': self._label_from_record(item),
                        'prediction': int(pred),
                        'probability': float(conf),
                        'raw_output': str(int(pred)),
                    })
                    if progress_callback is not None:
                        progress_callback(results, processed, total, split)
        return results


class _Phase1TextModel:
    def __init__(self, model_name: str, hidden_dim: int, num_classes: int, dropout: float):
        import torch.nn as nn
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fcn = nn.Sequential(
            nn.Linear(hidden, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def to(self, device):
        self.encoder.to(device)
        self.fcn.to(device)
        return self

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.fcn.parameters())

    def train(self):
        self.encoder.train()
        self.fcn.train()

    def eval(self):
        self.encoder.eval()
        self.fcn.eval()

    def state_dict(self):
        return {'encoder': self.encoder.state_dict(), 'fcn': self.fcn.state_dict()}

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state['encoder'])
        self.fcn.load_state_dict(state['fcn'])

    def __call__(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fcn(cls)


class _Phase2MultimodalModel:
    def __init__(self, text_model_name: str, image_model_name: str, hidden_dim: int, num_classes: int, dropout: float):
        import torch.nn as nn
        from transformers import AutoModel, CLIPModel

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.clip_model = CLIPModel.from_pretrained(image_model_name)
        text_hidden = self.text_encoder.config.hidden_size
        image_hidden = self.clip_model.config.vision_config.hidden_size
        self.fcn = nn.Sequential(
            nn.Linear(text_hidden + image_hidden, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def to(self, device):
        self.text_encoder.to(device)
        self.clip_model.to(device)
        self.fcn.to(device)
        return self

    def parameters(self):
        return list(self.text_encoder.parameters()) + list(self.clip_model.parameters()) + list(self.fcn.parameters())

    def train(self):
        self.text_encoder.train()
        self.clip_model.train()
        self.fcn.train()

    def eval(self):
        self.text_encoder.eval()
        self.clip_model.eval()
        self.fcn.eval()

    def state_dict(self):
        return {
            'text_encoder': self.text_encoder.state_dict(),
            'clip_model': self.clip_model.state_dict(),
            'fcn': self.fcn.state_dict(),
        }

    def load_state_dict(self, state):
        self.text_encoder.load_state_dict(state['text_encoder'])
        self.clip_model.load_state_dict(state['clip_model'])
        self.fcn.load_state_dict(state['fcn'])

    def __call__(self, input_ids, attention_mask, pixel_values):
        import torch

        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_emb = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        fused = torch.cat([text_emb, image_emb], dim=-1)
        return self.fcn(fused)


class _GatingNetwork:
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float):
        import torch.nn as nn

        self.fcn = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def to(self, device):
        self.fcn.to(device)
        return self

    def parameters(self):
        return list(self.fcn.parameters())

    def train(self):
        self.fcn.train()

    def eval(self):
        self.fcn.eval()

    def state_dict(self):
        return {'fcn': self.fcn.state_dict()}

    def load_state_dict(self, state):
        self.fcn.load_state_dict(state['fcn'])

    def __call__(self, prob1, prob2):
        import torch

        return self.fcn(torch.cat([prob1, prob2], dim=-1))


class _Approach2Model:
    def __init__(self, phase1, phase2, gating):
        self.phase1 = phase1
        self.phase2 = phase2
        self.gating = gating

    def to(self, device):
        self.phase1.to(device)
        self.phase2.to(device)
        self.gating.to(device)
        return self

    def train(self):
        self.phase1.train()
        self.phase2.train()
        self.gating.train()

    def eval(self):
        self.phase1.eval()
        self.phase2.eval()
        self.gating.eval()

    def __call__(self, input_ids, attention_mask, pixel_values):
        import torch

        logits1 = self.phase1(input_ids, attention_mask)
        logits2 = self.phase2(input_ids, attention_mask, pixel_values)
        prob1 = torch.softmax(logits1, dim=-1)
        prob2 = torch.softmax(logits2, dim=-1)
        final_logits = self.gating(prob1, prob2)
        return final_logits, logits1, logits2


class SarcasmDetectionStagedGatingAdapter(_SarcasmDetectionBaseAdapter):
    def _phase1_label(self, record: dict) -> int:
        return 1 if self._label_from_record(record) == 2 else 0

    def _phase2_label(self, record: dict) -> int:
        return 1 if self._label_from_record(record) == 3 else 0

    def _setup(self):
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer

        model_cfg = self.config['model']
        self.torch = torch
        self.device = self._resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg['pretrained_name'])
        self.image_processor = AutoImageProcessor.from_pretrained(model_cfg['vision_pretrained_name'])
        self.phase1 = _Phase1TextModel(
            model_name=model_cfg['pretrained_name'],
            hidden_dim=int(model_cfg.get('phase1_hidden_dim', 256)),
            num_classes=2,
            dropout=float(model_cfg.get('dropout', 0.3)),
        ).to(self.device)
        self.phase2 = _Phase2MultimodalModel(
            text_model_name=model_cfg['pretrained_name'],
            image_model_name=model_cfg['vision_pretrained_name'],
            hidden_dim=int(model_cfg.get('phase2_hidden_dim', 512)),
            num_classes=2,
            dropout=float(model_cfg.get('dropout', 0.3)),
        ).to(self.device)
        self.gating = _GatingNetwork(
            hidden_dim=int(model_cfg.get('gating_hidden_dim', 128)),
            num_classes=int(model_cfg.get('num_labels', 4)),
            dropout=float(model_cfg.get('dropout', 0.3)),
        ).to(self.device)
        self.full_model = _Approach2Model(self.phase1, self.phase2, self.gating).to(self.device)

    def _train_single_model(self, model, train_records, dev_records, scenario, checkpoint_name, label_builder, use_images):
        cfg = self.config['training']
        criterion = self.torch.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg['learning_rate']),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        best_score = -1.0
        patience = 0
        ckpt_path = self._checkpoint_path(scenario, checkpoint_name)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in range(int(cfg['epochs'])):
            model.train()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                inputs = self._encode_texts(batch)
                labels = self._targets_tensor(batch, label_builder)
                if use_images:
                    pixel_values = self._encode_images(batch, scenario)
                    logits = model(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                else:
                    logits = model(inputs['input_ids'], inputs['attention_mask'])
                loss = criterion(logits, labels)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            preds = self._predict_single_model(model, dev_records, scenario, label_builder, use_images)
            score = self._f1_score([row['label'] for row in preds], [row['prediction'] for row in preds])
            if score > best_score:
                best_score = score
                patience = 0
                self.torch.save(model.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        if ckpt_path.exists():
            state = self.torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)

    def _predict_single_model(self, model, records, scenario, label_builder, use_images):
        cfg = self.config['training']
        model.eval()
        rows = []
        with self.torch.no_grad():
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self._encode_texts(batch)
                if use_images:
                    pixel_values = self._encode_images(batch, scenario)
                    logits = model(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                else:
                    logits = model(inputs['input_ids'], inputs['attention_mask'])
                probs = self.torch.softmax(logits, dim=-1)
                confs, preds = probs.max(dim=-1)
                for item, pred, conf in zip(batch, preds.detach().cpu().tolist(), confs.detach().cpu().tolist()):
                    rows.append({
                        'id': item['id'],
                        'label': label_builder(item),
                        'prediction': int(pred),
                        'probability': float(conf),
                    })
        return rows

    def _train_gating(self, train_records, dev_records, scenario):
        cfg = self.config['training']
        for parameter in self.phase1.parameters():
            parameter.requires_grad = False
        for parameter in self.phase2.parameters():
            parameter.requires_grad = False

        criterion = self.torch.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.AdamW(
            self.gating.parameters(),
            lr=float(cfg.get('gating_learning_rate', cfg['learning_rate'])),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        best_score = -1.0
        patience = 0
        ckpt_path = self._checkpoint_path(scenario, 'gating_checkpoint.pt')
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in range(int(cfg['epochs'])):
            self.gating.train()
            self.phase1.eval()
            self.phase2.eval()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                labels = self._targets_tensor(batch, self._label_from_record)
                with self.torch.no_grad():
                    logits1 = self.phase1(inputs['input_ids'], inputs['attention_mask'])
                    logits2 = self.phase2(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                    prob1 = self.torch.softmax(logits1, dim=-1)
                    prob2 = self.torch.softmax(logits2, dim=-1)
                final_logits = self.gating(prob1, prob2)
                loss = criterion(final_logits, labels)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.gating.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            preds = self.predict(dev_records, scenario)
            score = self._f1_score([row['label'] for row in preds], [row['prediction'] for row in preds])
            if score > best_score:
                best_score = score
                patience = 0
                self.torch.save(self.gating.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        if ckpt_path.exists():
            state = self.torch.load(ckpt_path, map_location=self.device)
            self.gating.load_state_dict(state)

        for parameter in self.phase1.parameters():
            parameter.requires_grad = True
        for parameter in self.phase2.parameters():
            parameter.requires_grad = True

    def train(self, train_records: list[dict], dev_records: list[dict], scenario: str) -> None:
        self._setup()
        self._train_single_model(self.phase1, train_records, dev_records, scenario, 'phase1_checkpoint.pt', self._phase1_label, False)
        self._train_single_model(self.phase2, train_records, dev_records, scenario, 'phase2_checkpoint.pt', self._phase2_label, True)
        self._train_gating(train_records, dev_records, scenario)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'full_model'):
            self._setup()
            for model_obj, checkpoint_name in (
                (self.phase1, 'phase1_checkpoint.pt'),
                (self.phase2, 'phase2_checkpoint.pt'),
                (self.gating, 'gating_checkpoint.pt'),
            ):
                ckpt_path = self._checkpoint_path(scenario, checkpoint_name)
                if ckpt_path.exists():
                    state = self.torch.load(ckpt_path, map_location=self.device)
                    model_obj.load_state_dict(state)

        cfg = self.config['training']
        self.full_model.eval()
        results = []
        split = records[0].get('split', 'unknown') if records else 'unknown'
        total = len(records)

        with self.torch.no_grad():
            processed = 0
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                logits, _, _ = self.full_model(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                probs = self.torch.softmax(logits, dim=-1)
                confs, preds = probs.max(dim=-1)
                for item, pred, conf in zip(batch, preds.detach().cpu().tolist(), confs.detach().cpu().tolist()):
                    processed += 1
                    results.append({
                        'id': item['id'],
                        'label': self._label_from_record(item),
                        'prediction': int(pred),
                        'probability': float(conf),
                        'raw_output': str(int(pred)),
                    })
                    if progress_callback is not None:
                        progress_callback(results, processed, total, split)
        return results


class _PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        import math
        import torch
        import torch.nn as nn

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def to(self, device):
        self.pe = self.pe.to(device)
        self.dropout.to(device)
        return self

    def __call__(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class _Model1Binary:
    def __init__(self, text_model_name: str, image_model_name: str, hidden_dim: int, dropout: float):
        import torch.nn as nn
        from transformers import AutoModel, CLIPModel

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.clip_model = CLIPModel.from_pretrained(image_model_name)
        text_hidden = self.text_encoder.config.hidden_size
        image_hidden = self.clip_model.config.vision_config.hidden_size
        self.proj = nn.Linear(text_hidden + image_hidden, hidden_dim)
        self.fcn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.text_hidden = text_hidden
        self.image_hidden = image_hidden

    def to(self, device):
        self.text_encoder.to(device)
        self.clip_model.to(device)
        self.proj.to(device)
        self.fcn.to(device)
        return self

    def parameters(self):
        return list(self.text_encoder.parameters()) + list(self.clip_model.parameters()) + list(self.proj.parameters()) + list(self.fcn.parameters())

    def train(self):
        self.text_encoder.train()
        self.clip_model.train()
        self.proj.train()
        self.fcn.train()

    def eval(self):
        self.text_encoder.eval()
        self.clip_model.eval()
        self.proj.eval()
        self.fcn.eval()

    def state_dict(self):
        return {
            'text_encoder': self.text_encoder.state_dict(),
            'clip_model': self.clip_model.state_dict(),
            'proj': self.proj.state_dict(),
            'fcn': self.fcn.state_dict(),
        }

    def load_state_dict(self, state):
        self.text_encoder.load_state_dict(state['text_encoder'])
        self.clip_model.load_state_dict(state['clip_model'])
        self.proj.load_state_dict(state['proj'])
        self.fcn.load_state_dict(state['fcn'])

    def __call__(self, input_ids, attention_mask, pixel_values):
        import torch

        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_emb = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        fused = self.proj(torch.cat([text_emb, image_emb], dim=-1))
        logits = self.fcn(fused)
        return logits, text_emb, image_emb


class _Model2ThreeWay:
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, nhead: int, dropout: float, num_classes: int):
        import torch.nn as nn

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.pos_enc = _PositionalEncoding(d_model=hidden_dim, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.fcn = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def to(self, device):
        self.text_proj.to(device)
        self.image_proj.to(device)
        self.pos_enc.to(device)
        self.cross_attn.to(device)
        self.fcn.to(device)
        return self

    def parameters(self):
        return list(self.text_proj.parameters()) + list(self.image_proj.parameters()) + list(self.cross_attn.parameters()) + list(self.fcn.parameters())

    def train(self):
        self.text_proj.train()
        self.image_proj.train()
        self.cross_attn.train()
        self.fcn.train()

    def eval(self):
        self.text_proj.eval()
        self.image_proj.eval()
        self.cross_attn.eval()
        self.fcn.eval()

    def state_dict(self):
        return {
            'text_proj': self.text_proj.state_dict(),
            'image_proj': self.image_proj.state_dict(),
            'cross_attn': self.cross_attn.state_dict(),
            'fcn': self.fcn.state_dict(),
            'pos_enc_pe': self.pos_enc.pe,
        }

    def load_state_dict(self, state):
        self.text_proj.load_state_dict(state['text_proj'])
        self.image_proj.load_state_dict(state['image_proj'])
        self.cross_attn.load_state_dict(state['cross_attn'])
        self.fcn.load_state_dict(state['fcn'])
        self.pos_enc.pe = state['pos_enc_pe'].to(self.text_proj.weight.device)

    def __call__(self, text_emb, image_emb, pred_probs):
        import torch

        text_proj = self.pos_enc(self.text_proj(text_emb).unsqueeze(1))
        image_proj = self.pos_enc(self.image_proj(image_emb).unsqueeze(1))
        attn_out, _ = self.cross_attn(query=text_proj, key=image_proj, value=image_proj)
        combined = torch.cat([attn_out.squeeze(1), pred_probs], dim=-1)
        return self.fcn(combined)


class _Approach3Model:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def to(self, device):
        self.model1.to(device)
        self.model2.to(device)
        return self

    def eval(self):
        self.model1.eval()
        self.model2.eval()

    def __call__(self, input_ids, attention_mask, pixel_values):
        import torch

        logits1, text_emb, image_emb = self.model1(input_ids, attention_mask, pixel_values)
        prob1 = torch.softmax(logits1, dim=-1)
        logits2 = self.model2(text_emb, image_emb, prob1)
        return logits1, logits2, text_emb, image_emb

    def predict(self, input_ids, attention_mask, pixel_values):
        logits1, logits2, _, _ = self(input_ids, attention_mask, pixel_values)
        pred1 = logits1.argmax(dim=-1)
        pred2 = logits2.argmax(dim=-1)
        final_preds = []
        for p1, p2 in zip(pred1.detach().cpu().tolist(), pred2.detach().cpu().tolist()):
            if p1 == 1:
                final_preds.append(1)
            elif p2 == 0:
                final_preds.append(0)
            elif p2 == 1:
                final_preds.append(2)
            else:
                final_preds.append(3)
        return pred1, pred2, logits1, logits2, final_preds


class SarcasmDetectionHierarchicalCrossAttentionAdapter(_SarcasmDetectionBaseAdapter):
    def _model1_label(self, record: dict) -> int:
        return 1 if self._label_from_record(record) == 1 else 0

    def _model2_label(self, record: dict) -> int:
        mapping = {0: 0, 2: 1, 3: 2}
        return mapping[self._label_from_record(record)]

    def _setup(self):
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer

        model_cfg = self.config['model']
        self.torch = torch
        self.device = self._resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg['pretrained_name'])
        self.image_processor = AutoImageProcessor.from_pretrained(model_cfg['vision_pretrained_name'])
        self.model1 = _Model1Binary(
            text_model_name=model_cfg['pretrained_name'],
            image_model_name=model_cfg['vision_pretrained_name'],
            hidden_dim=int(model_cfg.get('model1_hidden_dim', 512)),
            dropout=float(model_cfg.get('dropout', 0.3)),
        ).to(self.device)
        self.model2 = _Model2ThreeWay(
            text_dim=self.model1.text_hidden,
            image_dim=self.model1.image_hidden,
            hidden_dim=int(model_cfg.get('model2_hidden_dim', 256)),
            nhead=int(model_cfg.get('nhead', 4)),
            dropout=float(model_cfg.get('dropout', 0.3)),
            num_classes=3,
        ).to(self.device)
        self.hierarchical_model = _Approach3Model(self.model1, self.model2).to(self.device)

    def _train_model1(self, train_records, dev_records, scenario):
        cfg = self.config['training']
        criterion = self.torch.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.AdamW(
            self.model1.parameters(),
            lr=float(cfg['learning_rate']),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        best_score = -1.0
        patience = 0
        ckpt_path = self._checkpoint_path(scenario, 'model1_checkpoint.pt')
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in range(int(cfg['epochs'])):
            self.model1.train()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                labels = self._targets_tensor(batch, self._model1_label)
                logits, _, _ = self.model1(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model1.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            preds = self._predict_model1(dev_records, scenario)
            score = self._f1_score([row['label'] for row in preds], [row['prediction'] for row in preds])
            if score > best_score:
                best_score = score
                patience = 0
                self.torch.save(self.model1.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        if ckpt_path.exists():
            state = self.torch.load(ckpt_path, map_location=self.device)
            self.model1.load_state_dict(state)

    def _predict_model1(self, records, scenario):
        cfg = self.config['training']
        self.model1.eval()
        rows = []
        with self.torch.no_grad():
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                logits, _, _ = self.model1(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                probs = self.torch.softmax(logits, dim=-1)
                confs, preds = probs.max(dim=-1)
                for item, pred, conf in zip(batch, preds.detach().cpu().tolist(), confs.detach().cpu().tolist()):
                    rows.append({
                        'id': item['id'],
                        'label': self._model1_label(item),
                        'prediction': int(pred),
                        'probability': float(conf),
                    })
        return rows

    def _train_model2(self, train_records, dev_records, scenario):
        cfg = self.config['training']
        criterion = self.torch.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.AdamW(
            self.model2.parameters(),
            lr=float(cfg.get('model2_learning_rate', cfg['learning_rate'])),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
        )
        best_score = -1.0
        patience = 0
        ckpt_path = self._checkpoint_path(scenario, 'model2_checkpoint.pt')
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        filtered_train = [item for item in train_records if self._label_from_record(item) != 1]
        filtered_dev = [item for item in dev_records if self._label_from_record(item) != 1]

        for parameter in self.model1.parameters():
            parameter.requires_grad = False

        for _ in range(int(cfg['epochs'])):
            self.model1.eval()
            self.model2.train()
            for batch in self._build_batches(filtered_train, int(cfg['batch_size'])):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                labels = self._targets_tensor(batch, self._model2_label)
                with self.torch.no_grad():
                    logits1, text_emb, image_emb = self.model1(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                    prob1 = self.torch.softmax(logits1, dim=-1)
                logits2 = self.model2(text_emb, image_emb, prob1)
                loss = criterion(logits2, labels)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model2.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            preds = self._predict_model2(filtered_dev, scenario)
            score = self._f1_score([row['label'] for row in preds], [row['prediction'] for row in preds])
            if score > best_score:
                best_score = score
                patience = 0
                self.torch.save(self.model2.state_dict(), ckpt_path)
            else:
                patience += 1
                if patience >= int(cfg.get('early_stopping_patience', 2)):
                    break

        for parameter in self.model1.parameters():
            parameter.requires_grad = True

        if ckpt_path.exists():
            state = self.torch.load(ckpt_path, map_location=self.device)
            self.model2.load_state_dict(state)

    def _predict_model2(self, records, scenario):
        cfg = self.config['training']
        self.model1.eval()
        self.model2.eval()
        rows = []
        with self.torch.no_grad():
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                logits1, text_emb, image_emb = self.model1(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                prob1 = self.torch.softmax(logits1, dim=-1)
                logits2 = self.model2(text_emb, image_emb, prob1)
                probs = self.torch.softmax(logits2, dim=-1)
                confs, preds = probs.max(dim=-1)
                for item, pred, conf in zip(batch, preds.detach().cpu().tolist(), confs.detach().cpu().tolist()):
                    rows.append({
                        'id': item['id'],
                        'label': self._model2_label(item),
                        'prediction': int(pred),
                        'probability': float(conf),
                    })
        return rows

    def train(self, train_records: list[dict], dev_records: list[dict], scenario: str) -> None:
        self._setup()
        self._train_model1(train_records, dev_records, scenario)
        self._train_model2(train_records, dev_records, scenario)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'hierarchical_model'):
            self._setup()
            for model_obj, checkpoint_name in (
                (self.model1, 'model1_checkpoint.pt'),
                (self.model2, 'model2_checkpoint.pt'),
            ):
                ckpt_path = self._checkpoint_path(scenario, checkpoint_name)
                if ckpt_path.exists():
                    state = self.torch.load(ckpt_path, map_location=self.device)
                    model_obj.load_state_dict(state)

        cfg = self.config['training']
        self.hierarchical_model.eval()
        results = []
        split = records[0].get('split', 'unknown') if records else 'unknown'
        total = len(records)

        with self.torch.no_grad():
            processed = 0
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                inputs = self._encode_texts(batch)
                pixel_values = self._encode_images(batch, scenario)
                _, _, logits1, logits2, final_preds = self.hierarchical_model.predict(inputs['input_ids'], inputs['attention_mask'], pixel_values)
                probs1 = self.torch.softmax(logits1, dim=-1)
                probs2 = self.torch.softmax(logits2, dim=-1)
                for idx, item in enumerate(batch):
                    processed += 1
                    if final_preds[idx] == 1:
                        confidence = float(probs1[idx].max().detach().cpu().item())
                    else:
                        confidence = float(probs2[idx].max().detach().cpu().item())
                    results.append({
                        'id': item['id'],
                        'label': self._label_from_record(item),
                        'prediction': int(final_preds[idx]),
                        'probability': confidence,
                        'raw_output': str(int(final_preds[idx])),
                    })
                    if progress_callback is not None:
                        progress_callback(results, processed, total, split)
        return results
