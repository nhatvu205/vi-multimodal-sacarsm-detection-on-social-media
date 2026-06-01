from __future__ import annotations

from pathlib import Path

from .dt4mid_arch import DT4MID
from ..data import load_image
from .base import ModelAdapter


class DT4MIDAdapter(ModelAdapter):
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
        from transformers import AutoImageProcessor, AutoTokenizer

        model_cfg = self.config['model']
        self.torch = torch
        self.device = self._resolve_device()
        self.text_only = bool(model_cfg.get('text_only', False))
        self.image_only = bool(model_cfg.get('image_only', False))
        self.tokenizer = None if self.image_only else AutoTokenizer.from_pretrained(model_cfg['pretrained_name'])
        self.image_processor = None if self.text_only else AutoImageProcessor.from_pretrained(model_cfg['vision_pretrained_name'])
        self.model = DT4MID(
            ten_mo_hinh_chu=model_cfg.get('pretrained_name', 'vinai/phobert-base'),
            ten_mo_hinh_anh=model_cfg.get('vision_pretrained_name', 'google/vit-base-patch16-224-in21k'),
            kich_thuoc_an=int(model_cfg.get('projection_dim', 64)),
            kich_thuoc_lop_an=int(model_cfg.get('hidden_dim', 32)),
            ti_le_dropout=float(model_cfg.get('dropout', 0.3)),
            so_nhan=int(model_cfg.get('num_labels', 2)),
            chi_dung_van_ban=self.text_only,
            chi_dung_anh=self.image_only,
            dong_bang_tang_van_ban=int(model_cfg.get('freeze_text_layers', 0)),
            dong_bang_tang_anh=int(model_cfg.get('freeze_vision_layers', 0)),
        ).to(self.device)

    def _build_batches(self, records: list[dict], batch_size: int):
        for start in range(0, len(records), batch_size):
            yield records[start:start + batch_size]

    def _checkpoint_path(self, scenario: str) -> Path:
        return self.run_dir / self.model_name / scenario / 'checkpoint.pt'

    def _load_checkpoint_if_available(self, scenario: str) -> None:
        checkpoint_path = self._checkpoint_path(scenario)
        if not checkpoint_path.exists():
            return
        state = self.torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)

    def _encode_batch(self, batch: list[dict], scenario: str):
        training_cfg = self.config['training']
        text_inputs = None
        pixel_values = None

        if not self.image_only:
            texts = [item['text'] for item in batch]
            text_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=int(training_cfg.get('max_length', 256)),
                return_tensors='pt',
            )

        if not self.text_only:
            images = [load_image(item, scenario, self.config) for item in batch]
            image_inputs = self.image_processor(images=images, return_tensors='pt')
            pixel_values = image_inputs['pixel_values']

        return text_inputs, pixel_values

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
        ckpt_path = self._checkpoint_path(scenario)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        epochs = int(cfg['epochs'])
        batch_size = int(cfg['batch_size'])

        for epoch in range(epochs):
            print(f"[dt4mid-train] scenario={scenario} | epoch {epoch + 1}/{epochs}")
            self.model.train()
            for batch in self._build_batches(train_records, batch_size):
                text_inputs, pixel_values = self._encode_batch(batch, scenario)
                labels = torch.tensor([item['label'] for item in batch], dtype=torch.long, device=self.device)

                model_kwargs = {}
                if text_inputs is not None:
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    model_kwargs['input_ids'] = text_inputs['input_ids']
                    model_kwargs['attention_mask'] = text_inputs['attention_mask']
                    if 'token_type_ids' in text_inputs:
                        model_kwargs['token_type_ids'] = text_inputs['token_type_ids']
                if pixel_values is not None:
                    model_kwargs['pixel_values'] = pixel_values.to(self.device)

                logits = self.model(**model_kwargs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad()

            dev_predictions = self.predict(dev_records, scenario)
            dev_labels = [row['label'] for row in dev_predictions]
            dev_preds = [row['prediction'] for row in dev_predictions]
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
            self._load_checkpoint_if_available(scenario)
        import torch

        cfg = self.config['training']
        self.model.eval()
        results = []
        split = records[0].get('split', 'unknown') if records else 'unknown'
        total = len(records)
        with torch.no_grad():
            processed = 0
            for batch in self._build_batches(records, int(cfg.get('eval_batch_size', cfg['batch_size']))):
                text_inputs, pixel_values = self._encode_batch(batch, scenario)
                model_kwargs = {}
                if text_inputs is not None:
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    model_kwargs['input_ids'] = text_inputs['input_ids']
                    model_kwargs['attention_mask'] = text_inputs['attention_mask']
                    if 'token_type_ids' in text_inputs:
                        model_kwargs['token_type_ids'] = text_inputs['token_type_ids']
                if pixel_values is not None:
                    model_kwargs['pixel_values'] = pixel_values.to(self.device)

                logits = self.model(**model_kwargs)
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                for item, pred, prob in zip(batch, preds, probs):
                    processed += 1
                    results.append({
                        'id': item['id'],
                        'label': item['label'],
                        'prediction': int(pred),
                        'probability': float(prob),
                        'raw_output': str(int(pred)),
                    })
                    if progress_callback is not None:
                        progress_callback(results, processed, total, split)
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'image_processor'):
            del self.image_processor
