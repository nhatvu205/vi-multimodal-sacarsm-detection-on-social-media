from __future__ import annotations

from pathlib import Path

from ..data import load_image
from .base import ModelAdapter


# Minimal MMSD3.0 core blocks adapted from the official repository's src/model.py
# Source: https://github.com/ZHCMOONWIND/MMSD3.0 and
# https://raw.githubusercontent.com/ZHCMOONWIND/MMSD3.0/main/src/model.py
class SequentialModeling:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _SequentialModeling(nn.Module):
            def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
                super().__init__()
                self.d_model = d_model
                self.d_state = d_state
                self.d_conv = d_conv
                self.d_inner = int(expand * d_model)
                self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    kernel_size=d_conv,
                    bias=True,
                    padding=d_conv - 1,
                    groups=self.d_inner,
                )
                self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
                self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
                A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
                self.A_log = nn.Parameter(torch.log(A))
                self.D = nn.Parameter(torch.ones(self.d_inner))
                self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                batch, seqlen, _dim = x.shape
                residual = x
                x = self.norm(x)
                xz = self.in_proj(x)
                x, z = xz.chunk(2, dim=-1)
                x = x.transpose(1, 2)
                x = self.conv1d(x)[..., :seqlen]
                x = x.transpose(1, 2)
                x = F.silu(x)
                x_dbl = self.x_proj(x)
                _delta_unused, B = x_dbl.chunk(2, dim=-1)
                delta = F.softplus(self.dt_proj(x))
                A_local = -torch.exp(self.A_log.float())
                y = self._selective_scan(x, delta, A_local, B)
                y = y * F.silu(z)
                output = self.out_proj(y)
                return output + residual

            def _selective_scan(self, u, delta, A_local, B):
                batch, seqlen, d_inner = u.shape
                d_state = A_local.shape[1]
                h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
                outputs = []
                for i in range(seqlen):
                    u_i = u[:, i, :]
                    delta_i = delta[:, i, :]
                    B_i = B[:, i, :].unsqueeze(1)
                    dA = torch.exp(delta_i.unsqueeze(-1) * A_local.unsqueeze(0))
                    dB = delta_i.unsqueeze(-1) * B_i
                    h = h * dA + u_i.unsqueeze(-1) * dB
                    outputs.append(torch.sum(h, dim=-1))
                y = torch.stack(outputs, dim=1)
                return y + u * self.D.unsqueeze(0).unsqueeze(0)

        return _SequentialModeling(*args, **kwargs)


class DualStageBridgeModule:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class _DualStageBridgeModule(nn.Module):
            def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
                super().__init__()
                self.text_mamba = SequentialModeling(d_model, d_state, d_conv, expand)
                self.vision_mamba = SequentialModeling(d_model, d_state, d_conv, expand)
                self.pre_text_cross_attn = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
                self.pre_vision_cross_attn = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
                self.post_text_cross_attn = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
                self.post_vision_cross_attn = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
                self.pre_text_gate = nn.Linear(d_model, d_model)
                self.pre_vision_gate = nn.Linear(d_model, d_model)
                self.post_text_gate = nn.Linear(d_model, d_model)
                self.post_vision_gate = nn.Linear(d_model, d_model)
                self.pre_text_norm = nn.LayerNorm(d_model)
                self.pre_vision_norm = nn.LayerNorm(d_model)
                self.post_text_norm = nn.LayerNorm(d_model)
                self.post_vision_norm = nn.LayerNorm(d_model)

            def forward(self, text_embeddings, vision_embeddings):
                text_cross_pre, _ = self.pre_text_cross_attn(
                    query=text_embeddings,
                    key=vision_embeddings,
                    value=vision_embeddings,
                )
                vision_cross_pre, _ = self.pre_vision_cross_attn(
                    query=vision_embeddings,
                    key=text_embeddings,
                    value=text_embeddings,
                )
                text_gate_pre = torch.sigmoid(self.pre_text_gate(text_embeddings))
                vision_gate_pre = torch.sigmoid(self.pre_vision_gate(vision_embeddings))
                text_input = self.pre_text_norm(text_embeddings + text_gate_pre * text_cross_pre)
                vision_input = self.pre_vision_norm(vision_embeddings + vision_gate_pre * vision_cross_pre)
                text_mamba_out = self.text_mamba(text_input)
                vision_mamba_out = self.vision_mamba(vision_input)
                text_cross_post, _ = self.post_text_cross_attn(
                    query=text_mamba_out,
                    key=vision_mamba_out,
                    value=vision_mamba_out,
                )
                vision_cross_post, _ = self.post_vision_cross_attn(
                    query=vision_mamba_out,
                    key=text_mamba_out,
                    value=text_mamba_out,
                )
                text_gate_post = torch.sigmoid(self.post_text_gate(text_mamba_out))
                vision_gate_post = torch.sigmoid(self.post_vision_gate(vision_mamba_out))
                enhanced_text = self.post_text_norm(text_mamba_out + text_gate_post * text_cross_post)
                enhanced_vision = self.post_vision_norm(vision_mamba_out + vision_gate_post * vision_cross_post)
                return enhanced_text, enhanced_vision

        return _DualStageBridgeModule(*args, **kwargs)


class CIRMWrapper:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModel, ViTModel

        class _CIRMWrapper(nn.Module):
            def __init__(
                self,
                text_model_name: str,
                vision_model_name: str,
                embedding_dim: int = 768,
                fusion_dim: int = 512,
                freeze_text_layers: int = 4,
                freeze_vision_layers: int = 4,
                num_labels: int = 2,
                dropout: float = 0.1,
            ):
                super().__init__()
                self.text_encoder = AutoModel.from_pretrained(text_model_name)
                self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
                self._freeze_transformer_layers(self.text_encoder, freeze_text_layers)
                self._freeze_transformer_layers(self.vision_encoder, freeze_vision_layers)
                self.text_proj = nn.Sequential(
                    nn.Linear(self.text_encoder.config.hidden_size, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.vision_proj = nn.Sequential(
                    nn.Linear(self.vision_encoder.config.hidden_size, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.dsbm = DualStageBridgeModule(d_model=embedding_dim)
                self.text_projection = nn.Linear(embedding_dim, embedding_dim)
                self.image_projection = nn.Linear(embedding_dim, embedding_dim)
                self.text_img_relevance = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim, embedding_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embedding_dim // 2, 1),
                )
                self.text_mapper = nn.Linear(embedding_dim, fusion_dim)
                self.mamba_mapper = nn.Linear(embedding_dim, fusion_dim)
                self.cross_modal_mapper = nn.Linear(embedding_dim, fusion_dim)
                self.fusion_layer = nn.Sequential(
                    nn.Linear(fusion_dim * 3, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim, fusion_dim),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(fusion_dim // 2, fusion_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim // 4, num_labels),
                )
                self._init_weights()

            @staticmethod
            def _freeze_transformer_layers(model: nn.Module, n: int):
                layers = None
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                    layers = model.encoder.layer
                elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                    layers = model.encoder.layers
                if layers is None or n <= 0:
                    return
                for idx, layer in enumerate(layers):
                    if idx < n:
                        for param in layer.parameters():
                            param.requires_grad = False

            def _init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.0)
                nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=0.02)
                if self.classifier[-1].bias is not None:
                    nn.init.constant_(self.classifier[-1].bias, 0.0)

            def forward(self, input_ids, attention_mask, pixel_values, token_type_ids=None):
                text_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                if token_type_ids is not None:
                    text_kwargs['token_type_ids'] = token_type_ids
                text_hidden = self.text_encoder(**text_kwargs).last_hidden_state
                vision_hidden = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
                text_seq = self.text_proj(text_hidden)
                vision_seq = self.vision_proj(vision_hidden)
                enhanced_text, enhanced_vision = self.dsbm(text_seq, vision_seq)
                text_agg = enhanced_text.mean(dim=1)
                vision_agg = enhanced_vision.mean(dim=1)
                text_cls = text_seq.mean(dim=1)
                vision_pool = vision_seq.mean(dim=1)
                t_proj = F.normalize(self.text_projection(text_cls), p=2, dim=-1)
                v_proj = F.normalize(self.image_projection(vision_pool), p=2, dim=-1)
                cos_sim = (t_proj * v_proj).sum(-1, keepdim=True)
                learned = self.text_img_relevance(torch.cat([text_cls, vision_pool], dim=-1))
                rel_w = torch.sigmoid(0.6 * cos_sim + 0.4 * learned)
                cross_feat = text_cls * vision_pool * rel_w
                fused = torch.cat([
                    self.text_mapper(text_agg),
                    self.mamba_mapper(vision_agg),
                    self.cross_modal_mapper(cross_feat),
                ], dim=-1)
                fused = self.fusion_layer(fused)
                return self.classifier(fused)

        return _CIRMWrapper(*args, **kwargs)


class CIRMAdapter(ModelAdapter):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg['pretrained_name'])
        self.image_processor = AutoImageProcessor.from_pretrained(model_cfg['vision_pretrained_name'])
        self.model = CIRMWrapper(
            text_model_name=model_cfg['pretrained_name'],
            vision_model_name=model_cfg['vision_pretrained_name'],
            embedding_dim=int(model_cfg.get('embedding_dim', 768)),
            fusion_dim=int(model_cfg.get('fusion_dim', 512)),
            freeze_text_layers=int(model_cfg.get('freeze_text_layers', 4)),
            freeze_vision_layers=int(model_cfg.get('freeze_vision_layers', 4)),
            dropout=float(model_cfg.get('dropout', 0.1)),
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
        texts = [item['text'] for item in batch]
        images = [load_image(item, scenario, self.config) for item in batch]
        training_cfg = self.config['training']
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(training_cfg.get('max_length', 256)),
            return_tensors='pt',
        )
        image_inputs = self.image_processor(images=images, return_tensors='pt')
        return text_inputs, image_inputs['pixel_values']

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

        for _epoch in range(int(cfg['epochs'])):
            self.model.train()
            for batch in self._build_batches(train_records, int(cfg['batch_size'])):
                text_inputs, pixel_values = self._encode_batch(batch, scenario)
                labels = torch.tensor([item['label'] for item in batch], dtype=torch.long, device=self.device)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                pixel_values = pixel_values.to(self.device)
                logits = self.model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    pixel_values=pixel_values,
                    token_type_ids=text_inputs.get('token_type_ids'),
                )
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
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                pixel_values = pixel_values.to(self.device)
                logits = self.model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    pixel_values=pixel_values,
                    token_type_ids=text_inputs.get('token_type_ids'),
                )
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
