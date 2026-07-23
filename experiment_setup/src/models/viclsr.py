from __future__ import annotations

from types import SimpleNamespace

from .classification import TextClassifierAdapter


class _ViCLSRBinaryClassifier:
    """ViCLSR projection head followed by a trainable binary classifier."""

    def __init__(self, backbone, projection, dropout: float = 0.1):
        import torch.nn as nn

        # Keep this wrapper lightweight while exposing the same methods used by
        # TextClassifierAdapter.
        self._module = nn.Module()
        self._module.add_module('backbone', backbone)
        self._module.add_module('projection', projection)
        self._module.add_module('dropout', nn.Dropout(dropout))
        self._module.add_module('classifier', nn.Linear(projection.out_features, 2))

    def __getattr__(self, name):
        if name == '_module':
            return object.__getattribute__(self, name)
        return getattr(self._module, name)

    def to(self, device):
        self._module.to(device)
        return self

    def __call__(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        import torch.nn.functional as F

        backbone_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            **kwargs,
        }
        if token_type_ids is not None:
            backbone_kwargs['token_type_ids'] = token_type_ids
        outputs = self._module.backbone(**backbone_kwargs)
        cls_embedding = outputs.last_hidden_state[:, 0]
        embedding = F.normalize(self._module.projection(cls_embedding), p=2, dim=-1)
        logits = self._module.classifier(self._module.dropout(embedding))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class ViCLSRClassifierAdapter(TextClassifierAdapter):
    """Fine-tune huynhtin/ViCLSR for binary Vietnamese sarcasm detection."""

    def _setup(self):
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, XLMRobertaModel

        self.torch = torch
        self.device = self._resolve_device()
        model_cfg = self.config['model']
        model_name = model_cfg['pretrained_name']

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = XLMRobertaModel.from_pretrained(
            model_name,
            use_safetensors=False,
        )

        hidden_size = int(backbone.config.hidden_size)
        projection = nn.Linear(hidden_size, hidden_size)
        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename=model_cfg.get('checkpoint_filename', 'pytorch_model.bin'),
        )
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:  # torch < 2.0 compatibility
            state_dict = torch.load(checkpoint_path, map_location='cpu')

        weight_key = model_cfg.get('projection_weight_key', 'mlp.dense.weight')
        bias_key = model_cfg.get('projection_bias_key', 'mlp.dense.bias')
        missing = [key for key in (weight_key, bias_key) if key not in state_dict]
        if missing:
            raise KeyError(f'ViCLSR checkpoint is missing projection parameters: {missing}')
        projection.load_state_dict({
            'weight': state_dict[weight_key],
            'bias': state_dict[bias_key],
        })

        self.model = _ViCLSRBinaryClassifier(
            backbone=backbone,
            projection=projection,
            dropout=float(model_cfg.get('classifier_dropout', 0.1)),
        ).to(self.device)
