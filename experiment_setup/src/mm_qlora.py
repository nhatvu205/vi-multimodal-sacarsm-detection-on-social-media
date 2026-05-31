from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import build_records, build_run_dir, load_cached_splits, prepare_cache
from .io_utils import save_json, save_yaml
from .prompting import get_system_prompt, get_user_prompt


@dataclass
class MMTrainingExample:
    text: str
    image_path: str
    target_text: str


class MMQLoRADataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        labels = record['labels']
        target_text = f"({labels['text_label']},{labels['image_label']},{labels['mm_label']})"
        return {
            'id': record['id'],
            'text': record['text'],
            'image_path': record['image_path'],
            'target_text': target_text,
        }


class MMQLoRADataCollator:
    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'

    def _load_image(self, image_path: str):
        from PIL import Image

        return Image.open(image_path).convert('RGB')

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        images = [self._load_image(x['image_path']) for x in features]
        prompt_texts = []
        full_texts = []

        for item in features:
            prompt_config = {'prompt': self.processor_prompt}
            messages = []
            system_prompt = get_system_prompt(prompt_config)
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': system_prompt},
                    ],
                })
            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': get_user_prompt({'text': item['text']}, prompt_config)},
                ],
            })
            assistant_messages = messages + [
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': item['target_text']},
                    ],
                }
            ]

            prompt_texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                self.processor.apply_chat_template(assistant_messages, tokenize=False, add_generation_prompt=False)
            )

        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        labels = full_inputs['input_ids'].clone()
        labels[full_inputs['attention_mask'] == 0] = -100
        prompt_lengths = prompt_inputs['attention_mask'].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            labels[idx, : int(prompt_len)] = -100

        batch = dict(full_inputs)
        batch['labels'] = labels
        return batch

    @property
    def processor_prompt(self) -> dict:
        return self._prompt_config

    @processor_prompt.setter
    def processor_prompt(self, value: dict) -> None:
        self._prompt_config = value


def _resolve_dtype(name: str):
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    return mapping.get(name, torch.float16)


def _build_quantization_config(finetune_cfg: dict):
    if not finetune_cfg.get('use_4bit', True):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=finetune_cfg.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_use_double_quant=bool(finetune_cfg.get('bnb_4bit_use_double_quant', True)),
        bnb_4bit_compute_dtype=_resolve_dtype(finetune_cfg.get('bnb_4bit_compute_dtype', 'float16')),
    )


def _load_mm_model_and_processor(config: dict):
    family = config['model']['family']
    pretrained_name = config['model']['pretrained_name']
    finetune_cfg = config['finetune']
    quantization_config = _build_quantization_config(finetune_cfg)
    torch_dtype = _resolve_dtype(finetune_cfg.get('torch_dtype', 'float16'))

    common_kwargs = {
        'device_map': finetune_cfg.get('device_map', 'auto'),
        'torch_dtype': torch_dtype,
    }
    if quantization_config is not None:
        common_kwargs['quantization_config'] = quantization_config

    if family == 'llava_next_generative':
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        processor = LlavaNextProcessor.from_pretrained(pretrained_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(pretrained_name, **common_kwargs)
    elif family == 'qwen3_vl_generative':
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(pretrained_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained_name, **common_kwargs)
    else:
        raise KeyError(f'Unsupported fine-tune family: {family}')

    return model, processor


def _freeze_non_lora_parts(model, finetune_cfg: dict):
    if finetune_cfg.get('freeze_vision_tower', True):
        for name, param in model.named_parameters():
            if any(token in name for token in ('vision_tower', 'visual', 'vision_model')):
                param.requires_grad = False
    if not finetune_cfg.get('train_mm_projector', True):
        for name, param in model.named_parameters():
            if 'multi_modal_projector' in name or 'mm_projector' in name:
                param.requires_grad = False


def _prepare_lora_model(model, config: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    finetune_cfg = config['finetune']
    model.config.use_cache = False
    if finetune_cfg.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    _freeze_non_lora_parts(model, finetune_cfg)

    lora_config = LoraConfig(
        r=int(finetune_cfg.get('lora_r', 8)),
        lora_alpha=int(finetune_cfg.get('lora_alpha', 16)),
        lora_dropout=float(finetune_cfg.get('lora_dropout', 0.05)),
        bias='none',
        target_modules=list(finetune_cfg.get('target_modules', ['q_proj', 'v_proj'])),
        task_type='CAUSAL_LM',
    )
    return get_peft_model(model, lora_config)


def _build_training_args(output_dir: Path, finetune_cfg: dict):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(finetune_cfg.get('num_train_epochs', 2)),
        per_device_train_batch_size=int(finetune_cfg.get('per_device_train_batch_size', 1)),
        per_device_eval_batch_size=int(finetune_cfg.get('per_device_eval_batch_size', 1)),
        gradient_accumulation_steps=int(finetune_cfg.get('gradient_accumulation_steps', 8)),
        learning_rate=float(finetune_cfg.get('learning_rate', 2e-4)),
        weight_decay=float(finetune_cfg.get('weight_decay', 0.0)),
        warmup_ratio=float(finetune_cfg.get('warmup_ratio', 0.03)),
        logging_steps=int(finetune_cfg.get('logging_steps', 10)),
        save_strategy=str(finetune_cfg.get('save_strategy', 'epoch')),
        eval_strategy=str(finetune_cfg.get('eval_strategy', 'epoch')),
        save_total_limit=int(finetune_cfg.get('save_total_limit', 2)),
        bf16=bool(finetune_cfg.get('bf16', False)),
        fp16=bool(finetune_cfg.get('fp16', True)),
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        load_best_model_at_end=False,
        label_names=['labels'],
    )


def finetune_multimodal_model(config: dict, scenario: str | None = None) -> Path:
    from transformers import Trainer

    run_dir = build_run_dir(config)
    prepare_cache(config, run_dir)
    cached = load_cached_splits(run_dir)

    finetune_cfg = config.setdefault('finetune', {})
    scenario = scenario or finetune_cfg.get('scenario', 's1')
    train_records = build_records(cached['train'], scenario, config)
    dev_records = build_records(cached['dev'], scenario, config)

    output_dir = run_dir / finetune_cfg.get('output_subdir', 'finetune') / config['model']['key'] / scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = _load_mm_model_and_processor(config)
    model = _prepare_lora_model(model, config)

    train_dataset = MMQLoRADataset(train_records)
    eval_dataset = MMQLoRADataset(dev_records)
    collator = MMQLoRADataCollator(processor, max_length=int(finetune_cfg.get('max_length', 1024)))
    collator.processor_prompt = config.get('prompt', {})

    trainer = Trainer(
        model=model,
        args=_build_training_args(output_dir, finetune_cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()

    adapter_dir = output_dir / 'adapter'
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(output_dir / 'processor')
    save_yaml(output_dir / 'resolved_finetune_config.yaml', {k: v for k, v in config.items() if k != '_meta'})
    save_json(
        output_dir / 'artifacts.json',
        {
            'scenario': scenario,
            'adapter_path': str(adapter_dir),
            'processor_path': str(output_dir / 'processor'),
            'model_name': config['model']['pretrained_name'],
            'family': config['model']['family'],
        },
    )
    return adapter_dir
