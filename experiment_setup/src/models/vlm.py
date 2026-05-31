from __future__ import annotations

import gc
import time
from datetime import datetime, timedelta
from pathlib import Path

from ..data import load_image
from ..prompting import build_prompt, get_system_prompt, get_user_prompt, parse_combo_prediction
from .base import ModelAdapter


def _resolve_dtype(torch_module, value: str):
    mapping = {
        'float16': torch_module.float16,
        'fp16': torch_module.float16,
        'bfloat16': torch_module.bfloat16,
        'bf16': torch_module.bfloat16,
        'float32': torch_module.float32,
        'fp32': torch_module.float32,
        'auto': None,
        None: None,
    }
    return mapping.get(value, None)


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, 'device', 'cpu')


def _maybe_load_adapter(model, config: dict):
    adapter_path = config.get('model', {}).get('adapter_path')
    if not adapter_path:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


def _set_pad_token(processor=None, tokenizer=None, model=None):
    tok = tokenizer
    if tok is None and processor is not None and hasattr(processor, 'tokenizer'):
        tok = processor.tokenizer
    if tok is None:
        return None

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    pad_token_id = tok.pad_token_id
    if model is not None and pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            model.generation_config.pad_token_id = pad_token_id
    return pad_token_id


def _log_sample_progress(model_name: str, scenario: str, split: str, index: int, total: int, start_time: float):
    elapsed_seconds = max(time.monotonic() - start_time, 0.0)
    average_seconds = elapsed_seconds / max(index, 1)
    remaining_seconds = max(int(average_seconds * max(total - index, 0)), 0)
    eta = str(timedelta(seconds=remaining_seconds))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[infer] {timestamp} | {model_name} | {scenario} | {split} | sample {index}/{total} | eta {eta}')


def _cleanup_cuda(torch_module) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.ipc_collect()


class LlavaGenerativeAdapter(ModelAdapter):
    def _setup(self):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        dtype = _resolve_dtype(torch, self.config.get('inference', {}).get('torch_dtype', 'auto'))
        model_kwargs = {'device_map': self.config.get('inference', {}).get('device_map', 'auto')}
        if dtype is not None:
            model_kwargs['torch_dtype'] = dtype
        self.processor = AutoProcessor.from_pretrained(self.config['model']['pretrained_name'])
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config['model']['pretrained_name'],
            **model_kwargs,
        )
        self.model = _maybe_load_adapter(self.model, self.config)
        self.pad_token_id = _set_pad_token(processor=self.processor, model=self.model)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
        import torch

        results = []
        model_device = _get_model_device(self.model)
        total = len(records)
        split = records[0].get('split', 'unknown') if records else 'unknown'
        start_time = time.monotonic()
        with torch.inference_mode():
            for idx, record in enumerate(records, start=1):
                _log_sample_progress(self.model_name, scenario, split, idx, total, start_time)
                image = load_image(record, scenario, self.config)
                prompt = build_prompt(record, self.config, few_shot_examples=few_shot_examples)
                messages = [{
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt},
                    ],
                }]
                if hasattr(self.processor, 'apply_chat_template'):
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                else:
                    text = f'USER: <image>\n{prompt}\nASSISTANT:'
                inputs = self.processor(images=image, text=text, return_tensors='pt')
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.config.get('inference', {}).get('max_new_tokens', 24)),
                    do_sample=False,
                    pad_token_id=self.pad_token_id,
                )
                decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                combo, raw_output = parse_combo_prediction(decoded)
                results.append({
                    'id': record['id'],
                    'label': record['label'],
                    'prediction': int(combo[2]),
                    'predicted_combo': f'({combo[0]},{combo[1]},{combo[2]})',
                    'probability': None,
                    'raw_output': raw_output,
                })
                if progress_callback is not None:
                    progress_callback(results, idx, total, split)
                del inputs, generated, decoded, image, prompt, text, messages
                _cleanup_cuda(torch)
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor


class LlavaNextGenerativeAdapter(ModelAdapter):
    def _setup(self):
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        dtype = _resolve_dtype(torch, self.config.get('inference', {}).get('torch_dtype', 'auto'))
        model_kwargs = {'device_map': self.config.get('inference', {}).get('device_map', 'auto')}
        if dtype is not None:
            model_kwargs['torch_dtype'] = dtype
        self.processor = LlavaNextProcessor.from_pretrained(self.config['model']['pretrained_name'])
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config['model']['pretrained_name'],
            **model_kwargs,
        )
        self.model = _maybe_load_adapter(self.model, self.config)
        self.pad_token_id = _set_pad_token(processor=self.processor, model=self.model)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
        import torch

        results = []
        model_device = _get_model_device(self.model)
        total = len(records)
        split = records[0].get('split', 'unknown') if records else 'unknown'
        start_time = time.monotonic()
        with torch.inference_mode():
            for idx, record in enumerate(records, start=1):
                _log_sample_progress(self.model_name, scenario, split, idx, total, start_time)
                image = load_image(record, scenario, self.config)
                system_prompt = get_system_prompt(self.config)
                messages = []
                if system_prompt:
                    messages.append({
                        'role': 'system',
                        'content': [{'type': 'text', 'text': system_prompt}],
                    })
                messages.append({
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': get_user_prompt(record, self.config)},
                    ],
                })
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(images=image, text=text, return_tensors='pt')
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.config.get('inference', {}).get('max_new_tokens', 24)),
                    do_sample=False,
                    pad_token_id=self.pad_token_id,
                )
                decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                combo, raw_output = parse_combo_prediction(decoded)
                results.append({
                    'id': record['id'],
                    'label': record['label'],
                    'prediction': int(combo[2]),
                    'predicted_combo': f'({combo[0]},{combo[1]},{combo[2]})',
                    'probability': None,
                    'raw_output': raw_output,
                })
                if progress_callback is not None:
                    progress_callback(results, idx, total, split)
                del inputs, generated, decoded, image, text, messages
                _cleanup_cuda(torch)
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor


class QwenVLChatAdapter(ModelAdapter):
    def _setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {'device_map': self.config.get('inference', {}).get('device_map', 'auto')}
        dtype = _resolve_dtype(torch, self.config.get('inference', {}).get('torch_dtype', 'auto'))
        if dtype is not None:
            model_kwargs['torch_dtype'] = dtype
        trust_remote_code = bool(self.config['model'].get('trust_remote_code', True))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['pretrained_name'],
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['pretrained_name'],
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        ).eval()
        self.model = _maybe_load_adapter(self.model, self.config)
        _set_pad_token(tokenizer=self.tokenizer, model=self.model)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
        import torch

        results = []
        total = len(records)
        split = records[0].get('split', 'unknown') if records else 'unknown'
        start_time = time.monotonic()
        with torch.inference_mode():
            for idx, record in enumerate(records, start=1):
                _log_sample_progress(self.model_name, scenario, split, idx, total, start_time)
                prompt = build_prompt(record, self.config, few_shot_examples=few_shot_examples)
                image_path = str(Path(record['image_path']).resolve())
                query = self.tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': prompt},
                ])
                response, _history = self.model.chat(self.tokenizer, query=query, history=None)
                combo, raw_output = parse_combo_prediction(response)
                results.append({
                    'id': record['id'],
                    'label': record['label'],
                    'prediction': int(combo[2]),
                    'predicted_combo': f'({combo[0]},{combo[1]},{combo[2]})',
                    'probability': None,
                    'raw_output': raw_output,
                })
                if progress_callback is not None:
                    progress_callback(results, idx, total, split)
                del prompt, query, response, _history
                _cleanup_cuda(torch)
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer


class Qwen3VLGenerativeAdapter(ModelAdapter):
    def _setup(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype = _resolve_dtype(torch, self.config.get('inference', {}).get('torch_dtype', 'auto'))
        model_kwargs = {'device_map': self.config.get('inference', {}).get('device_map', 'auto')}
        if dtype is not None:
            model_kwargs['torch_dtype'] = dtype
        self.processor = AutoProcessor.from_pretrained(self.config['model']['pretrained_name'])
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config['model']['pretrained_name'],
            **model_kwargs,
        )
        self.model = _maybe_load_adapter(self.model, self.config)
        self.pad_token_id = _set_pad_token(processor=self.processor, model=self.model)

    def predict(self, records: list[dict], scenario: str, few_shot_examples: list[dict] | None = None, progress_callback=None) -> list[dict]:
        if not hasattr(self, 'model'):
            self._setup()
        import torch

        results = []
        model_device = _get_model_device(self.model)
        total = len(records)
        split = records[0].get('split', 'unknown') if records else 'unknown'
        start_time = time.monotonic()
        with torch.inference_mode():
            for idx, record in enumerate(records, start=1):
                _log_sample_progress(self.model_name, scenario, split, idx, total, start_time)
                image = load_image(record, scenario, self.config)
                system_prompt = get_system_prompt(self.config)
                messages = []
                if system_prompt:
                    messages.append({
                        'role': 'system',
                        'content': [{'type': 'text', 'text': system_prompt}],
                    })
                messages.append({
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': get_user_prompt(record, self.config)},
                    ],
                })
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors='pt')
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.config.get('inference', {}).get('max_new_tokens', 24)),
                    do_sample=False,
                    pad_token_id=self.pad_token_id,
                )
                decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                combo, raw_output = parse_combo_prediction(decoded)
                results.append({
                    'id': record['id'],
                    'label': record['label'],
                    'prediction': int(combo[2]),
                    'predicted_combo': f'({combo[0]},{combo[1]},{combo[2]})',
                    'probability': None,
                    'raw_output': raw_output,
                })
                if progress_callback is not None:
                    progress_callback(results, idx, total, split)
                del inputs, generated, decoded, image, text, messages
                _cleanup_cuda(torch)
        return results

    def release(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
