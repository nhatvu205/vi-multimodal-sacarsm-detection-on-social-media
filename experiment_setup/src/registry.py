from __future__ import annotations

from pathlib import Path

from .models import (
    CIRMAdapter,
    DT4MIDAdapter,
    ImageClassifierAdapter,
    LlavaGenerativeAdapter,
    LlavaNextGenerativeAdapter,
    Qwen3VLGenerativeAdapter,
    QwenVLChatAdapter,
    TextClassifierAdapter,
)


REGISTRY = {
    'text_classifier': TextClassifierAdapter,
    'image_classifier': ImageClassifierAdapter,
    'cirm_classifier': CIRMAdapter,
    'dt4mid_classifier': DT4MIDAdapter,
    'llava_generative': LlavaGenerativeAdapter,
    'llava_next_generative': LlavaNextGenerativeAdapter,
    'qwen_vl_chat': QwenVLChatAdapter,
    'qwen3_vl_generative': Qwen3VLGenerativeAdapter,
}


def create_adapter(config: dict, run_dir: Path):
    family = config['model']['family']
    if family not in REGISTRY:
        raise KeyError(f'Unsupported model family: {family}')
    return REGISTRY[family](config, run_dir)
