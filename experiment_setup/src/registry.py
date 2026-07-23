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
    SarcasmDetectionHierarchicalCrossAttentionAdapter,
    SarcasmDetectionMultimodalFusionAdapter,
    SarcasmDetectionStagedGatingAdapter,
    TextClassifierAdapter,
    ViCLSRClassifierAdapter,
)


REGISTRY = {
    'text_classifier': TextClassifierAdapter,
    'viclsr_classifier': ViCLSRClassifierAdapter,
    'image_classifier': ImageClassifierAdapter,
    'cirm_classifier': CIRMAdapter,
    'dt4mid_classifier': DT4MIDAdapter,
    'sarcasm_detection_multimodal_fusion': SarcasmDetectionMultimodalFusionAdapter,
    'sarcasm_detection_staged_gating': SarcasmDetectionStagedGatingAdapter,
    'sarcasm_detection_hierarchical_cross_attention': SarcasmDetectionHierarchicalCrossAttentionAdapter,
    'sarcasm_detection_approach1': SarcasmDetectionMultimodalFusionAdapter,
    'sarcasm_detection_approach2': SarcasmDetectionStagedGatingAdapter,
    'sarcasm_detection_approach3': SarcasmDetectionHierarchicalCrossAttentionAdapter,
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
