from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


COMBO_RE = re.compile(r"\(([01])\s*,\s*([01])\s*,\s*([01])\)")


def get_system_prompt(config: dict) -> str:
    return config.get('prompt', {}).get('system', '').strip()


def get_user_prompt(record: dict, config: dict) -> str:
    template = config.get('prompt', {}).get('user_template', '{text}').strip()
    return template.format(text=record['text'])


def select_few_shot_examples(records: Iterable[dict], per_label: int) -> list[dict]:
    if per_label <= 0:
        return []
    buckets = defaultdict(list)
    for record in records:
        label = int(record['label'])
        if len(buckets[label]) < per_label:
            buckets[label].append(record)
        if all(len(v) >= per_label for v in buckets.values()) and len(buckets) >= 2:
            break
    examples = []
    for label in sorted(buckets):
        examples.extend(buckets[label])
    return examples


def _format_example_label(example: dict) -> str:
    labels = example.get('labels') or {}
    if {'text_label', 'image_label', 'mm_label'} <= set(labels):
        return f"({labels['text_label']},{labels['image_label']},{labels['mm_label']})"
    return str(example['label'])


def build_prompt(record: dict, config: dict, few_shot_examples: list[dict] | None = None) -> str:
    sections = []
    system = get_system_prompt(config)
    if system:
        sections.append(system)

    if few_shot_examples:
        sections.append('Examples:')
        for idx, example in enumerate(few_shot_examples, start=1):
            sections.append(
                f"Example {idx}\nText:\n{example['text']}\nLabel: {_format_example_label(example)}"
            )

    sections.append(get_user_prompt(record, config))
    return '\n\n'.join(part for part in sections if part)


def parse_combo_prediction(text: str) -> tuple[tuple[int, int, int], str]:
    cleaned = str(text).strip()
    match = COMBO_RE.search(cleaned)
    if match:
        combo = tuple(int(match.group(i)) for i in range(1, 4))
        return combo, cleaned

    binary_tokens = [tok for tok in re.findall(r'[01]', cleaned)]
    if len(binary_tokens) >= 3:
        combo = tuple(int(tok) for tok in binary_tokens[:3])
        return combo, cleaned

    lowered = cleaned.lower()
    if lowered.startswith('1') or ('sarcastic' in lowered and 'not sarcastic' not in lowered):
        return (1, 0, 1), cleaned
    return (0, 0, 0), cleaned
