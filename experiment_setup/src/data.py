from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .io_utils import ensure_dir, load_json, load_jsonl, save_json, save_jsonl
from .preprocess import build_text_variants


SPLIT_NAMES = ('train', 'dev', 'test')
RAW_TEXT_SCENARIOS = {'s1', 's3'}
PREPROCESSED_TEXT_SCENARIOS = {'s2', 's4'}
RAW_IMAGE_SCENARIOS = {'s1', 's2'}
PREPROCESSED_IMAGE_SCENARIOS = {'s3', 's4'}


def repo_root(config: dict) -> Path:
    return Path(config['_meta']['repo_root'])


def resolve_image_path(raw_path: str, config: dict) -> Path:
    raw = Path(str(raw_path))
    root = repo_root(config)
    image_root = root / config['data'].get('image_root', '.')
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            root / raw,
            image_root / raw,
            image_root / raw.name,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot resolve image path: {raw_path}')


def build_run_dir(config: dict) -> Path:
    root = repo_root(config) / config['experiment'].get('output_root', 'experiment_setup/runs')
    return ensure_dir(root / config['experiment']['name'])


def prepare_cache(config: dict, run_dir: Path) -> dict[str, list[dict]]:
    cache_dir = ensure_dir(run_dir / 'cache')
    report = {'splits': {}, 'label_field': config['data']['label_field']}
    cached = {}

    for split in SPLIT_NAMES:
        print(f'[cache] preparing {split} split...')
        split_path = repo_root(config) / config['data'][f'{split}_path']
        records = load_json(split_path)
        cached_rows = []
        labels = Counter()

        for sample in records:
            text_variants = build_text_variants(sample, config)
            image_path = resolve_image_path(sample['image_path'], config)
            label = int(sample[config['data']['label_field']])
            labels[label] += 1
            cached_rows.append({
                'id': int(sample['id']),
                'split': split,
                'source': sample.get('source', ''),
                'label': label,
                'labels': {
                    'mm_label': int(sample.get('mm_label', 0)),
                    'text_label': int(sample.get('text_label', 0)),
                    'image_label': int(sample.get('image_label', 0)),
                },
                'image_path': str(image_path),
                'raw_text': text_variants.raw_text,
                'preprocessed_text': text_variants.preprocessed_text,
            })

        save_jsonl(cache_dir / f'{split}.jsonl', cached_rows)
        report['splits'][split] = {
            'num_samples': len(cached_rows),
            'label_distribution': dict(labels),
        }
        cached[split] = cached_rows
        print(f'[cache] {split}: {len(cached_rows)} samples')

    save_json(run_dir / 'reports' / 'dataset_report.json', report)
    print(f'[cache] report saved to {run_dir / "reports" / "dataset_report.json"}')
    return cached


def load_cached_splits(run_dir: Path) -> dict[str, list[dict]]:
    cache_dir = run_dir / 'cache'
    return {split: load_jsonl(cache_dir / f'{split}.jsonl') for split in SPLIT_NAMES}


def get_text_for_scenario(record: dict, scenario: str) -> str:
    if scenario in RAW_TEXT_SCENARIOS:
        return record['raw_text']
    if scenario in PREPROCESSED_TEXT_SCENARIOS:
        return record['preprocessed_text']
    raise ValueError(f'Unknown scenario: {scenario}')


def load_image(record: dict, scenario: str, config: dict):
    from PIL import Image

    image = Image.open(record['image_path'])
    settings = config.get('preprocessing', {}).get('image', {})
    if settings.get('convert_rgb', True):
        image = image.convert('RGB')
    if scenario in PREPROCESSED_IMAGE_SCENARIOS and settings.get('enabled', True):
        resize = settings.get('resize')
        if resize:
            image = image.resize(tuple(resize))
    return image


def build_records(records: Iterable[dict], scenario: str, config: dict) -> list[dict]:
    built = []
    for record in records:
        built.append({
            'id': record['id'],
            'split': record['split'],
            'label': record['label'],
            'labels': record['labels'],
            'source': record['source'],
            'image_path': record['image_path'],
            'text': get_text_for_scenario(record, scenario),
            'raw_text': record['raw_text'],
            'preprocessed_text': record['preprocessed_text'],
        })
    return built
