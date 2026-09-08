from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .io_utils import ensure_dir, load_json, load_jsonl, save_json, save_jsonl
from .preprocess import build_text_variants


SCENARIO_TEXT_FIELD = {
    's1': 'raw_text',
    's2': 'emoji_removed_text',
    's3': 'preprocessed_text',
    's4': 'preprocessed_emoji_removed_text',
}


def repo_root(config: dict) -> Path:
    return Path(config['_meta']['repo_root'])


def _resolve_path(value: str, config: dict) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(config) / path


def resolve_image_path(raw_path: str, config: dict) -> Path:
    raw = Path(str(raw_path))
    image_root = _resolve_path(config['data'].get('image_root', '.'), config)
    candidates = (image_root / raw.name, image_root / 'images' / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot resolve image path: {raw_path}; tried {candidates[0]} and {candidates[1]}')


def build_run_dir(config: dict) -> Path:
    output_root = _resolve_path(config['experiment'].get('output_root', 'experiment_setup/runs'), config)
    name = Path(config['experiment']['name'])
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('experiment.name must be a relative path without ".."')
    return ensure_dir(output_root / name)


def _split_specs(config: dict) -> dict[str, dict]:
    configured = config['data'].get('splits')
    if configured:
        return {name: dict(spec) for name, spec in configured.items()}
    return {
        name: {'path': config['data'][f'{name}_path']}
        for name in ('train', 'dev', 'test')
    }


def _matches_filter(sample: dict, spec: dict) -> bool:
    source = spec.get('source')
    if source is not None:
        allowed = {source} if isinstance(source, str) else set(source)
        if sample.get('source') not in allowed:
            return False
    if 'has_ocr' in spec:
        has_ocr = bool(str(sample.get('ocr_text', '') or '').strip())
        if has_ocr != bool(spec['has_ocr']):
            return False
    return True


def prepare_cache(config: dict, run_dir: Path) -> dict[str, list[dict]]:
    cache_dir = ensure_dir(run_dir / 'cache')
    report = {'splits': {}, 'label_field': config['data']['label_field']}
    cached = {}

    for split, spec in _split_specs(config).items():
        print(f'[cache] preparing {split} split...')
        records = load_json(_resolve_path(spec['path'], config))
        cached_rows = []
        labels = Counter()

        for sample in records:
            if not _matches_filter(sample, spec):
                continue
            text_variants = build_text_variants(sample, config)
            image_path = resolve_image_path(sample['image_path'], config)
            label = int(sample[config['data']['label_field']])
            labels[label] += 1
            cached_rows.append({
                'id': int(sample['id']),
                'split': split,
                'source': sample.get('source', ''),
                'has_ocr': bool(str(sample.get('ocr_text', '') or '').strip()),
                'label': label,
                'labels': {
                    'mm_label': int(sample.get('mm_label', 0)),
                    'text_label': int(sample.get('text_label', 0)),
                    'image_label': int(sample.get('image_label', 0)),
                },
                'image_path': str(image_path),
                'raw_text': text_variants.raw_text,
                'emoji_removed_text': text_variants.emoji_removed_text,
                'preprocessed_text': text_variants.preprocessed_text,
                'preprocessed_emoji_removed_text': text_variants.preprocessed_emoji_removed_text,
            })

        save_jsonl(cache_dir / f'{split}.jsonl', cached_rows)
        report['splits'][split] = {
            'num_samples': len(cached_rows),
            'label_distribution': dict(labels),
            'filter': {key: spec[key] for key in ('source', 'has_ocr') if key in spec},
        }
        cached[split] = cached_rows
        print(f'[cache] {split}: {len(cached_rows)} samples')

    save_json(run_dir / 'reports' / 'dataset_report.json', report)
    print(f'[cache] report saved to {run_dir / "reports" / "dataset_report.json"}')
    return cached


def load_cached_splits(run_dir: Path, config: dict) -> dict[str, list[dict]]:
    cache_dir = run_dir / 'cache'
    return {split: load_jsonl(cache_dir / f'{split}.jsonl') for split in _split_specs(config)}


def get_text_for_scenario(record: dict, scenario: str) -> str:
    field = SCENARIO_TEXT_FIELD.get(scenario)
    if field is None:
        raise ValueError(f'Unknown scenario: {scenario}')
    return record[field]


def load_image(record: dict, scenario: str, config: dict):
    from PIL import Image

    image = Image.open(record['image_path'])
    settings = config.get('preprocessing', {}).get('image', {})
    if settings.get('convert_rgb', True):
        image = image.convert('RGB')
    if settings.get('enabled', True):
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
            'has_ocr': record['has_ocr'],
            'image_path': record['image_path'],
            'text': get_text_for_scenario(record, scenario),
            'raw_text': record['raw_text'],
            'emoji_removed_text': record['emoji_removed_text'],
            'preprocessed_text': record['preprocessed_text'],
            'preprocessed_emoji_removed_text': record['preprocessed_emoji_removed_text'],
        })
    return built
