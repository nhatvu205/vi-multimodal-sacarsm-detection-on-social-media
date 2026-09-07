from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _resolved_config(prediction_path: Path) -> dict:
    for parent in prediction_path.parents:
        candidate = parent / 'resolved_config.yaml'
        if candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding='utf-8')) or {}
    return {}


def _study_name(value: str) -> str:
    path = Path(value)
    return str(path.parent) if re.fullmatch(r'seed-\d+', path.name) else value


def _summary_model(prediction_path: Path) -> str:
    summary_path = prediction_path.parent / 'summary.json'
    if not summary_path.exists():
        return ''
    try:
        rows = json.loads(summary_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return ''
    return rows[0].get('model', '') if rows else ''


def _metric(rows: list[dict]) -> dict:
    labels = [int(row['label']) for row in rows]
    predictions = [int(row['prediction']) for row in rows]
    return {
        'num_samples': len(rows),
        'num_negative': labels.count(0),
        'num_positive': labels.count(1),
        'accuracy': sum(label == prediction for label, prediction in zip(labels, predictions)) / len(rows) if rows else 0.0,
        'f1_macro': _f1_macro(labels, predictions),
    }


def _f1_macro(labels, predictions) -> float:
    scores = []
    for class_id in (0, 1):
        tp = sum(label == class_id and prediction == class_id for label, prediction in zip(labels, predictions))
        fp = sum(label != class_id and prediction == class_id for label, prediction in zip(labels, predictions))
        fn = sum(label == class_id and prediction != class_id for label, prediction in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _bootstrap_f1(rows: list[dict], seed: int, iterations: int) -> tuple[float, float]:
    if not rows:
        return (float('nan'), float('nan'))
    labels = np.asarray([int(row['label']) for row in rows])
    predictions = np.asarray([int(row['prediction']) for row in rows])
    generator = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = generator.integers(0, len(rows), len(rows))
        values[index] = _f1_macro(labels[sample], predictions[sample])
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def _subset_rows(rows: list[dict], split: str) -> dict[str, list[dict]]:
    subsets = {'full': rows}
    if split == 'test':
        subsets['ocr_present'] = [row for row in rows if row['has_ocr']]
        subsets['ocr_absent'] = [row for row in rows if not row['has_ocr']]
    return subsets


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) if rows else []
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate(rows: list[dict], path: Path) -> list[str]:
    errors = []
    ids = [int(row['id']) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append(f'{path}: duplicate prediction IDs')
    for row in rows:
        labels = row.get('labels')
        if labels and int(row['label']) != int(labels['mm_label']):
            errors.append(f"{path}: id={row['id']} label does not match labels.mm_label")
            break
        if row.get('seed') is None or not row.get('input_mode'):
            errors.append(f"{path}: missing seed or input_mode metadata")
            break
    return errors


def _error_slices(rows: list[dict], metadata_by_path: dict[Path, dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        metadata = metadata_by_path[Path(row['_path'])]
        combo = row['labels']
        key = (
            metadata['experiment_name'], metadata['model'], metadata['input_mode'], metadata['seed'],
            metadata['split'], row['source'], row['has_ocr'],
            combo['text_label'], combo['image_label'], combo['mm_label'],
        )
        groups[key].append(row)

    output = []
    for key, values in groups.items():
        labels = [int(row['label']) for row in values]
        predictions = [int(row['prediction']) for row in values]
        fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
        fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
        negative = labels.count(0)
        positive = labels.count(1)
        output.append({
            'experiment_name': key[0], 'model': key[1], 'input_mode': key[2], 'seed': key[3],
            'split': key[4], 'source': key[5], 'has_ocr': key[6],
            'text_label': key[7], 'image_label': key[8], 'mm_label': key[9],
            'support': len(values), 'negative_support': negative, 'positive_support': positive,
            'fp': fp, 'fn': fn,
            'fpr': fp / negative if negative else None,
            'fnr': fn / positive if positive else None,
        })
    return output


def _candidate_examples(rows: list[dict], metadata_by_path: dict[Path, dict]) -> list[dict]:
    candidates = []
    by_model_seed_input: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        meta = metadata_by_path[Path(row['_path'])]
        if meta['split'] != 'test':
            continue
        common = (meta['model'], meta['seed'], meta['input_mode'])
        by_model_seed_input[common][int(row['id'])] = row
        combo = row['labels']
        if (combo['text_label'], combo['image_label'], combo['mm_label']) == (0, 0, 1) and row['label'] != row['prediction']:
            candidates.append(_candidate('error_001', row, meta))
        elif row['label'] == 0 and row['prediction'] == 1:
            candidates.append(_candidate('false_positive', row, meta))
        elif row['label'] == row['prediction']:
            candidates.append(_candidate('correct_prediction', row, meta))

    pairs: dict[tuple, dict[str, dict[int, dict]]] = defaultdict(dict)
    for (model, seed, input_mode), values in by_model_seed_input.items():
        pairs[(model, seed)][input_mode] = values
    for (model, seed), inputs in pairs.items():
        if 'caption' not in inputs or 'caption_ocr' not in inputs:
            continue
        for sample_id in set(inputs['caption']) & set(inputs['caption_ocr']):
            caption, caption_ocr = inputs['caption'][sample_id], inputs['caption_ocr'][sample_id]
            if not caption['has_ocr']:
                continue
            meta = metadata_by_path[Path(caption['_path'])]
            if caption['prediction'] != caption['label'] and caption_ocr['prediction'] == caption_ocr['label']:
                candidates.append(_candidate('ocr_improves', caption_ocr, meta, comparison_prediction=caption['prediction']))
            if caption['prediction'] == caption['label'] and caption_ocr['prediction'] != caption_ocr['label']:
                candidates.append(_candidate('ocr_harms', caption_ocr, meta, comparison_prediction=caption['prediction']))
    return candidates


def _candidate(category: str, row: dict, metadata: dict, comparison_prediction: int | None = None) -> dict:
    labels = row['labels']
    return {
        'category': category, 'id': row['id'], 'model': metadata['model'], 'seed': metadata['seed'],
        'input_mode': metadata['input_mode'], 'split': metadata['split'], 'source': row['source'],
        'has_ocr': row['has_ocr'], 'label': row['label'], 'prediction': row['prediction'],
        'comparison_prediction': comparison_prediction, 'text_label': labels['text_label'],
        'image_label': labels['image_label'], 'mm_label': labels['mm_label'],
    }


def analyze(runs_root: Path, output_dir: Path, bootstrap_iterations: int = 10_000) -> int:
    prediction_paths = sorted(
        path for path in runs_root.rglob('predictions_*.jsonl')
        if '.checkpoint.' not in path.name
    )
    if not prediction_paths:
        raise FileNotFoundError(f'No prediction files found under {runs_root}')

    errors: list[str] = []
    metric_rows: list[dict] = []
    all_rows: list[dict] = []
    metadata_by_path: dict[Path, dict] = {}
    for path in prediction_paths:
        rows = _load_jsonl(path)
        errors.extend(_validate(rows, path))
        if not rows:
            continue
        config = _resolved_config(path)
        split = path.stem.removeprefix('predictions_')
        metadata = {
            'experiment_name': rows[0].get('experiment_name') or config.get('experiment', {}).get('name', ''),
            'model': rows[0].get('model') or config.get('model', {}).get('key', ''),
            'input_mode': rows[0].get('input_mode'),
            'seed': int(rows[0].get('seed', config.get('experiment', {}).get('seed', 0))),
            'split': split,
        }
        if not metadata['experiment_name']:
            parent = path.parent
            metadata['experiment_name'] = parent.parent.name if re.fullmatch(r'seed-\d+', parent.name) else parent.name
        if not metadata['model']:
            metadata['model'] = _summary_model(path) or (path.parents[1].name if len(path.parents) > 1 else '')
        metadata['experiment_name'] = _study_name(metadata['experiment_name'])
        metadata_by_path[path] = metadata
        for row in rows:
            row['_path'] = str(path)
        all_rows.extend(rows)
        for subset, subset_rows in _subset_rows(rows, split).items():
            values = _metric(subset_rows)
            lower, upper = _bootstrap_f1(subset_rows, metadata['seed'] + len(subset), bootstrap_iterations)
            metric_rows.append({**metadata, 'subset': subset, **values, 'bootstrap_ci_lower': lower, 'bootstrap_ci_upper': upper})

    validation = {'prediction_files': len(prediction_paths), 'errors': errors}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    if errors:
        raise ValueError('Prediction validation failed; see validation.json')

    summaries: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    keys = ('experiment_name', 'model', 'input_mode', 'split', 'subset')
    for row in metric_rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for key, values in groups.items():
        f1s = np.asarray([value['f1_macro'] for value in values])
        summaries.append({
            **dict(zip(keys, key)), 'num_seeds': len(values),
            'f1_macro_mean': float(f1s.mean()),
            'f1_macro_sd': float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            'bootstrap_ci_lower_mean': float(np.mean([value['bootstrap_ci_lower'] for value in values])),
            'bootstrap_ci_upper_mean': float(np.mean([value['bootstrap_ci_upper'] for value in values])),
        })

    _write_csv(output_dir / 'seed_metrics.csv', metric_rows)
    _write_csv(output_dir / 'seed_summary.csv', summaries)
    _write_csv(output_dir / 'error_slices.csv', _error_slices(all_rows, metadata_by_path))
    _write_csv(output_dir / 'candidate_examples.csv', _candidate_examples(all_rows, metadata_by_path))
    print(f'Analysis written to {output_dir}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Aggregate camera-ready prediction artifacts.')
    parser.add_argument('--runs_root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--bootstrap_iterations', type=int, default=10_000)
    args = parser.parse_args()
    return analyze(args.runs_root, args.output_dir, args.bootstrap_iterations)


if __name__ == '__main__':
    raise SystemExit(main())
