from __future__ import annotations

from pathlib import Path

from .data import build_records, build_run_dir, load_cached_splits, prepare_cache
from .io_utils import ensure_dir, save_csv, save_json, save_jsonl, save_yaml


def _write_resolved_config(config: dict, run_dir: Path) -> None:
    clean = {k: v for k, v in config.items() if k != '_meta'}
    save_yaml(run_dir / 'resolved_config.yaml', clean)


def _scenario_output_dir(run_dir: Path, model_name: str, scenario: str) -> Path:
    return ensure_dir(run_dir / model_name / scenario)


def _save_predictions(path: Path, source_records: list[dict], predictions: list[dict], split: str, scenario: str) -> None:
    by_id = {record['id']: record for record in source_records}
    rows = []
    for pred in predictions:
        source = by_id[pred['id']]
        rows.append({
            'id': pred['id'],
            'split': split,
            'scenario': scenario,
            'label': pred['label'],
            'prediction': pred['prediction'],
            'probability': pred['probability'],
            'predicted_combo': pred.get('predicted_combo'),
            'raw_output': pred['raw_output'],
            'text': source['text'],
            'image_path': source['image_path'],
        })
    save_jsonl(path, rows)


def _save_summary(run_dir: Path, model_name: str, rows: list[dict]) -> None:
    save_json(run_dir / model_name / 'summary.json', rows)
    save_csv(
        run_dir / model_name / 'summary.csv',
        rows,
        [
            'model',
            'scenario',
            'split',
            'accuracy',
            'f1_macro',
            'f1_weighted',
            'precision_weighted',
            'recall_weighted',
            'auc',
            'num_samples',
        ],
    )


def run_pipeline(config: dict, stage: str = 'all') -> None:
    run_dir = build_run_dir(config)
    _write_resolved_config(config, run_dir)
    print(f'[run] output dir: {run_dir}')

    if stage in {'preprocess', 'all'}:
        prepare_cache(config, run_dir)

    if stage == 'preprocess':
        return

    from .metrics import compute_classification_metrics
    from .prompting import select_few_shot_examples
    from .registry import create_adapter

    cached = load_cached_splits(run_dir)
    model_name = config['model']['key']
    summary_rows = []

    for scenario in config['run']['scenarios']:
        print(f'[run] model={model_name} scenario={scenario}')
        adapter = create_adapter(config, run_dir)
        train_records = build_records(cached['train'], scenario, config)
        dev_records = build_records(cached['dev'], scenario, config)
        test_records = build_records(cached['test'], scenario, config)

        if adapter.supports_training and config.get('training', {}).get('enabled', False):
            adapter.train(train_records, dev_records, scenario)

        few_shot_examples = []
        n_examples = int(config.get('inference', {}).get('few_shot_examples', 0))
        if n_examples > 0 and not adapter.supports_training:
            few_shot_examples = select_few_shot_examples(train_records, per_label=n_examples)

        for split in config['run']['eval_splits']:
            split_records = {'dev': dev_records, 'test': test_records}[split]
            out_dir = _scenario_output_dir(run_dir, model_name, scenario)
            checkpoint_every = int(config.get('inference', {}).get('checkpoint_every_samples', 0))

            def progress_callback(predictions: list[dict], processed: int, total: int, split_name: str) -> None:
                if checkpoint_every <= 0:
                    return
                if processed % checkpoint_every != 0 and processed != total:
                    return
                checkpoint_path = out_dir / f'predictions_{split_name}.checkpoint.jsonl'
                _save_predictions(checkpoint_path, split_records, predictions, split_name, scenario)
                save_json(
                    out_dir / f'progress_{split_name}.json',
                    {
                        'model': model_name,
                        'scenario': scenario,
                        'split': split_name,
                        'processed_samples': processed,
                        'total_samples': total,
                        'checkpoint_every_samples': checkpoint_every,
                    },
                )
                print(f"[run] saved inference checkpoint: {checkpoint_path} ({processed}/{total})")

            predictions = adapter.predict(
                split_records,
                scenario,
                few_shot_examples=few_shot_examples,
                progress_callback=progress_callback,
            )
            labels = [row['label'] for row in predictions]
            preds = [row['prediction'] for row in predictions]
            probs = [row['probability'] for row in predictions]
            probabilities = probs if all(value is not None for value in probs) else None
            metrics = compute_classification_metrics(labels, preds, probabilities)
            metrics_row = {
                'model': model_name,
                'scenario': scenario,
                'split': split,
                **metrics,
            }
            summary_rows.append(metrics_row)
            print(
                f"[run] {model_name} {scenario} {split} | "
                f"acc={metrics_row['accuracy']:.4f} f1w={metrics_row['f1_weighted']:.4f}"
            )

            if config['run'].get('save_metrics', True):
                save_json(out_dir / f'metrics_{split}.json', metrics_row)
            if config['run'].get('save_predictions', True):
                _save_predictions(out_dir / f'predictions_{split}.jsonl', split_records, predictions, split, scenario)

        adapter.release()

    _save_summary(run_dir, model_name, summary_rows)
    print(f'[run] summary saved to {run_dir / model_name / "summary.json"}')
