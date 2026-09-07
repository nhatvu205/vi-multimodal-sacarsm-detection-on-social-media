from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiment_setup.camera_ready.analyze import analyze
from experiment_setup.camera_ready.run_jobs import build_command
from experiment_setup.src.data import build_run_dir, prepare_cache
from experiment_setup.src.preprocess import build_text_variants
from experiment_setup.src.runtime_overrides import apply_path_overrides


class CameraReadyDataTests(unittest.TestCase):
    def test_launcher_builds_a_self_contained_kaggle_command(self) -> None:
        command = build_command(
            'experiment_setup/configs/camera_ready/ocr_phobert_caption.yaml',
            'ocr/phobert-caption/seed-42',
            42,
            Path('/kaggle/input/data/final-data'),
            Path('/kaggle/input/data/images'),
            Path('/kaggle/working/camera_ready'),
        )
        self.assertIn('--seed', command)
        self.assertIn('42', command)
        self.assertIn('/kaggle/input/data/images', command)
        self.assertIn('ocr/phobert-caption/seed-42', command)

    def test_text_input_modes_and_legacy_flag(self) -> None:
        sample = {'text': 'caption', 'ocr_text': 'image words'}
        settings = {'preprocessing': {'text': {}}}
        self.assertEqual(build_text_variants(sample, {**settings, 'data': {'text_input': 'caption'}}).raw_text, 'caption')
        self.assertEqual(build_text_variants(sample, {**settings, 'data': {'text_input': 'ocr'}}).raw_text, 'image words')
        self.assertEqual(
            build_text_variants(sample, {**settings, 'data': {'text_input': 'caption_ocr'}}).raw_text,
            'caption [OCR] image words',
        )
        self.assertEqual(
            build_text_variants({'text': 'caption', 'ocr_text': ''}, {**settings, 'data': {'text_input': 'caption_ocr'}}).raw_text,
            'caption',
        )
        self.assertEqual(
            build_text_variants(sample, {**settings, 'data': {'include_ocr_in_text': False}}).raw_text,
            'caption',
        )

    def test_filtered_splits_and_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'images' / 'one.webp').write_bytes(b'not-read-in-this-test')
            rows = [
                {'id': 1, 'text': 'a', 'ocr_text': '', 'image_path': 'old/one.webp', 'source': 'facebook', 'mm_label': 0, 'text_label': 0, 'image_label': 0},
                {'id': 2, 'text': 'b', 'ocr_text': 'ocr', 'image_path': 'old/one.webp', 'source': 'threads', 'mm_label': 1, 'text_label': 0, 'image_label': 0},
            ]
            (root / 'records.json').write_text(json.dumps(rows), encoding='utf-8')
            config = {
                '_meta': {'repo_root': str(root)},
                'experiment': {'name': 'run', 'output_root': 'outputs'},
                'data': {
                    'label_field': 'mm_label', 'image_root': str(root / 'images'), 'text_input': 'caption',
                    'splits': {'train': {'path': 'records.json', 'source': 'facebook'}, 'dev': {'path': 'records.json', 'has_ocr': True}, 'test': {'path': 'records.json'}},
                },
                'preprocessing': {'text': {}, 'image': {}},
            }
            cached = prepare_cache(config, build_run_dir(config))
            self.assertEqual([row['id'] for row in cached['train']], [1])
            self.assertEqual([row['id'] for row in cached['dev']], [2])
            self.assertTrue(cached['test'][0]['image_path'].endswith('images/one.webp'))

    def test_runtime_overrides_keep_seed_outputs_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {'_meta': {'repo_root': directory}, 'experiment': {'name': 'base'}, 'data': {}}
            apply_path_overrides(config, output_root='results', run_name='ocr/phobert/seed-42', seed=42)
            other = {'_meta': {'repo_root': directory}, 'experiment': {'name': 'base'}, 'data': {}}
            apply_path_overrides(other, output_root='results', run_name='ocr/phobert/seed-123', seed=123)
            self.assertEqual(config['experiment']['seed'], 42)
            self.assertNotEqual(build_run_dir(config), build_run_dir(other))


class CameraReadyAnalyzerTests(unittest.TestCase):
    def test_analyzer_writes_metrics_slices_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs' / 'ocr' / 'caption' / 'seed-42'
            output = root / 'analysis'
            prediction_dir = run / 'phobert-base' / 's1'
            prediction_dir.mkdir(parents=True)
            (run / 'resolved_config.yaml').write_text(yaml.safe_dump({'experiment': {'name': 'ocr/caption', 'seed': 42}, 'model': {'key': 'phobert-base'}}), encoding='utf-8')
            rows = [
                {'id': 1, 'label': 1, 'prediction': 0, 'seed': 42, 'input_mode': 'caption', 'source': 'facebook', 'has_ocr': True, 'labels': {'text_label': 0, 'image_label': 0, 'mm_label': 1}},
                {'id': 2, 'label': 0, 'prediction': 0, 'seed': 42, 'input_mode': 'caption', 'source': 'threads', 'has_ocr': False, 'labels': {'text_label': 0, 'image_label': 0, 'mm_label': 0}},
            ]
            (prediction_dir / 'predictions_test.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            analyze(root / 'runs', output, bootstrap_iterations=20)
            self.assertTrue((output / 'seed_metrics.csv').exists())
            self.assertTrue((output / 'error_slices.csv').exists())
            candidates = (output / 'candidate_examples.csv').read_text(encoding='utf-8')
            self.assertIn('error_001', candidates)


if __name__ == '__main__':
    unittest.main()
