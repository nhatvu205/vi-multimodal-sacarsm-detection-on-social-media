from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_command(
    config: str,
    run_name: str,
    seed: int,
    data_root: Path,
    image_root: Path,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable, '-u', '-m', 'experiment_setup.main',
        '--config', config,
        '--stage', 'all',
        '--seed', str(seed),
        '--output_root', str(output_root),
        '--run_name', run_name,
        '--json_splits', str(data_root / 'train.json'), str(data_root / 'dev.json'), str(data_root / 'test.json'),
        '--image_root', str(image_root),
    ]


def compact_run_output(run_dir: Path) -> None:
    """Keep only completed evaluation artifacts after a successful camera-ready run."""
    patterns = ('summary.json', 'metrics_*.json', 'predictions_*.jsonl')
    artifacts = sorted({path for pattern in patterns for path in run_dir.rglob(pattern)})
    if not artifacts:
        raise FileNotFoundError(f'No final artifacts found in {run_dir}')

    names = [path.name for path in artifacts]
    if len(names) != len(set(names)):
        raise ValueError(f'Final artifact names collide in {run_dir}: {names}')

    staging_dir = run_dir / '.final-artifacts'
    staging_dir.mkdir(exist_ok=False)
    for path in artifacts:
        shutil.move(str(path), str(staging_dir / path.name))
    for path in list(run_dir.iterdir()):
        if path != staging_dir:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for path in staging_dir.iterdir():
        shutil.move(str(path), str(run_dir / path.name))
    staging_dir.rmdir()


def _run_one(
    config: str,
    study: str,
    seed: int,
    gpu: int,
    data_root: Path,
    image_root: Path,
    output_root: Path,
) -> Path:
    run_name = f'{study}/seed-{seed}'
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(config, run_name, seed, data_root, image_root, output_root)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    label = f'{study} | seed={seed} | gpu={gpu}'
    print(f'[{label}] Started', flush=True)
    with (run_dir / 'launcher.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            cwd=_repo_root(),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f'[{label}] {line}', end='', flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f'Run failed: {run_name}; see {run_dir / "launcher.log"}')
    compact_run_output(run_dir)
    print(f'[{label}] Finished', flush=True)
    return run_dir


def run_study(
    config: str,
    study: str,
    data_root: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    seeds: tuple[int, ...] = (42, 123, 2026),
    gpus: tuple[int, ...] = (0, 1),
) -> list[Path]:
    if not seeds:
        raise ValueError('At least one seed is required')
    if not gpus:
        raise ValueError('At least one GPU is required')
    os.chdir(_repo_root())
    data_path = Path(data_root)
    image_path = Path(image_root)
    output_path = Path(output_root)
    missing = [path for path in (data_path / 'train.json', data_path / 'dev.json', data_path / 'test.json', image_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing Kaggle input paths: {missing}')

    completed: list[Path] = []
    for start in range(0, len(seeds), len(gpus)):
        batch = seeds[start:start + len(gpus)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(_run_one, config, study, seed, gpus[index], data_path, image_path, output_path): seed
                for index, seed in enumerate(batch)
            }
            for future in as_completed(futures):
                run_dir = future.result()
                completed.append(run_dir)
                print(f'Completed: {run_dir}')
    return sorted(completed)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run one camera-ready study across reproducible seeds.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026])
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    run_study(args.config, args.study, args.data_root, args.image_root, args.output_root, tuple(args.seeds), tuple(args.gpus))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
