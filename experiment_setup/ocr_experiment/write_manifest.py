from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

from experiment_setup.src.io_utils import save_json


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(command: str) -> dict:
    manifest = {
        'command': command,
        'python': platform.python_version(),
        'git_commit': _command('git', 'rev-parse', 'HEAD'),
        'git_branch': _command('git', 'branch', '--show-current'),
        'pip_freeze': (_command('python', '-m', 'pip', 'freeze') or '').splitlines(),
    }
    try:
        import torch

        manifest['torch'] = torch.__version__
        manifest['cuda_available'] = torch.cuda.is_available()
        manifest['cuda_version'] = torch.version.cuda
        manifest['gpus'] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        manifest['cuda_available'] = False
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description='Write reproducibility metadata for a Kaggle run.')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--command', required=True)
    args = parser.parse_args()
    save_json(args.output_dir / 'run_manifest.json', build_manifest(args.command))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
