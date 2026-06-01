from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config at {path} must be a mapping')
    return data


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "AGENT.md").exists():
            return candidate
    return start.parent


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = _load_yaml(path)
    parent = raw.pop('extends', None)
    if parent:
        base = load_config((path.parent / parent).resolve())
        config = _deep_merge(base, raw)
    else:
        config = raw
    config.setdefault('_meta', {})
    config['_meta']['config_path'] = str(path)
    config['_meta']['repo_root'] = str(_find_repo_root(path.parent))
    return config
