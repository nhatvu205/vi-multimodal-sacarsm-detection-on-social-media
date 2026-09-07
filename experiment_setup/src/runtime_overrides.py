from __future__ import annotations


def apply_path_overrides(
    config: dict,
    json_splits: list[str] | tuple[str, str, str] | None = None,
    image_root: str | None = None,
    output_root: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
) -> dict:
    data_cfg = config.setdefault('data', {})
    if json_splits:
        train_path, dev_path, test_path = json_splits
        data_cfg['train_path'] = train_path
        data_cfg['dev_path'] = dev_path
        data_cfg['test_path'] = test_path
        for name, spec in data_cfg.get('splits', {}).items():
            spec['path'] = {'train': train_path, 'dev': dev_path}.get(name, test_path)
    if image_root:
        data_cfg['image_root'] = image_root
    experiment_cfg = config.setdefault('experiment', {})
    if output_root:
        experiment_cfg['output_root'] = output_root
    if run_name:
        experiment_cfg['name'] = run_name
    if seed is not None:
        experiment_cfg['seed'] = seed
    return config
