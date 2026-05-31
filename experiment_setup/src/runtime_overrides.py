from __future__ import annotations


def apply_path_overrides(
    config: dict,
    json_splits: list[str] | tuple[str, str, str] | None = None,
    image_root: str | None = None,
) -> dict:
    data_cfg = config.setdefault('data', {})
    if json_splits:
        train_path, dev_path, test_path = json_splits
        data_cfg['train_path'] = train_path
        data_cfg['dev_path'] = dev_path
        data_cfg['test_path'] = test_path
    if image_root:
        data_cfg['image_root'] = image_root
    return config
