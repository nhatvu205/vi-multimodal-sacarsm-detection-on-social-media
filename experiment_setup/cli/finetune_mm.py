import argparse
from pathlib import Path

from experiment_setup.src.config import load_config
from experiment_setup.src.mm_qlora import finetune_multimodal_model
from experiment_setup.src.runtime_overrides import apply_path_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for multimodal models")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scenario", default=None, help="Override ablation scenario for fine-tuning")
    parser.add_argument(
        "--json_splits",
        nargs=3,
        metavar=("TRAIN_JSON", "DEV_JSON", "TEST_JSON"),
        help="Override train/dev/test JSON paths at runtime",
    )
    parser.add_argument("--image_root", help="Override image root directory at runtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    apply_path_overrides(config, json_splits=args.json_splits, image_root=args.image_root)
    finetune_multimodal_model(config, scenario=args.scenario)


if __name__ == "__main__":
    main()
