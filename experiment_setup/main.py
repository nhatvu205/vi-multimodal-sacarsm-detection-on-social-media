import argparse
from pathlib import Path

from experiment_setup.src.config import load_config
from experiment_setup.src.runner import run_pipeline
from experiment_setup.src.runtime_overrides import apply_path_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified sarcasm experiment pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "run", "all"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--scenario",
        choices=["s1", "s2", "s3", "s4", "all"],
        default=None,
        help="Override ablation scenario(s) at runtime",
    )
    parser.add_argument(
        "--json_splits",
        nargs=3,
        metavar=("TRAIN_JSON", "DEV_JSON", "TEST_JSON"),
        help="Override train/dev/test JSON paths at runtime",
    )
    parser.add_argument(
        "--image_root",
        help="Override image root directory at runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    apply_path_overrides(config, json_splits=args.json_splits, image_root=args.image_root)
    if args.scenario:
        config.setdefault("run", {})
        if args.scenario == "all":
            config["run"]["scenarios"] = ["s1", "s2", "s3", "s4"]
        else:
            config["run"]["scenarios"] = [args.scenario]
    run_pipeline(config, stage=args.stage)


if __name__ == "__main__":
    main()
