from __future__ import annotations

import argparse
import json

from .config import PretrainConfig
from .trainer import pretrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssldet-pretrain",
        description=(
            "Pretrain a YOLO backbone with SimCLR, BYOL, MoCo, DINOv2, "
            "DINOv3 guidance, MAE, or I-JEPA."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = pretrain(PretrainConfig.from_yaml(arguments.config))
    print(json.dumps({key: str(value) for key, value in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
