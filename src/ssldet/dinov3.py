"""Reusable DINOv3-to-YOLO26 distillation through LightlyTrain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

DEFAULT_DINOV3_VITB16_WEIGHTS = (
    "/kaggle/input/datasets/mrifatrashid/dinov3-weigths/"
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


@dataclass
class DINOv3YOLOConfig:
    """Configuration for frozen DINOv3 teacher distillation into YOLO26."""

    data: str | list[str]
    output_dir: str = "/kaggle/working/dinov3_yolo26_football"
    teacher_weights: str = DEFAULT_DINOV3_VITB16_WEIGHTS
    teacher_model: str = "dinov3/vitb16"
    student_model: str = "ultralytics/yolo26n.yaml"
    method: str = "distillation"
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 2
    devices: int | str = 1
    accelerator: str = "gpu"
    precision: str = "16-mixed"
    seed: int = 42
    overwrite: bool = True
    image_size: int = 224
    min_scale: float = 0.35
    max_scale: float = 1.0
    horizontal_flip_probability: float = 0.5
    grayscale_probability: float = 0.1
    extra_arguments: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_weights: bool = True) -> "DINOv3YOLOConfig":
        if not self.data or (isinstance(self.data, list) and not self.data):
            raise ValueError("data must identify one or more unlabeled image sources")
        if self.teacher_model != "dinov3/vitb16":
            raise ValueError("The bundled checkpoint requires teacher_model='dinov3/vitb16'")
        if self.method != "distillation":
            raise ValueError("DINOv3 YOLO26 training requires method='distillation'")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs and batch_size must be positive; num_workers must be non-negative")
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("image_size must be at least 32 and divisible by the ViT-B/16 patch size")
        if not 0.0 < self.min_scale <= self.max_scale <= 1.0:
            raise ValueError("Require 0 < min_scale <= max_scale <= 1")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")
        if not 0.0 <= self.grayscale_probability <= 1.0:
            raise ValueError("grayscale_probability must be between 0 and 1")
        if require_weights and not Path(self.teacher_weights).is_file():
            raise FileNotFoundError(
                f"DINOv3 ViT-B/16 weights were not found: {self.teacher_weights}"
            )
        return self

    def lightly_arguments(self) -> dict[str, Any]:
        self.validate(require_weights=False)
        arguments: dict[str, Any] = {
            "out": self.output_dir,
            "data": self.data,
            "model": self.student_model,
            "method": self.method,
            "method_args": {
                "teacher": self.teacher_model,
                "teacher_weights": self.teacher_weights,
            },
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "devices": self.devices,
            "accelerator": self.accelerator,
            "precision": self.precision,
            "seed": self.seed,
            "overwrite": self.overwrite,
            "transform_args": {
                "image_size": (self.image_size, self.image_size),
                "random_resize": {
                    "min_scale": self.min_scale,
                    "max_scale": self.max_scale,
                },
                "random_flip": {
                    "horizontal_prob": self.horizontal_flip_probability,
                    "vertical_prob": 0.0,
                },
                "random_gray_scale": self.grayscale_probability,
            },
        }
        arguments.update(self.extra_arguments)
        return arguments

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DINOv3YOLOResult:
    output_dir: Path
    training_checkpoint: Path
    yolo_checkpoint: Path
    metrics_jsonl: Path
    teacher_model: str
    teacher_weights: Path
    student_model: str


def _lightly_train(module: ModuleType | None = None) -> ModuleType:
    if module is not None:
        return module
    try:
        return import_module("lightly_train")
    except ImportError as error:
        raise ImportError(
            "Install lightly-train[ultralytics]>=0.16.2 to use DINOv3 YOLO26 distillation"
        ) from error


def validate_dinov3_yolo26_support(
    module: ModuleType | None = None,
    *,
    teacher_model: str = "dinov3/vitb16",
    student_model: str = "ultralytics/yolo26n.yaml",
    method: str = "distillation",
) -> dict[str, bool]:
    """Check that the active LightlyTrain build contains the required components."""

    lightly_train = _lightly_train(module)
    available_methods = set(lightly_train.list_methods())
    available_models = set(lightly_train.list_models())
    support = {
        method: method in available_methods,
        teacher_model: teacher_model in available_models,
        student_model: student_model in available_models,
    }
    missing = [name for name, available in support.items() if not available]
    if missing:
        raise RuntimeError(
            "The installed LightlyTrain build does not support: " + ", ".join(missing)
        )
    return support


def pretrain_dinov3_yolo26(
    config: DINOv3YOLOConfig,
    *,
    module: ModuleType | None = None,
) -> DINOv3YOLOResult:
    """Distill local DINOv3 ViT-B/16 weights into an Ultralytics YOLO26 student."""

    config.validate(require_weights=True)
    lightly_train = _lightly_train(module)
    validate_dinov3_yolo26_support(
        lightly_train,
        teacher_model=config.teacher_model,
        student_model=config.student_model,
        method=config.method,
    )
    lightly_train.pretrain(**config.lightly_arguments())

    output_dir = Path(config.output_dir)
    result = DINOv3YOLOResult(
        output_dir=output_dir,
        training_checkpoint=output_dir / "checkpoints" / "last.ckpt",
        yolo_checkpoint=output_dir / "exported_models" / "exported_last.pt",
        metrics_jsonl=output_dir / "metrics.jsonl",
        teacher_model=config.teacher_model,
        teacher_weights=Path(config.teacher_weights),
        student_model=config.student_model,
    )
    missing = [
        path
        for path in (result.training_checkpoint, result.yolo_checkpoint)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "LightlyTrain completed without the expected output files: "
            + ", ".join(str(path) for path in missing)
        )
    return result


__all__ = [
    "DEFAULT_DINOV3_VITB16_WEIGHTS",
    "DINOv3YOLOConfig",
    "DINOv3YOLOResult",
    "pretrain_dinov3_yolo26",
    "validate_dinov3_yolo26_support",
]
