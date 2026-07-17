from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PretrainConfig:
    """Configuration shared by every SSL method.

    ``batch_size`` is per GPU. With two GPUs, the effective batch is
    ``2 * batch_size * grad_accum_steps`` for non-contrastive methods.
    Gradient accumulation does not create extra negatives for SimCLR or MoCo.
    """

    method: str = "ijepa"
    image_roots: list[str] = field(default_factory=list)
    output_dir: str = "/kaggle/working/ssldet_pretraining"
    yolo_model: str = "yolo26n.yaml"

    epochs: int = 25
    batch_size: int = 16
    image_size: int = 224
    workers: int = 2
    max_images: int | None = None
    seed: int = 42

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_accum_steps: int = 1
    gradient_clip: float = 5.0
    amp: bool = True

    projection_dim: int = 256
    hidden_dim: int = 1024
    temperature: float = 0.2
    momentum: float = 0.996
    final_momentum: float = 1.0
    queue_size: int = 16384

    dino_output_dim: int = 4096
    student_temperature: float = 0.10
    teacher_temperature: float = 0.04
    center_momentum: float = 0.90
    koleo_weight: float = 0.10
    local_crops: int = 4
    local_crop_size: int = 96

    dinov3_model: str = "dinov3_vits16"
    dinov3_weights: str | None = None
    dinov3_repository: str = "facebookresearch/dinov3"
    dinov3_source: str = "github"
    dinov3_global_weight: float = 1.0
    dinov3_dense_weight: float = 1.0

    mask_ratio: float = 0.60
    num_target_blocks: int = 4
    target_scale_min: float = 0.10
    target_scale_max: float = 0.25
    target_aspect_min: float = 0.75
    target_aspect_max: float = 1.50
    predictor_depth: int = 2
    predictor_heads: int = 4

    save_every: int = 1
    resume: str | None = None

    def validate(self) -> "PretrainConfig":
        supported = {"simclr", "byol", "moco", "dinov2", "dinov3", "mae", "ijepa"}
        self.method = self.method.lower().strip()
        if self.method not in supported:
            raise ValueError(f"method must be one of {sorted(supported)}, got {self.method!r}")
        if not self.image_roots:
            raise ValueError("image_roots must contain at least one image directory")
        if self.epochs < 1 or self.batch_size < 1 or self.image_size < 32:
            raise ValueError("epochs and batch_size must be positive; image_size must be >= 32")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if not 0.0 < self.momentum <= self.final_momentum <= 1.0:
            raise ValueError("Require 0 < momentum <= final_momentum <= 1")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if self.method == "dinov2":
            if self.dino_output_dim < 2 or self.local_crops < 0:
                raise ValueError("dino_output_dim must be >= 2 and local_crops must be >= 0")
            if not 32 <= self.local_crop_size <= self.image_size:
                raise ValueError("local_crop_size must be between 32 and image_size")
            if not 0.0 < self.teacher_temperature < self.student_temperature:
                raise ValueError("Require 0 < teacher_temperature < student_temperature")
            if not 0.0 <= self.center_momentum < 1.0 or self.koleo_weight < 0.0:
                raise ValueError("Invalid DINOv2 center_momentum or koleo_weight")
        if self.method == "dinov3":
            if not self.dinov3_weights:
                raise ValueError("dinov3_weights is required for frozen-teacher distillation")
            if self.dinov3_source not in {"github", "local"}:
                raise ValueError("dinov3_source must be 'github' or 'local'")
            if self.dinov3_global_weight < 0.0 or self.dinov3_dense_weight < 0.0:
                raise ValueError("DINOv3 loss weights must be non-negative")
            if self.dinov3_global_weight + self.dinov3_dense_weight <= 0.0:
                raise ValueError("At least one DINOv3 loss weight must be positive")
        if not 0.0 < self.target_scale_min <= self.target_scale_max < 1.0:
            raise ValueError("Invalid I-JEPA target scale interval")
        if self.predictor_heads < 1 or self.projection_dim % self.predictor_heads:
            raise ValueError("projection_dim must be divisible by predictor_heads")
        if self.method == "ijepa" and self.projection_dim % 4:
            raise ValueError("I-JEPA projection_dim must also be divisible by 4")
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PretrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise TypeError("The YAML root must be a mapping")
        return cls(**values).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return destination
