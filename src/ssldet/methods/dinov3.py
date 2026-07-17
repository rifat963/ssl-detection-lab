"""DINOv3-guided, label-free feature distillation for spatial student encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SSLMethod


def cosine_regression(student: torch.Tensor, teacher: torch.Tensor, dim: int) -> torch.Tensor:
    student = F.normalize(student.float(), dim=dim)
    teacher = F.normalize(teacher.detach().float(), dim=dim)
    return 2.0 - 2.0 * (student * teacher).sum(dim=dim).mean()


class DINOv3Distillation(SSLMethod):
    """Distill frozen DINOv3 global and dense features into a YOLO-style encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        teacher: nn.Module,
        feature_channels: int,
        feature_dim: int,
        teacher_dim: int,
        global_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if global_weight < 0.0 or dense_weight < 0.0 or global_weight + dense_weight <= 0.0:
            raise ValueError("At least one non-negative DINOv3 loss weight must be positive")
        self.online_encoder = encoder
        self.global_projector = nn.Linear(feature_dim, teacher_dim)
        self.dense_projector = nn.Conv2d(feature_channels, teacher_dim, 1)
        self.global_weight = global_weight
        self.dense_weight = dense_weight
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        for key in list(state):
            if key.startswith(f"{prefix}teacher."):
                state.pop(key)
        return state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        complete_state = dict(state_dict)
        for key, value in super().state_dict().items():
            if key.startswith("teacher."):
                complete_state[key] = value
        return super().load_state_dict(complete_state, strict=strict, assign=assign)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_global, teacher_dense = self.teacher.forward_global_and_dense(images)

        student_dense = self.online_encoder.forward_feature_map(images)
        student_global = F.adaptive_avg_pool2d(student_dense, 1).flatten(1)
        student_global = self.global_projector(student_global)
        student_dense = self.dense_projector(student_dense)
        student_dense = F.interpolate(
            student_dense,
            size=teacher_dense.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        global_loss = cosine_regression(student_global, teacher_global, dim=1)
        dense_loss = cosine_regression(student_dense, teacher_dense, dim=1)
        return self.global_weight * global_loss + self.dense_weight * dense_loss


__all__ = ["DINOv3Distillation", "cosine_regression"]
