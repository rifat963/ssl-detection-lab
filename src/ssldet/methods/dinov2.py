"""Compute-scaled DINOv2-style self-distillation for YOLO backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import concat_all_gather
from .common import SSLMethod, ema_update, frozen_copy


class DINOHead(nn.Module):
    """Small normalized projection head used by both student and teacher."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.mlp(features), dim=-1)
        return self.last_layer(features)


def dino_cross_view_loss(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    center: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
) -> torch.Tensor:
    """Cross-entropy from each global teacher view to every different student view."""

    teacher_probabilities = [
        F.softmax((logits.detach() - center) / teacher_temperature, dim=-1)
        for logits in teacher_logits
    ]
    student_log_probabilities = [
        F.log_softmax(logits / student_temperature, dim=-1) for logits in student_logits
    ]
    losses = []
    for teacher_index, teacher_probability in enumerate(teacher_probabilities):
        for student_index, student_log_probability in enumerate(student_log_probabilities):
            if student_index == teacher_index:
                continue
            losses.append(
                -(teacher_probability * student_log_probability).sum(dim=-1).mean()
            )
    if not losses:
        raise ValueError("DINOv2 requires at least two image views")
    return torch.stack(losses).mean()


def koleo_loss(features: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Encourage a uniform representation distribution using nearest-neighbour entropy."""

    if features.shape[0] < 2:
        return features.sum() * 0.0
    normalized = F.normalize(features, dim=-1)
    similarities = normalized @ normalized.T
    similarities.fill_diagonal_(-1.0)
    nearest_similarity = similarities.max(dim=1).values
    nearest_distance = torch.sqrt((2.0 - 2.0 * nearest_similarity).clamp_min(epsilon))
    return -torch.log(nearest_distance + epsilon).mean()


class DINOv2(SSLMethod):
    """DINOv2-style global/local distillation adapted to a CNN detector backbone.

    The full official DINOv2 recipe is ViT-specific and also uses iBOT patch masking.
    This compute-scaled adaptation retains multi-crop DINO distillation, an EMA teacher,
    centering/sharpening, and KoLeo regularization while transferring the student YOLO
    backbone directly into the detector.
    """

    requires_two_views = True

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        student_temperature: float,
        teacher_temperature: float,
        center_momentum: float,
        momentum: float,
        koleo_weight: float,
    ) -> None:
        super().__init__()
        self.student_encoder = encoder
        self.student_head = DINOHead(
            feature_dim, hidden_dim, bottleneck_dim, output_dim
        )
        self.teacher_encoder = frozen_copy(encoder)
        self.teacher_head = frozen_copy(self.student_head)
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.current_momentum = momentum
        self.koleo_weight = koleo_weight
        self.register_buffer("center", torch.zeros(1, output_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def _update_center(self, teacher_logits: list[torch.Tensor]) -> None:
        gathered = concat_all_gather(torch.cat(teacher_logits, dim=0).detach())
        batch_center = gathered.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )

    def forward(self, batch) -> torch.Tensor:
        views = tuple(batch) if isinstance(batch, (tuple, list)) else (batch,)
        if len(views) < 2:
            raise ValueError("DINOv2 requires two global views")

        student_features = [self.student_encoder(view) for view in views]
        student_logits = [self.student_head(features) for features in student_features]
        with torch.no_grad():
            teacher_logits = [
                self.teacher_head(self.teacher_encoder(view)) for view in views[:2]
            ]

        distillation = dino_cross_view_loss(
            student_logits,
            teacher_logits,
            self.center,
            self.student_temperature,
            self.teacher_temperature,
        )
        regularization = 0.5 * (
            koleo_loss(student_features[0]) + koleo_loss(student_features[1])
        )
        self._update_center(teacher_logits)
        return distillation + self.koleo_weight * regularization

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        ema_update(self.student_encoder, self.teacher_encoder, self.current_momentum)
        ema_update(self.student_head, self.teacher_head, self.current_momentum)

