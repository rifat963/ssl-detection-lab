from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ProjectionMLP, SSLMethod, ema_update, frozen_copy


def negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=-1).mean()


class BYOL(SSLMethod):
    requires_two_views = True

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        momentum: float,
    ) -> None:
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = ProjectionMLP(feature_dim, hidden_dim, projection_dim)
        self.predictor = ProjectionMLP(projection_dim, hidden_dim, projection_dim)
        self.target_encoder = frozen_copy(encoder)
        self.target_projector = frozen_copy(self.online_projector)
        self.current_momentum = momentum

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    def _online(self, images: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.online_projector(self.online_encoder(images)))

    @torch.no_grad()
    def _target(self, images: torch.Tensor) -> torch.Tensor:
        return self.target_projector(self.target_encoder(images))

    def forward(self, batch) -> torch.Tensor:
        first, second = batch
        first_online = self._online(first)
        second_online = self._online(second)
        with torch.no_grad():
            first_target = self._target(first)
            second_target = self._target(second)
        return 0.5 * (
            negative_cosine_similarity(first_online, second_target)
            + negative_cosine_similarity(second_online, first_target)
        )

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        ema_update(self.online_encoder, self.target_encoder, self.current_momentum)
        ema_update(self.online_projector, self.target_projector, self.current_momentum)
