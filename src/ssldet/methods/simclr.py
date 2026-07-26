from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ProjectionMLP, SSLMethod


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError(f"View shapes differ: {first.shape} vs {second.shape}")
        batch_size = first.shape[0]
        if batch_size < 2:
            raise ValueError("SimCLR requires at least two samples per process")
        representations = F.normalize(torch.cat([first, second], dim=0), dim=1)
        logits = representations @ representations.T / self.temperature
        diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        positives = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (
            2 * batch_size
        )
        return F.cross_entropy(logits.float(), positives)


class SimCLR(SSLMethod):
    requires_two_views = True

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.online_encoder = encoder
        self.projector = ProjectionMLP(feature_dim, hidden_dim, projection_dim)
        self.criterion = NTXentLoss(temperature)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.online_encoder(images)), dim=1)

    def forward(self, batch) -> torch.Tensor:
        first, second = batch
        return self.criterion(self.encode(first), self.encode(second))
