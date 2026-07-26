from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ProjectionMLP(nn.Module):
    """LayerNorm makes the head stable with small per-GPU Kaggle batches."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def frozen_copy(module: nn.Module) -> nn.Module:
    clone = copy.deepcopy(module)
    clone.requires_grad_(False)
    clone.eval()
    return clone


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, momentum: float) -> None:
    for online_parameter, target_parameter in zip(
        online.parameters(), target.parameters(), strict=True
    ):
        target_parameter.data.mul_(momentum).add_(online_parameter.data, alpha=1.0 - momentum)
    for online_buffer, target_buffer in zip(online.buffers(), target.buffers(), strict=True):
        target_buffer.copy_(online_buffer)


class SSLMethod(nn.Module):
    """Minimal interface consumed by the shared trainer."""

    requires_two_views: bool = False

    def set_momentum(self, momentum: float) -> None:
        self.current_momentum = momentum

    def after_optimizer_step(self) -> None:
        return None
