"""Public base types and reusable building blocks for SSL objectives."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ..methods.common import ProjectionMLP, SSLMethod, ema_update, frozen_copy


@runtime_checkable
class Encoder(Protocol):
    """Smallest encoder interface accepted by pooled-feature SSL objectives."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class SpatialEncoder(Encoder, Protocol):
    """Encoder interface required by spatial objectives such as MAE and I-JEPA."""

    def forward_feature_map(self, images: torch.Tensor) -> torch.Tensor: ...


__all__ = [
    "Encoder",
    "ProjectionMLP",
    "SSLMethod",
    "SpatialEncoder",
    "ema_update",
    "frozen_copy",
]
