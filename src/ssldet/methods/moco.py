from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import concat_all_gather
from .common import ProjectionMLP, SSLMethod, ema_update, frozen_copy


class MoCo(SSLMethod):
    requires_two_views = True

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        projection_dim: int,
        temperature: float,
        momentum: float,
        queue_size: int,
    ) -> None:
        super().__init__()
        self.online_encoder = encoder
        self.query_projector = ProjectionMLP(feature_dim, hidden_dim, projection_dim)
        self.target_encoder = frozen_copy(encoder)
        self.key_projector = frozen_copy(self.query_projector)
        self.temperature = temperature
        self.current_momentum = momentum

        queue = F.normalize(torch.randn(projection_dim, queue_size), dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))
        # Keys are held until the optimizer steps so a gradient-accumulation group
        # enqueues one batch of negatives, matching the effective batch size.
        self._pending_keys: list[torch.Tensor] = []

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.key_projector.eval()
        return self

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        keys = concat_all_gather(keys.detach())
        count = keys.shape[0]
        size = self.queue.shape[1]
        if count >= size:
            self.queue.copy_(keys[-size:].T)
            self.queue_pointer.zero_()
            return
        pointer = int(self.queue_pointer.item())
        end = pointer + count
        if end <= size:
            self.queue[:, pointer:end] = keys.T
        else:
            first = size - pointer
            self.queue[:, pointer:] = keys[:first].T
            self.queue[:, : end - size] = keys[first:].T
        self.queue_pointer[0] = end % size

    def forward(self, batch) -> torch.Tensor:
        query_images, key_images = batch
        queries = F.normalize(self.query_projector(self.online_encoder(query_images)), dim=1)
        with torch.no_grad():
            keys = F.normalize(self.key_projector(self.target_encoder(key_images)), dim=1)

        positive = torch.einsum("nc,nc->n", queries, keys).unsqueeze(1)
        negative = torch.einsum("nc,ck->nk", queries, self.queue.detach().clone())
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.float(), labels)
        self._pending_keys.append(keys)
        return loss

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        ema_update(self.online_encoder, self.target_encoder, self.current_momentum)
        ema_update(self.query_projector, self.key_projector, self.current_momentum)
        if self._pending_keys:
            self._enqueue(torch.cat(self._pending_keys, dim=0))
            self._pending_keys.clear()
