from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SSLMethod, ema_update, frozen_copy


def sinusoidal_2d_position(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4:
        raise ValueError("I-JEPA projection_dim must be divisible by 4")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4)))
    x = x.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).unsqueeze(0)


def sample_target_blocks(
    batch_size: int,
    height: int,
    width: int,
    number: int,
    scale: tuple[float, float],
    aspect: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Return BxKxHxW rectangular target masks."""

    masks = torch.zeros(batch_size, number, height, width, dtype=torch.bool, device=device)
    grid_area = height * width
    for batch_index in range(batch_size):
        for block_index in range(number):
            area = grid_area * float(torch.empty(1).uniform_(*scale).item())
            ratio = float(torch.empty(1).uniform_(*aspect).item())
            block_height = max(1, min(height, round(math.sqrt(area / ratio))))
            block_width = max(1, min(width, round(math.sqrt(area * ratio))))
            top = int(torch.randint(0, height - block_height + 1, (1,)).item())
            left = int(torch.randint(0, width - block_width + 1, (1,)).item())
            masks[
                batch_index,
                block_index,
                top : top + block_height,
                left : left + block_width,
            ] = True
    return masks


class LatentPredictor(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, tokens: torch.Tensor, target_union: torch.Tensor) -> torch.Tensor:
        mask_tokens = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        predictor_input = torch.where(target_union.unsqueeze(-1), mask_tokens, tokens)
        return self.transformer(predictor_input)


class IJEPA(SSLMethod):
    """YOLO-native, compute-scaled I-JEPA latent block-prediction objective."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_channels: int,
        projection_dim: int,
        predictor_depth: int,
        predictor_heads: int,
        momentum: float,
        num_target_blocks: int,
        target_scale: tuple[float, float],
        target_aspect: tuple[float, float],
    ) -> None:
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = nn.Conv2d(feature_channels, projection_dim, 1)
        self.target_encoder = frozen_copy(encoder)
        self.target_projector = frozen_copy(self.online_projector)
        self.predictor = LatentPredictor(projection_dim, predictor_depth, predictor_heads)
        self.current_momentum = momentum
        self.num_target_blocks = num_target_blocks
        self.target_scale = target_scale
        self.target_aspect = target_aspect

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = images.shape
        with torch.no_grad():
            target_map = self.target_projector(self.target_encoder.forward_feature_map(images))
        _, channels, grid_height, grid_width = target_map.shape
        block_masks = sample_target_blocks(
            batch_size,
            grid_height,
            grid_width,
            self.num_target_blocks,
            self.target_scale,
            self.target_aspect,
            images.device,
        )
        target_union_grid = block_masks.any(dim=1, keepdim=True)
        target_union_pixels = F.interpolate(
            target_union_grid.float(), size=(image_height, image_width), mode="nearest"
        ).bool()

        # Zero is the ImageNet-normalized dataset mean, a neutral mask value.
        context_images = images.masked_fill(target_union_pixels, 0.0)
        context_map = self.online_projector(self.online_encoder.forward_feature_map(context_images))
        context_tokens = context_map.flatten(2).transpose(1, 2)
        target_tokens = target_map.flatten(2).transpose(1, 2).detach()
        target_union = target_union_grid.flatten(2).squeeze(1)

        position = sinusoidal_2d_position(grid_height, grid_width, channels, images.device)
        predictions = self.predictor(context_tokens + position, target_union)
        predictions = F.layer_norm(predictions, (channels,))
        targets = F.layer_norm(target_tokens, (channels,))

        expanded_masks = block_masks.flatten(2).unsqueeze(-1)
        prediction_blocks = predictions[:, None, :, :].expand(-1, self.num_target_blocks, -1, -1)
        target_blocks = targets[:, None, :, :].expand_as(prediction_blocks)
        selected_predictions = prediction_blocks[expanded_masks.expand_as(prediction_blocks)]
        selected_targets = target_blocks[expanded_masks.expand_as(target_blocks)]
        return F.smooth_l1_loss(selected_predictions.float(), selected_targets.float())

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        ema_update(self.online_encoder, self.target_encoder, self.current_momentum)
        ema_update(self.online_projector, self.target_projector, self.current_momentum)
