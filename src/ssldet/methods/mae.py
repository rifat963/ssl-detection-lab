from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SSLMethod


def random_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    tokens = height * width
    masked = max(1, min(tokens - 1, round(tokens * ratio)))
    noise = torch.rand(batch_size, tokens, device=device)
    indices = noise.argsort(dim=1)[:, :masked]
    mask = torch.zeros(batch_size, tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask.view(batch_size, 1, height, width)


class MAE(SSLMethod):
    """Compute-scaled masked autoencoder adapted to a spatial YOLO backbone.

    This is intentionally a CNN-compatible educational adaptation, not the
    original ViT MAE architecture. It keeps the defining masked-pixel
    reconstruction objective while making the learned YOLO modules directly
    transferable to a detector.
    """

    def __init__(self, encoder: nn.Module, feature_channels: int, mask_ratio: float) -> None:
        super().__init__()
        self.online_encoder = encoder
        hidden = max(64, feature_channels // 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        self.mask_ratio = mask_ratio

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = images.shape
        grid_height = max(2, image_height // 16)
        grid_width = max(2, image_width // 16)
        patch_mask = random_patch_mask(
            batch_size, grid_height, grid_width, self.mask_ratio, images.device
        )
        pixel_mask = F.interpolate(patch_mask.float(), size=(image_height, image_width), mode="nearest")
        masked_images = images.masked_fill(pixel_mask.bool(), 0.0)
        features = self.online_encoder.forward_feature_map(masked_images)
        reconstruction = self.decoder(features)
        reconstruction = F.interpolate(
            reconstruction, size=(image_height, image_width), mode="bilinear", align_corners=False
        )
        squared_error = (reconstruction - images).pow(2)
        return (squared_error * pixel_mask).sum() / (pixel_mask.sum() * images.shape[1]).clamp_min(1.0)

