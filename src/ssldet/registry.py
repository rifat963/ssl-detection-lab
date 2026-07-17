from __future__ import annotations

import torch
import torch.nn as nn

from .config import PretrainConfig
from .methods import BYOL, DINOv2, IJEPA, MAE, MoCo, SimCLR


@torch.no_grad()
def infer_pooled_dimension(encoder: nn.Module, image_size: int, device: torch.device) -> int:
    was_training = encoder.training
    encoder.eval()
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    features = encoder(sample)
    encoder.train(was_training)
    return int(features.shape[1])


def build_method(
    config: PretrainConfig,
    encoder: nn.Module,
    device: torch.device,
) -> nn.Module:
    feature_channels, _, _ = encoder.infer_dimensions(config.image_size, device)
    feature_dim = infer_pooled_dimension(encoder, config.image_size, device)

    if config.method == "simclr":
        method = SimCLR(
            encoder,
            feature_dim,
            config.hidden_dim,
            config.projection_dim,
            config.temperature,
        )
    elif config.method == "byol":
        method = BYOL(
            encoder,
            feature_dim,
            config.hidden_dim,
            config.projection_dim,
            config.momentum,
        )
    elif config.method == "moco":
        method = MoCo(
            encoder,
            feature_dim,
            config.hidden_dim,
            config.projection_dim,
            config.temperature,
            config.momentum,
            config.queue_size,
        )
    elif config.method == "dinov2":
        method = DINOv2(
            encoder,
            feature_dim,
            config.hidden_dim,
            config.projection_dim,
            config.dino_output_dim,
            config.student_temperature,
            config.teacher_temperature,
            config.center_momentum,
            config.momentum,
            config.koleo_weight,
        )
    elif config.method == "mae":
        method = MAE(encoder, feature_channels, config.mask_ratio)
    elif config.method == "ijepa":
        method = IJEPA(
            encoder,
            feature_channels,
            config.projection_dim,
            config.predictor_depth,
            config.predictor_heads,
            config.momentum,
            config.num_target_blocks,
            (config.target_scale_min, config.target_scale_max),
            (config.target_aspect_min, config.target_aspect_max),
        )
    else:  # guarded by config validation
        raise KeyError(config.method)

    return method.to(device)
