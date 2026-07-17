from __future__ import annotations

import torch
import torch.nn as nn

from .config import PretrainConfig
from .ssl import create_ssl_module


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

    arguments = {
        "simclr": {
            "feature_dim": feature_dim,
            "hidden_dim": config.hidden_dim,
            "projection_dim": config.projection_dim,
            "temperature": config.temperature,
        },
        "byol": {
            "feature_dim": feature_dim,
            "hidden_dim": config.hidden_dim,
            "projection_dim": config.projection_dim,
            "momentum": config.momentum,
        },
        "moco": {
            "feature_dim": feature_dim,
            "hidden_dim": config.hidden_dim,
            "projection_dim": config.projection_dim,
            "temperature": config.temperature,
            "momentum": config.momentum,
            "queue_size": config.queue_size,
        },
        "dinov2": {
            "feature_dim": feature_dim,
            "hidden_dim": config.hidden_dim,
            "bottleneck_dim": config.projection_dim,
            "output_dim": config.dino_output_dim,
            "student_temperature": config.student_temperature,
            "teacher_temperature": config.teacher_temperature,
            "center_momentum": config.center_momentum,
            "momentum": config.momentum,
            "koleo_weight": config.koleo_weight,
        },
        "mae": {
            "feature_channels": feature_channels,
            "mask_ratio": config.mask_ratio,
        },
        "ijepa": {
            "feature_channels": feature_channels,
            "projection_dim": config.projection_dim,
            "predictor_depth": config.predictor_depth,
            "predictor_heads": config.predictor_heads,
            "momentum": config.momentum,
            "num_target_blocks": config.num_target_blocks,
            "target_scale": (config.target_scale_min, config.target_scale_max),
            "target_aspect": (config.target_aspect_min, config.target_aspect_max),
        },
    }
    try:
        method_arguments = arguments[config.method]
    except KeyError as error:  # guarded by config validation
        raise KeyError(config.method) from error
    method = create_ssl_module(config.method, encoder, **method_arguments)

    return method.to(device)
