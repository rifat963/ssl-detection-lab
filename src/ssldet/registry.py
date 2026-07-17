from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import DINOV3_SPECS, load_dinov3_backbone
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

    dinov3_teacher = None
    dinov3_spec = DINOV3_SPECS.get(config.dinov3_model)
    if config.method == "dinov3":
        if dinov3_spec is None:
            raise ValueError(
                f"Unknown DINOv3 model {config.dinov3_model!r}; "
                f"choose from {sorted(DINOV3_SPECS)}"
            )
        if config.image_size % dinov3_spec.patch_size:
            raise ValueError(
                f"image_size must be divisible by {dinov3_spec.patch_size} for "
                f"{config.dinov3_model}"
            )
        dinov3_teacher = load_dinov3_backbone(
            config.dinov3_model,
            weights=config.dinov3_weights,
            repository=config.dinov3_repository,
            source=config.dinov3_source,
            device=device,
            freeze=True,
        )

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
        "dinov3": {
            "teacher": dinov3_teacher,
            "feature_channels": feature_channels,
            "feature_dim": feature_dim,
            "teacher_dim": dinov3_spec.embedding_dim if dinov3_spec else 0,
            "global_weight": config.dinov3_global_weight,
            "dense_weight": config.dinov3_dense_weight,
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
