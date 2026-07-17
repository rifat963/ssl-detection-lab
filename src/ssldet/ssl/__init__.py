"""Reusable self-supervised learning objectives and extension interfaces."""

from ..methods import BYOL, DINOv2, DINOv3Distillation, IJEPA, MAE, MoCo, SimCLR
from ..methods.byol import negative_cosine_similarity
from ..methods.dinov2 import DINOHead, dino_cross_view_loss, koleo_loss
from ..methods.dinov3 import cosine_regression
from ..methods.ijepa import sample_target_blocks, sinusoidal_2d_position
from ..methods.mae import random_patch_mask
from ..methods.simclr import NTXentLoss
from .base import Encoder, ProjectionMLP, SSLMethod, SpatialEncoder, ema_update, frozen_copy
from .factory import available_ssl_modules, create_ssl_module, register_ssl_module

__all__ = [
    "BYOL",
    "DINOHead",
    "DINOv2",
    "DINOv3Distillation",
    "Encoder",
    "IJEPA",
    "MAE",
    "MoCo",
    "NTXentLoss",
    "ProjectionMLP",
    "SSLMethod",
    "SimCLR",
    "SpatialEncoder",
    "available_ssl_modules",
    "create_ssl_module",
    "cosine_regression",
    "dino_cross_view_loss",
    "ema_update",
    "frozen_copy",
    "koleo_loss",
    "negative_cosine_similarity",
    "random_patch_mask",
    "register_ssl_module",
    "sample_target_blocks",
    "sinusoidal_2d_position",
]
