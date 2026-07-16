"""Small, readable SSL building blocks for object-detection backbones."""

from .config import PretrainConfig
from .trainer import PretrainResult, pretrain

__all__ = ["PretrainConfig", "PretrainResult", "pretrain"]
__version__ = "0.1.0"

