"""Official DINOv3 backbone loading with user-supplied pretrained weights."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DINOv3Spec:
    name: str
    architecture: str
    embedding_dim: int
    patch_size: int
    backbone_type: str
    official_pretrained_weights: bool = True


DINOV3_SPECS = {
    item.name: item
    for item in (
        DINOv3Spec("dinov3_vits16", "ViT-S/16", 384, 16, "vit"),
        DINOv3Spec("dinov3_vits16plus", "ViT-S+/16", 384, 16, "vit"),
        DINOv3Spec("dinov3_vitb16", "ViT-B/16", 768, 16, "vit"),
        DINOv3Spec("dinov3_vitl16", "ViT-L/16", 1024, 16, "vit"),
        DINOv3Spec("dinov3_vith16plus", "ViT-H+/16", 1280, 16, "vit"),
        DINOv3Spec("dinov3_vit7b16", "ViT-7B/16", 4096, 16, "vit"),
        DINOv3Spec("dinov3_convnext_tiny", "ConvNeXt Tiny", 768, 32, "convnext"),
        DINOv3Spec("dinov3_convnext_small", "ConvNeXt Small", 768, 32, "convnext"),
        DINOv3Spec("dinov3_convnext_base", "ConvNeXt Base", 1024, 32, "convnext"),
        DINOv3Spec("dinov3_convnext_large", "ConvNeXt Large", 1536, 32, "convnext"),
    )
}


def _numeric_version(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    padded = (parts + [0, 0, 0])[:3]
    return padded[0], padded[1], padded[2]


class DINOv3FeatureEncoder(nn.Module):
    """Expose global embeddings and dense features from official DINOv3 backbones."""

    def __init__(self, model: nn.Module, model_name: str) -> None:
        super().__init__()
        if model_name not in DINOV3_SPECS:
            raise ValueError(
                f"Unknown DINOv3 model {model_name!r}; choose from {sorted(DINOV3_SPECS)}"
            )
        self.model = model
        self.spec = DINOV3_SPECS[model_name]

    def _raw_features(self, images: torch.Tensor):
        forward_features = getattr(self.model, "forward_features", None)
        return forward_features(images) if callable(forward_features) else self.model(images)

    def feature_dict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._raw_features(images)
        if not isinstance(features, dict):
            raise TypeError(
                f"{self.spec.architecture} does not expose token dictionaries; "
                "use forward() for global embeddings"
            )
        required = {"x_norm_clstoken", "x_norm_patchtokens"}
        missing = required.difference(features)
        if missing:
            raise KeyError(f"DINOv3 feature output is missing {sorted(missing)}")
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self._raw_features(images)
        if isinstance(features, dict):
            return features["x_norm_clstoken"]
        if not isinstance(features, torch.Tensor):
            raise TypeError("DINOv3 forward_features() must return a tensor or dictionary")
        if features.ndim == 4:
            return features.mean(dim=(-2, -1))
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Unsupported DINOv3 feature shape: {tuple(features.shape)}")

    def forward_global_and_dense(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global and dense features from one teacher forward pass."""

        features = self._raw_features(images)
        if isinstance(features, dict):
            global_features = features["x_norm_clstoken"]
            tokens = features["x_norm_patchtokens"]
            height = images.shape[-2] // self.spec.patch_size
            width = images.shape[-1] // self.spec.patch_size
            if tokens.shape[1] != height * width:
                raise ValueError(
                    f"Expected {height * width} patch tokens, received {tokens.shape[1]}"
                )
            dense_features = tokens.transpose(1, 2).reshape(
                tokens.shape[0], self.spec.embedding_dim, height, width
            )
            return global_features, dense_features
        if isinstance(features, torch.Tensor) and features.ndim == 4:
            return features.mean(dim=(-2, -1)), features
        raise TypeError("DINOv3 teacher must expose a token dictionary or dense feature map")

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized ViT patch tokens with shape BxNxC."""

        return self.feature_dict(images)["x_norm_patchtokens"]

    def forward_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return dense features where the selected official backbone exposes them."""

        features = self._raw_features(images)
        if isinstance(features, torch.Tensor) and features.ndim == 4:
            return features
        if isinstance(features, dict):
            tokens = features["x_norm_patchtokens"]
            height = images.shape[-2] // self.spec.patch_size
            width = images.shape[-1] // self.spec.patch_size
            if tokens.shape[1] != height * width:
                raise ValueError(
                    f"Expected {height * width} patch tokens, received {tokens.shape[1]}"
                )
            return tokens.transpose(1, 2).reshape(
                tokens.shape[0], self.spec.embedding_dim, height, width
            )
        raise TypeError(
            f"{self.spec.architecture} exposes pooled features only through this adapter"
        )


def build_dinov3_transform(
    image_size: int = 256,
    *,
    weights_dataset: str = "lvd1689m",
):
    """Build the official DINOv3 web- or satellite-weight inference transform."""

    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    normalized_dataset = weights_dataset.lower().replace("-", "")
    statistics = {
        "lvd1689m": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "sat493m": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
    }
    if normalized_dataset not in statistics:
        raise ValueError("weights_dataset must be 'lvd1689m' or 'sat493m'")
    mean, standard_deviation = statistics[normalized_dataset]
    from torchvision.transforms import v2

    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=standard_deviation),
        ]
    )


def load_dinov3_backbone(
    model_name: str = "dinov3_vits16",
    *,
    weights: str | Path | None = None,
    repository: str | Path = "facebookresearch/dinov3",
    source: str = "github",
    device: str | torch.device | None = None,
    freeze: bool = True,
) -> DINOv3FeatureEncoder:
    """Load an official DINOv3 architecture and optional pretrained weights.

    ``weights`` may be a local checkpoint path or an authorized URL received from
    Meta. When omitted, the architecture is created with random weights. Supplying
    DINOv3 weights does not change their separate DINOv3 License terms.
    """

    if model_name not in DINOV3_SPECS:
        raise ValueError(f"model_name must be one of {sorted(DINOV3_SPECS)}")
    if source not in {"github", "local"}:
        raise ValueError("source must be 'github' or 'local'")
    if _numeric_version(torch.__version__) < (2, 7, 1):
        raise RuntimeError("The official DINOv3 repository requires PyTorch 2.7.1+")

    model = torch.hub.load(
        str(repository),
        model_name,
        source=source,
        weights=str(weights) if weights is not None else None,
        trust_repo=True,
    )
    encoder = DINOv3FeatureEncoder(model, model_name)
    if freeze:
        encoder.requires_grad_(False)
        encoder.eval()
    if device is not None:
        encoder.to(device)
    return encoder


__all__ = [
    "DINOV3_SPECS",
    "DINOv3FeatureEncoder",
    "DINOv3Spec",
    "build_dinov3_transform",
    "load_dinov3_backbone",
]
