"""Official DINOv2 backbone loading and feature-map extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DINOv2Spec:
    name: str
    architecture: str
    embedding_dim: int
    patch_size: int = 14
    registers: bool = False


DINOV2_SPECS = {
    f"dinov2_{architecture}{'_reg' if registers else ''}": DINOv2Spec(
        name=f"dinov2_{architecture}{'_reg' if registers else ''}",
        architecture=f"ViT-{scale}/14",
        embedding_dim=embedding_dim,
        registers=registers,
    )
    for architecture, scale, embedding_dim in (
        ("vits14", "S", 384),
        ("vitb14", "B", 768),
        ("vitl14", "L", 1024),
        ("vitg14", "g", 1536),
    )
    for registers in (False, True)
}


class DINOv2FeatureEncoder(nn.Module):
    """Expose official DINOv2 global embeddings and dense patch feature maps."""

    def __init__(self, model: nn.Module, model_name: str) -> None:
        super().__init__()
        if model_name not in DINOV2_SPECS:
            raise ValueError(
                f"Unknown DINOv2 model {model_name!r}; choose from {sorted(DINOV2_SPECS)}"
            )
        self.model = model
        self.spec = DINOV2_SPECS[model_name]

    def feature_dict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model.forward_features(images)
        if not isinstance(features, dict):
            raise TypeError("Official DINOv2 forward_features() must return a dictionary")
        required = {"x_norm_clstoken", "x_norm_patchtokens"}
        missing = required.difference(features)
        if missing:
            raise KeyError(f"DINOv2 feature output is missing {sorted(missing)}")
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one normalized global embedding per image."""

        return self.feature_dict(images)["x_norm_clstoken"]

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized patch tokens with shape BxNxC."""

        return self.feature_dict(images)["x_norm_patchtokens"]

    def forward_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized patch features reshaped to BxCxHxW."""

        tokens = self.forward_tokens(images)
        grid_height = images.shape[-2] // self.spec.patch_size
        grid_width = images.shape[-1] // self.spec.patch_size
        expected_tokens = grid_height * grid_width
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens for input {tuple(images.shape[-2:])}, "
                f"received {tokens.shape[1]}"
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.spec.embedding_dim, grid_height, grid_width
        )


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility
        return torch.load(path, map_location="cpu")


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint
    for key in ("teacher", "model", "state_dict", "backbone"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("DINOv2 checkpoint must contain a state dictionary")

    prefixes = (
        "module.",
        "teacher.backbone.",
        "student.backbone.",
        "backbone.",
    )
    cleaned: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if isinstance(value, torch.Tensor):
            cleaned[key] = value
    if not cleaned:
        raise ValueError("No tensor weights were found in the DINOv2 checkpoint")
    return cleaned


def build_dinov2_transform(image_size: int = 518):
    """Build the normalized inference transform expected by DINOv2 backbones."""

    if image_size < 14:
        raise ValueError("image_size must be at least one 14x14 patch")
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_dinov2_backbone(
    model_name: str = "dinov2_vits14",
    *,
    pretrained: bool = True,
    weights_file: str | Path | None = None,
    repository: str | Path = "facebookresearch/dinov2",
    source: str = "github",
    device: str | torch.device | None = None,
    freeze: bool = True,
) -> DINOv2FeatureEncoder:
    """Load an official DINOv2 architecture from PyTorch Hub.

    Set ``source='local'`` and point ``repository`` to a cloned official DINOv2
    repository for offline use. When ``weights_file`` is supplied, the architecture
    is created without downloading weights and the local state dictionary is loaded.
    """

    if model_name not in DINOV2_SPECS:
        raise ValueError(f"model_name must be one of {sorted(DINOV2_SPECS)}")
    if source not in {"github", "local"}:
        raise ValueError("source must be 'github' or 'local'")
    use_hub_weights = bool(pretrained and weights_file is None)
    model = torch.hub.load(
        str(repository),
        model_name,
        source=source,
        pretrained=use_hub_weights,
    )
    if weights_file is not None:
        state = _checkpoint_state(_torch_load(weights_file))
        incompatible = model.load_state_dict(state, strict=False)
        if len(incompatible.unexpected_keys) == len(state):
            raise ValueError("The checkpoint does not match the selected DINOv2 architecture")

    encoder = DINOv2FeatureEncoder(model, model_name)
    if freeze:
        encoder.requires_grad_(False)
        encoder.eval()
    if device is not None:
        encoder.to(device)
    return encoder
