"""DINOv3 feature extraction with a user-supplied official checkpoint."""

from pathlib import Path

import torch
from PIL import Image

from ssldet import build_dinov3_transform, load_dinov3_backbone


WEIGHTS = Path("/path/to/dinov3_vitb16_pretrain_lvd1689m.pth")
IMAGE = Path("/path/to/image.jpg")

encoder = load_dinov3_backbone(
    "dinov3_vitb16",
    weights=WEIGHTS,
    device="cuda",
)
transform = build_dinov3_transform(256, weights_dataset="lvd1689m")
with Image.open(IMAGE) as image:
    batch = transform(image.convert("RGB")).unsqueeze(0).to("cuda")

with torch.inference_mode():
    global_embedding = encoder(batch)
    patch_tokens = encoder.forward_tokens(batch)
    feature_map = encoder.forward_feature_map(batch)

print(global_embedding.shape, patch_tokens.shape, feature_map.shape)
