import torch
import torch.nn as nn

from ssldet.backbones.dinov2 import DINOv2FeatureEncoder


class FakeOfficialDINOv2(nn.Module):
    def forward_features(self, images):
        batch_size = images.shape[0]
        patches = (images.shape[-2] // 14) * (images.shape[-1] // 14)
        return {
            "x_norm_clstoken": torch.zeros(batch_size, 384),
            "x_norm_patchtokens": torch.zeros(batch_size, patches, 384),
        }


def test_official_dinov2_wrapper_exports_global_and_dense_features():
    encoder = DINOv2FeatureEncoder(FakeOfficialDINOv2(), "dinov2_vits14")
    images = torch.randn(2, 3, 56, 70)

    assert encoder(images).shape == (2, 384)
    assert encoder.forward_tokens(images).shape == (2, 20, 384)
    assert encoder.forward_feature_map(images).shape == (2, 384, 4, 5)

