import torch
import torch.nn as nn

from ssldet.backbones.dinov3 import (
    DINOV3_SPECS,
    DINOv3FeatureEncoder,
    load_dinov3_backbone,
)


class FakeOfficialDINOv3(nn.Module):
    def forward_features(self, images):
        batch_size = images.shape[0]
        patches = (images.shape[-2] // 16) * (images.shape[-1] // 16)
        return {
            "x_norm_clstoken": torch.zeros(batch_size, 384),
            "x_norm_patchtokens": torch.zeros(batch_size, patches, 384),
        }


def test_dinov3_catalog_has_official_pretrained_backbone_families():
    assert len(DINOV3_SPECS) == 10
    assert "dinov3_vitb16" in DINOV3_SPECS
    assert "dinov3_convnext_large" in DINOV3_SPECS


def test_dinov3_vit_wrapper_exports_global_tokens_and_dense_features():
    encoder = DINOv3FeatureEncoder(FakeOfficialDINOv3(), "dinov3_vits16")
    images = torch.randn(2, 3, 64, 80)

    assert encoder(images).shape == (2, 384)
    assert encoder.forward_tokens(images).shape == (2, 20, 384)
    assert encoder.forward_feature_map(images).shape == (2, 384, 4, 5)


def test_dinov3_loader_forwards_user_supplied_weights(monkeypatch, tmp_path):
    captured = {}

    def fake_hub_load(repository, model_name, **kwargs):
        captured.update(repository=repository, model_name=model_name, **kwargs)
        return FakeOfficialDINOv3()

    monkeypatch.setattr(torch, "__version__", "2.7.1")
    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    weights = tmp_path / "dinov3_vits16.pth"
    encoder = load_dinov3_backbone(
        "dinov3_vits16",
        weights=weights,
        repository=tmp_path / "dinov3",
        source="local",
    )

    assert isinstance(encoder, DINOv3FeatureEncoder)
    assert captured["weights"] == str(weights)
    assert captured["source"] == "local"
