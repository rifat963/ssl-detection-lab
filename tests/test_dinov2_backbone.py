import torch
import torch.nn as nn

from ssldet.backbones.dinov2 import DINOv2FeatureEncoder, _checkpoint_state, load_dinov2_backbone


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


def test_dinov2_loader_enables_official_pretrained_weights_by_default(monkeypatch):
    captured = {}

    def fake_hub_load(repository, model_name, **kwargs):
        captured.update(repository=repository, model_name=model_name, **kwargs)
        return FakeOfficialDINOv2()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    encoder = load_dinov2_backbone("dinov2_vits14")

    assert isinstance(encoder, DINOv2FeatureEncoder)
    assert captured["pretrained"] is True


def test_nested_official_checkpoint_wrappers_are_unpacked():
    weight = torch.ones(1)

    assert _checkpoint_state({"teacher": {"backbone": {"module.weight": weight}}}) == {
        "weight": weight
    }
