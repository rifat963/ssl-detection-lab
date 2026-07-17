from pathlib import Path

import torch
import torch.nn as nn

import ssldet.downstream as downstream


class FakeEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.layers


class FakeTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2)])


class FakeYOLO:
    last_instance = None

    def __init__(self, source):
        self.source = source
        self.model = FakeTaskModel()
        self.saved = None
        FakeYOLO.last_instance = self

    def save(self, path):
        self.saved = path
        Path(path).write_bytes(b"detector")


def test_transfer_simclr_online_encoder(monkeypatch, tmp_path):
    source_model = FakeTaskModel()
    expected = {f"online_encoder.{key}": value for key, value in source_model.state_dict().items()}
    checkpoint = tmp_path / "best_ssl.pt"
    torch.save(
        {
            "method": expected,
            "config": {"method": "simclr", "yolo_model": "yolo26n.yaml"},
        },
        checkpoint,
    )
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    monkeypatch.setattr("ssldet.backbones.YOLOBackboneEncoder", FakeEncoder)

    result = downstream.transfer_ssl_backbone_to_yolo(
        checkpoint,
        tmp_path / "initialized.pt",
    )

    assert result.ssl_method == "simclr"
    assert result.encoder_prefix == "online_encoder."
    assert result.coverage == 1.0
    assert result.detector_checkpoint.is_file()
    assert result.report_json.is_file()


def test_transfer_rejects_an_encoder_architecture_mismatch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best_ssl.pt"
    torch.save(
        {
            "method": {"online_encoder.unrelated": torch.ones(1)},
            "config": {"method": "simclr", "yolo_model": "yolo26n.yaml"},
        },
        checkpoint,
    )
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    monkeypatch.setattr("ssldet.backbones.YOLOBackboneEncoder", FakeEncoder)

    try:
        downstream.transfer_ssl_backbone_to_yolo(checkpoint, tmp_path / "initialized.pt")
    except RuntimeError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("Expected an incompatible backbone to be rejected")
