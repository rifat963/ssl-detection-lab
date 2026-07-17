from pathlib import Path

import pytest

from ssldet.dinov3 import DINOv3YOLOConfig, pretrain_dinov3_yolo26


class FakeLightlyTrain:
    def __init__(self):
        self.arguments = None

    @staticmethod
    def list_methods():
        return ["distillation"]

    @staticmethod
    def list_models():
        return ["dinov3/vitb16", "ultralytics/yolo26n.yaml"]

    def pretrain(self, **arguments):
        self.arguments = arguments
        output = Path(arguments["out"])
        (output / "checkpoints").mkdir(parents=True)
        (output / "exported_models").mkdir(parents=True)
        (output / "checkpoints" / "last.ckpt").touch()
        (output / "exported_models" / "exported_last.pt").touch()


def test_dinov3_yolo26_workflow_uses_local_vitb16_weights(tmp_path):
    weights = tmp_path / "dinov3_vitb16.pth"
    weights.touch()
    output = tmp_path / "run"
    module = FakeLightlyTrain()
    config = DINOv3YOLOConfig(
        data=str(tmp_path / "images"),
        output_dir=str(output),
        teacher_weights=str(weights),
        epochs=2,
        batch_size=4,
    )

    result = pretrain_dinov3_yolo26(config, module=module)

    assert result.yolo_checkpoint == output / "exported_models" / "exported_last.pt"
    assert module.arguments["method_args"] == {
        "teacher": "dinov3/vitb16",
        "teacher_weights": str(weights),
    }
    assert module.arguments["model"] == "ultralytics/yolo26n.yaml"
    assert module.arguments["precision"] == "16-mixed"


def test_dinov3_yolo26_workflow_requires_weights(tmp_path):
    config = DINOv3YOLOConfig(
        data=str(tmp_path / "images"),
        teacher_weights=str(tmp_path / "missing.pth"),
    )

    with pytest.raises(FileNotFoundError, match="weights were not found"):
        config.validate()
