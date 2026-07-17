from pathlib import Path

import torch
from PIL import Image

from ssldet.config import PretrainConfig
from ssldet.data import build_transform


def test_training_source_does_not_use_deprecated_cuda_amp_namespace():
    trainer = Path(__file__).parents[1] / "src" / "ssldet" / "trainer.py"
    source = trainer.read_text(encoding="utf-8")

    assert "torch.cuda.amp" not in source
    assert 'torch.amp.GradScaler("cuda"' in source
    assert "torch.amp.autocast(" in source


def test_torchvision_v2_pipeline_returns_float_tensor():
    config = PretrainConfig(
        method="mae",
        image_roots=["unused"],
        image_size=64,
    ).validate()
    transformed = build_transform(config)(Image.new("RGB", (96, 96), color="red"))

    assert isinstance(transformed, torch.Tensor)
    assert transformed.dtype == torch.float32
    assert transformed.shape == (3, 64, 64)
