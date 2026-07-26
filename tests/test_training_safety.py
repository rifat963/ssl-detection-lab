from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ssldet.trainer as trainer
from ssldet.config import PretrainConfig
from ssldet.methods.moco import MoCo
from ssldet.methods.simclr import NTXentLoss


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(3, 8)

    def forward(self, images):
        return self.projection(self.pool(images).flatten(1))


def test_short_final_accumulation_group_uses_its_actual_size():
    assert [trainer._accumulation_group_size(index, 5, 3) for index in range(1, 6)] == [
        3,
        3,
        3,
        2,
        2,
    ]


def test_pretrain_cleans_up_when_setup_fails(monkeypatch):
    calls = []
    context = SimpleNamespace(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    monkeypatch.setattr(trainer, "initialize_distributed", lambda: context)
    monkeypatch.setattr(trainer, "cleanup_distributed", lambda: calls.append("cleanup"))
    monkeypatch.setattr(trainer, "_pretrain", lambda config, distributed: (_ for _ in ()).throw(RuntimeError("setup")))

    with pytest.raises(RuntimeError, match="setup"):
        trainer.pretrain(PretrainConfig(method="mae", image_roots=["unused"]))

    assert calls == ["cleanup"]


def test_moco_enqueues_every_accumulated_microbatch():
    method = MoCo(
        TinyEncoder(),
        feature_dim=8,
        hidden_dim=16,
        projection_dim=4,
        temperature=0.2,
        momentum=0.99,
        queue_size=8,
    )
    for _ in range(2):
        method((torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)))

    method.after_optimizer_step()

    assert method.queue_pointer.item() == 4


def test_simclr_rejects_a_degenerate_single_sample_batch():
    with pytest.raises(ValueError, match="at least two samples"):
        NTXentLoss()(torch.randn(1, 4), torch.randn(1, 4))
