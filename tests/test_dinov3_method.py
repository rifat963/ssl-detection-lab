import torch
import torch.nn as nn
import torch.nn.functional as F

from ssldet.methods.dinov3 import DINOv3Distillation, cosine_regression


class StudentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward_feature_map(self, images):
        return F.avg_pool2d(self.conv(images), 4)


class TeacherEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 12, 3, padding=1)

    def forward_global_and_dense(self, images):
        dense = F.avg_pool2d(self.conv(images), 2)
        return dense.mean(dim=(-2, -1)), dense


def test_cosine_regression_is_zero_for_identical_features():
    features = torch.randn(4, 8)
    assert torch.allclose(
        cosine_regression(features, features, dim=1),
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_dinov3_distillation_trains_only_the_student():
    teacher = TeacherEncoder()
    method = DINOv3Distillation(
        StudentEncoder(),
        teacher,
        feature_channels=8,
        feature_dim=8,
        teacher_dim=12,
    )
    loss = method(torch.randn(2, 3, 32, 32))
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not any(key.startswith("teacher.") for key in method.state_dict())
    method.load_state_dict(method.state_dict())
