import torch
import torch.nn as nn

from ssldet.methods.dinov2 import DINOv2, dino_cross_view_loss, koleo_loss


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(3, 8)

    def forward(self, images):
        return self.projection(self.pool(images).flatten(1))


def test_dino_cross_view_loss_is_finite():
    student = [torch.randn(3, 16, requires_grad=True) for _ in range(4)]
    teacher = [torch.randn(3, 16) for _ in range(2)]
    loss = dino_cross_view_loss(student, teacher, torch.zeros(1, 16), 0.1, 0.04)
    assert torch.isfinite(loss)
    loss.backward()


def test_koleo_handles_single_item_batch():
    features = torch.randn(1, 8, requires_grad=True)
    loss = koleo_loss(features)
    assert loss.item() == 0.0


def test_dinov2_accepts_global_and_local_crops():
    method = DINOv2(
        TinyEncoder(),
        feature_dim=8,
        hidden_dim=16,
        bottleneck_dim=8,
        output_dim=16,
        student_temperature=0.1,
        teacher_temperature=0.04,
        center_momentum=0.9,
        momentum=0.996,
        koleo_weight=0.1,
    )
    views = (
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 3, 16, 16),
    )
    loss = method(views)
    assert torch.isfinite(loss)
    loss.backward()
    method.after_optimizer_step()

