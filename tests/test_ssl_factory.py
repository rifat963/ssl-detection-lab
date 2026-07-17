import torch
import torch.nn as nn

from ssldet.ssl import (
    SSLMethod,
    available_ssl_modules,
    create_ssl_module,
    register_ssl_module,
)


class IdentitySSL(SSLMethod):
    def __init__(self, encoder: nn.Module, scale: float = 1.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.scale = scale

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images).mean() * self.scale


def test_builtin_ssl_modules_are_publicly_registered():
    assert set(available_ssl_modules()) == {
        "byol",
        "dinov2",
        "ijepa",
        "mae",
        "moco",
        "simclr",
    }


def test_custom_ssl_module_can_be_registered_and_created():
    register_ssl_module("identity_test", IdentitySSL, replace=True)
    module = create_ssl_module("identity_test", nn.Flatten(), scale=2.0)

    assert isinstance(module, IdentitySSL)
    assert module.scale == 2.0
