"""Extensible registry for creating SSL modules with arbitrary encoders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from ..methods import BYOL, DINOv2, IJEPA, MAE, MoCo, SimCLR
from ..methods.common import SSLMethod


SSLFactory = Callable[..., SSLMethod]
_SSL_MODULES: dict[str, SSLFactory] = {
    "simclr": SimCLR,
    "byol": BYOL,
    "moco": MoCo,
    "dinov2": DINOv2,
    "mae": MAE,
    "ijepa": IJEPA,
}


def available_ssl_modules() -> tuple[str, ...]:
    """Return registered SSL objective names in stable alphabetical order."""

    return tuple(sorted(_SSL_MODULES))


def register_ssl_module(name: str, factory: SSLFactory, *, replace: bool = False) -> None:
    """Register a custom SSL objective factory.

    A factory receives ``encoder=...`` plus keyword arguments passed to
    :func:`create_ssl_module` and must return an ``SSLMethod``.
    """

    normalized = name.lower().strip()
    if not normalized:
        raise ValueError("SSL module name cannot be empty")
    if normalized in _SSL_MODULES and not replace:
        raise KeyError(f"SSL module {normalized!r} is already registered")
    _SSL_MODULES[normalized] = factory


def create_ssl_module(name: str, encoder: nn.Module, **kwargs: Any) -> SSLMethod:
    """Create a reusable SSL module around any compatible PyTorch encoder."""

    normalized = name.lower().strip()
    try:
        factory = _SSL_MODULES[normalized]
    except KeyError as error:
        raise KeyError(
            f"Unknown SSL module {name!r}; choose from {available_ssl_modules()}"
        ) from error
    module = factory(encoder=encoder, **kwargs)
    if not isinstance(module, SSLMethod):
        raise TypeError("An SSL factory must return an SSLMethod instance")
    return module


__all__ = [
    "SSLFactory",
    "available_ssl_modules",
    "create_ssl_module",
    "register_ssl_module",
]
