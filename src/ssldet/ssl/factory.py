"""Extensible registry for creating SSL modules with arbitrary encoders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from ..methods import BYOL, IJEPA, MAE, DINOv2, DINOv3Distillation, MoCo, SimCLR
from ..methods.common import SSLMethod

SSLFactory = Callable[..., SSLMethod]
_SSL_MODULES: dict[str, SSLFactory] = {
    "simclr": SimCLR,
    "byol": BYOL,
    "moco": MoCo,
    "dinov2": DINOv2,
    "dinov3": DINOv3Distillation,
    "mae": MAE,
    "ijepa": IJEPA,
}


def available_ssl_modules() -> tuple[str, ...]:
    """Return registered SSL objective names in stable alphabetical order."""

    return tuple(sorted(_SSL_MODULES))


def ssl_module_requires_two_views(name: str) -> bool:
    """Return whether an objective needs at least two images per process.

    Reads the ``requires_two_views`` class attribute so the trainer's dataset guard
    stays in step with the registry instead of repeating a hand-written method list.
    Factories that are plain callables rather than classes are treated as single-view.
    """

    return bool(getattr(_SSL_MODULES.get(name.lower().strip()), "requires_two_views", False))


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
    "ssl_module_requires_two_views",
]
