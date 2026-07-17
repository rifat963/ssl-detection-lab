"""Extensible registry for object-detection backends."""

from __future__ import annotations

from collections.abc import Callable

from .base import ObjectDetector
from .ultralytics import UltralyticsDetector


DetectorFactory = Callable[[str, str], ObjectDetector]
_DETECTION_BACKENDS: dict[str, DetectorFactory] = {
    "ultralytics": UltralyticsDetector.load,
}


def available_detection_backends() -> tuple[str, ...]:
    return tuple(sorted(_DETECTION_BACKENDS))


def register_detection_backend(
    name: str,
    factory: DetectorFactory,
    *,
    replace: bool = False,
) -> None:
    normalized = name.lower().strip()
    if not normalized:
        raise ValueError("Detection backend name cannot be empty")
    if normalized in _DETECTION_BACKENDS and not replace:
        raise KeyError(f"Detection backend {normalized!r} is already registered")
    _DETECTION_BACKENDS[normalized] = factory


def create_detector(
    model_name: str,
    weights_file: str,
    *,
    backend: str = "ultralytics",
) -> ObjectDetector:
    normalized = backend.lower().strip()
    try:
        factory = _DETECTION_BACKENDS[normalized]
    except KeyError as error:
        raise KeyError(
            f"Unknown detection backend {backend!r}; "
            f"choose from {available_detection_backends()}"
        ) from error
    detector = factory(model_name, weights_file)
    if not isinstance(detector, ObjectDetector):
        raise TypeError("A detection factory must return an ObjectDetector-compatible instance")
    return detector


__all__ = [
    "DetectorFactory",
    "available_detection_backends",
    "create_detector",
    "register_detection_backend",
]
