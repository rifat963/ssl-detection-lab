"""Ultralytics object-detection backend and third-party license notice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..catalog import resolve_model_family

ULTRALYTICS_LICENSE_NOTICE = {
    "dependency": "ultralytics",
    "open_source_license": "AGPL-3.0",
    "commercial_option": "Ultralytics Enterprise License",
    "license_url": "https://www.ultralytics.com/license",
    "documentation_url": "https://docs.ultralytics.com/",
    "notice": (
        "ssl-detection-lab's MIT license does not replace the license terms of the "
        "Ultralytics software or model assets. Review AGPL-3.0 obligations or obtain "
        "an Ultralytics Enterprise License when appropriate."
    ),
}


def _load_ultralytics_model(model_name: str, weights_file: str):
    family = resolve_model_family(model_name)
    if family.name == "RT-DETR":
        from ultralytics import RTDETR

        return RTDETR(weights_file)
    if family.name == "YOLO-NAS":
        from ultralytics import NAS

        return NAS(weights_file)
    from ultralytics import YOLO

    return YOLO(weights_file)


@dataclass
class UltralyticsDetector:
    """Reusable adapter over YOLO, RT-DETR, YOLO-NAS, and custom Ultralytics weights."""

    model_name: str
    weights_file: str
    model: Any

    @classmethod
    def load(cls, model_name: str, weights_file: str) -> "UltralyticsDetector":
        return cls(model_name, weights_file, _load_ultralytics_model(model_name, weights_file))

    def predict(self, **kwargs: Any) -> Any:
        return self.model.predict(**kwargs)

    def track(self, **kwargs: Any) -> Any:
        return self.model.track(**kwargs)

    def validate(self, **kwargs: Any) -> Any:
        return self.model.val(**kwargs)


def load_detector(model_name: str, weights_file: str) -> UltralyticsDetector:
    """Load an object detector through the supported Ultralytics backend."""

    return UltralyticsDetector.load(model_name, weights_file)


__all__ = [
    "ULTRALYTICS_LICENSE_NOTICE",
    "UltralyticsDetector",
    "load_detector",
]
