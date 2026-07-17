"""Backend-neutral object-detection interfaces."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ObjectDetector(Protocol):
    """Common operations used by evaluation and video-analysis modules."""

    def predict(self, **kwargs: Any) -> Any: ...

    def track(self, **kwargs: Any) -> Any: ...

    def validate(self, **kwargs: Any) -> Any: ...


__all__ = ["ObjectDetector"]
