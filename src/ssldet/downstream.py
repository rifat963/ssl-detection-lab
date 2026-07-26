"""Transfer reusable SSL encoder checkpoints into Ultralytics YOLO detectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BackboneTransferResult:
    ssl_checkpoint: Path
    detector_checkpoint: Path
    report_json: Path
    ssl_method: str
    source_model: str
    encoder_prefix: str
    loaded_keys: int
    total_backbone_keys: int
    coverage: float
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _checkpoint_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _encoder_prefix(state: Mapping[str, Any], requested: str | None) -> str:
    candidates = (requested,) if requested else ("online_encoder.", "student_encoder.")
    for candidate in candidates:
        if candidate and any(key.startswith(candidate) for key in state):
            return candidate
    available = sorted({key.split(".", 1)[0] for key in state})
    raise KeyError(
        "No transferable encoder was found. Expected online_encoder.* or "
        f"student_encoder.* keys; available roots: {available}"
    )


def transfer_ssl_backbone_to_yolo(
    ssl_checkpoint: str | Path,
    output_file: str | Path,
    *,
    yolo_model: str | None = None,
    encoder_prefix: str | None = None,
    minimum_coverage: float = 0.95,
    map_location: str = "cpu",
) -> BackboneTransferResult:
    """Create a detector checkpoint initialized from an SSL encoder.

    The SSL projection head, predictor, target network, optimizer, and scheduler are
    intentionally ignored. Only the trainable online/student encoder is transferred.
    """

    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")

    source = Path(ssl_checkpoint)
    destination = Path(output_file)
    if not source.is_file():
        raise FileNotFoundError(f"SSL checkpoint does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("output_file must differ from ssl_checkpoint")

    import torch

    from ultralytics import YOLO

    from .backbones import YOLOBackboneEncoder

    raw_checkpoint = torch.load(source, map_location=map_location, weights_only=True)
    checkpoint = _checkpoint_mapping(raw_checkpoint, "SSL checkpoint")
    method_state = _checkpoint_mapping(checkpoint.get("method", checkpoint), "method state")
    normalized_state = {
        key.removeprefix("module."): value for key, value in method_state.items()
    }
    config = _checkpoint_mapping(checkpoint.get("config", {}), "checkpoint config")
    selected_prefix = _encoder_prefix(normalized_state, encoder_prefix)
    backbone_state = {
        key.removeprefix(selected_prefix): value
        for key, value in normalized_state.items()
        if key.startswith(selected_prefix)
    }
    if not backbone_state:
        raise KeyError(f"No state entries matched encoder prefix {selected_prefix!r}")

    source_model = str(yolo_model or config.get("yolo_model") or "").strip()
    if not source_model:
        raise ValueError(
            "yolo_model is required when the SSL checkpoint does not contain config.yolo_model"
        )

    detector = YOLO(source_model)
    encoder = YOLOBackboneEncoder(detector.model)
    target_keys = set(encoder.state_dict())
    source_keys = set(backbone_state)
    loaded_keys = len(target_keys & source_keys)
    coverage = loaded_keys / max(1, len(target_keys))
    try:
        incompatible = encoder.load_state_dict(backbone_state, strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            f"SSL encoder tensor shapes do not match {source_model!r}. "
            "Use the same YOLO family and scale used during pretraining."
        ) from error
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"SSL-to-YOLO backbone coverage is {coverage:.1%}, below the required "
            f"{minimum_coverage:.1%}. Confirm that {source_model!r} matches pretraining."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(destination))
    report_path = destination.with_suffix(".transfer.json")
    payload = {
        "ssl_checkpoint": str(source),
        "detector_checkpoint": str(destination),
        "ssl_method": str(config.get("method", "unknown")),
        "source_model": source_model,
        "encoder_prefix": selected_prefix,
        "loaded_keys": loaded_keys,
        "total_backbone_keys": len(target_keys),
        "coverage": coverage,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "excluded_ssl_components": [
            "projection head",
            "predictor",
            "target or teacher network",
            "optimizer",
            "scheduler",
            "gradient scaler",
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BackboneTransferResult(
        source,
        destination,
        report_path,
        str(config.get("method", "unknown")),
        source_model,
        selected_prefix,
        loaded_keys,
        len(target_keys),
        coverage,
        missing_keys,
        unexpected_keys,
    )


__all__ = ["BackboneTransferResult", "transfer_ssl_backbone_to_yolo"]
