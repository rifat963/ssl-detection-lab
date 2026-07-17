"""Dataset evaluation with complete Ultralytics metric export."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .catalog import resolve_model_family


@dataclass
class EvaluationConfig:
    model_name: str
    weights_file: str
    data: str
    output_dir: str = "runs/ssldet/evaluation"
    split: str = "val"
    image_size: int = 640
    batch_size: int = 16
    confidence: float = 0.001
    iou: float = 0.7
    max_detections: int = 300
    device: str | int | None = None
    workers: int = 8
    half: bool = False
    plots: bool = True
    save_json: bool = True

    def validate(self) -> "EvaluationConfig":
        if not self.model_name.strip() or not self.weights_file.strip() or not self.data.strip():
            raise ValueError("model_name, weights_file, and data are required")
        if self.image_size < 32 or self.batch_size < 1 or self.max_detections < 1:
            raise ValueError("image_size >= 32, batch_size >= 1, and max_detections >= 1 required")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.iou <= 1.0:
            raise ValueError("confidence and iou must be between 0 and 1")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        return self


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics_json: Path
    summary_csv: Path
    per_class_csv: Path | None
    per_image_csv: Path | None
    confusion_matrix_csv: Path | None


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else None
    except (TypeError, ValueError):
        return str(value)


def _rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(({key: _plain(value) for key, value in row.items()} for row in rows))
    return path


def _load_model(model_name: str, weights_file: str):
    family = resolve_model_family(model_name)
    if family.name == "RT-DETR":
        from ultralytics import RTDETR

        return RTDETR(weights_file)
    if family.name == "YOLO-NAS":
        from ultralytics import NAS

        return NAS(weights_file)
    from ultralytics import YOLO

    return YOLO(weights_file)


def _component_metrics(metrics: Any, names: dict[int, str]) -> tuple[dict, list[dict], list[dict]]:
    components: dict[str, Any] = {}
    per_class: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for component_name in ("box", "seg", "pose", "obb", "probs"):
        component = getattr(metrics, component_name, None)
        if component is None:
            continue
        values = {}
        for key in ("mp", "mr", "map", "map50", "map75", "maps", "p", "r", "f1", "ap", "ap50"):
            if hasattr(component, key):
                values[key] = _plain(getattr(component, key))
        components[component_name] = values

        class_result = getattr(component, "class_result", None)
        ap_class_index = _plain(getattr(component, "ap_class_index", [])) or []
        if callable(class_result):
            for position, class_index in enumerate(ap_class_index):
                result = _plain(class_result(position))
                if isinstance(result, list) and len(result) >= 4:
                    precision, recall, ap50, ap50_95 = result[:4]
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if precision + recall
                        else 0.0
                    )
                    row = {
                        "task": component_name,
                        "class_id": int(class_index),
                        "class_name": names.get(int(class_index), str(class_index)),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "ap50": ap50,
                        "ap50_95": ap50_95,
                    }
                    per_class.append(row)
        image_metrics = _plain(getattr(component, "image_metrics", {}))
        if isinstance(image_metrics, dict):
            for image, row in image_metrics.items():
                per_image.append({"task": component_name, "image": str(image), **row})
    return components, per_class, per_image


def evaluate(config: EvaluationConfig) -> EvaluationResult:
    """Validate labelled data and export every metric exposed by Ultralytics.

    The raw ``results_dict``, task-specific aggregates, per-class metrics,
    per-image metrics, speed measurements, curves, and confusion matrix are retained.
    """

    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(config.model_name, config.weights_file)
    kwargs: dict[str, Any] = {
        "data": config.data,
        "split": config.split,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "conf": config.confidence,
        "iou": config.iou,
        "max_det": config.max_detections,
        "workers": config.workers,
        "half": config.half,
        "plots": config.plots,
        "save_json": config.save_json,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
    }
    if config.device is not None:
        kwargs["device"] = config.device
    metrics = model.val(**kwargs)
    names = {int(key): str(value) for key, value in dict(getattr(metrics, "names", {})).items()}
    components, per_class, per_image = _component_metrics(metrics, names)

    summary = _plain(metrics.summary()) if hasattr(metrics, "summary") else []
    if not isinstance(summary, list):
        summary = []
    results_dict = _plain(getattr(metrics, "results_dict", {}))
    confusion = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    confusion_values = _plain(confusion)
    fitness = getattr(metrics, "fitness", None)
    if callable(fitness):
        fitness = fitness()
    payload = {
        "schema_version": 1,
        "evaluation_type": "labelled_dataset",
        "config": asdict(config),
        "model_family": resolve_model_family(config.model_name).name,
        "task": getattr(metrics, "task", None),
        "class_names": names,
        "headline_metrics": results_dict,
        "task_metrics": components,
        "speed_ms_per_image": _plain(getattr(metrics, "speed", {})),
        "fitness": _plain(fitness),
        "per_class": per_class,
        "per_image": per_image,
        "summary": summary,
        "curves": _plain(getattr(metrics, "curves_results", [])),
        "confusion_matrix": confusion_values,
    }
    metrics_json = output_dir / "metrics.json"
    metrics_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary_csv = _rows_to_csv(output_dir / "summary.csv", summary or [results_dict])
    assert summary_csv is not None
    per_class_csv = _rows_to_csv(output_dir / "per_class_metrics.csv", per_class)
    per_image_csv = _rows_to_csv(output_dir / "per_image_metrics.csv", per_image)

    confusion_csv = None
    if isinstance(confusion_values, list) and confusion_values:
        labels = [names.get(index, "background") for index in range(len(confusion_values))]
        confusion_rows = []
        for index, values in enumerate(confusion_values):
            row = {"actual/predicted": labels[index]}
            row.update({label: value for label, value in zip(labels, values)})
            confusion_rows.append(row)
        confusion_csv = _rows_to_csv(output_dir / "confusion_matrix.csv", confusion_rows)

    return EvaluationResult(
        output_dir, metrics_json, summary_csv, per_class_csv, per_image_csv, confusion_csv
    )
