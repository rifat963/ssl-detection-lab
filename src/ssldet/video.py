"""Detection and tracking analysis for local videos, URLs, and streams."""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .catalog import resolve_model_family
from .evaluation import _load_model, _plain


@dataclass
class VideoAnalysisConfig:
    video_source: str | int | Path
    model_name: str
    weights_file: str | Path
    output_dir: str | Path = "runs/ssldet/video"
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    max_detections: int = 300
    device: str | int | None = None
    tracker: str | None = "botsort.yaml"
    video_stride: int = 1
    max_frames: int | None = None
    save_annotated: bool = True
    save_txt: bool = False
    save_confidence: bool = False

    def validate(self) -> "VideoAnalysisConfig":
        source_missing = (
            self.video_source < 0
            if isinstance(self.video_source, int)
            else not str(self.video_source).strip()
        )
        if source_missing or not str(self.model_name).strip() or not str(self.weights_file).strip():
            raise ValueError("video_source, model_name, and weights_file are required")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.iou <= 1.0:
            raise ValueError("confidence and iou must be between 0 and 1")
        if self.image_size < 32 or self.max_detections < 1 or self.video_stride < 1:
            raise ValueError(
                "image_size >= 32, max_detections >= 1, and video_stride >= 1 required"
            )
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive when provided")
        family = resolve_model_family(self.model_name)
        if not family.video_analysis:
            raise ValueError(f"{family.name} is not supported for video analysis")
        return self


@dataclass(frozen=True)
class VideoAnalysisResult:
    output_dir: Path
    report_json: Path
    outcome_markdown: Path
    frames_csv: Path
    detections_csv: Path
    annotated_media: tuple[Path, ...]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> Path:
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "max", "mean", "median", "std", "p5", "p95", "p99")}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p5": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _list(value: Any) -> list:
    plain = _plain(value)
    if plain is None:
        return []
    if isinstance(plain, list):
        return plain
    return [plain]


def _probe_local_video(source: str | int | Path) -> dict[str, Any]:
    if isinstance(source, int):
        return {}
    path = Path(source)
    if not path.is_file():
        return {}
    try:
        import cv2
    except ImportError:
        return {}
    try:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return {}
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        return {
            "source_fps": fps if fps > 0 else None,
            "source_frames": frames if frames > 0 else None,
            "source_width": width if width > 0 else None,
            "source_height": height if height > 0 else None,
            "source_duration_seconds": frames / fps if fps > 0 and frames > 0 else None,
        }
    except (OSError, cv2.error):
        return {}


def _detection_container(result: Any):
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        return boxes
    return getattr(result, "obb", None)


def _make_outcome(report: dict[str, Any]) -> str:
    overall = report["video_metrics"]
    confidence = overall["confidence"]
    latency = overall["latency_ms"]
    lines = [
        "# Video analysis outcome",
        "",
        f"- Model: **{report['model']['name']}** (`{report['model']['weights']}`)",
        f"- Frames analysed: **{overall['frames_processed']}**",
        f"- Total detections: **{overall['total_detections']}**",
        f"- Frames with detections: **{overall['frames_with_detections']}** "
        f"({overall['detection_frame_coverage']:.2%})",
        (
            f"- Mean confidence: **{confidence['mean']:.4f}**"
            if confidence["mean"] is not None
            else "- Mean confidence: n/a"
        ),
        (
            f"- Mean inference latency: **{latency['mean']:.2f} ms**"
            if latency["mean"] is not None
            else "- Mean inference latency: n/a"
        ),
        f"- End-to-end processing throughput: **{overall['processing_fps']:.2f} FPS**",
        f"- Unique tracks: **{overall['tracking']['unique_tracks']}**",
        "",
        "## Per-class outcome",
        "",
        "| Class | Detections | Frames present | Unique tracks | Mean confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["per_class"]:
        mean_conf = item["confidence"]["mean"]
        lines.append(
            f"| {item['class_name']} | {item['detections']} | {item['frames_present']} | "
            f"{item['unique_tracks']} | {mean_conf:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an unlabelled-video analysis. Detection counts, confidence, timing, coverage, "
            "box occupancy, and track statistics are measurable here. Precision, recall, F1, AP, "
            "mAP, and confusion-matrix accuracy require ground-truth labels; run the labelled "
            "dataset evaluator for those metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_video(config: VideoAnalysisConfig) -> VideoAnalysisResult:
    """Run streaming inference/tracking and export frame-, object-, and video-level outcomes."""

    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(config.model_name, config.weights_file)
    source_metadata = _probe_local_video(config.video_source)
    kwargs: dict[str, Any] = {
        "source": config.video_source,
        "stream": True,
        "conf": config.confidence,
        "iou": config.iou,
        "imgsz": config.image_size,
        "max_det": config.max_detections,
        "vid_stride": config.video_stride,
        "save": config.save_annotated,
        "save_txt": config.save_txt,
        "save_conf": config.save_confidence,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
        "verbose": False,
    }
    if config.device is not None:
        kwargs["device"] = config.device

    if config.tracker:
        result_stream = model.track(tracker=config.tracker, persist=True, **kwargs)
    else:
        result_stream = model.predict(**kwargs)

    frame_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    confidence_values: list[float] = []
    box_area_ratios: list[float] = []
    inference_times: list[float] = []
    preprocess_times: list[float] = []
    postprocess_times: list[float] = []
    per_class_confidence: dict[int, list[float]] = defaultdict(list)
    per_class_frames: dict[int, set[int]] = defaultdict(set)
    per_class_tracks: dict[int, set[int]] = defaultdict(set)
    track_lengths: Counter[int] = Counter()
    class_counts: Counter[int] = Counter()
    names: dict[int, str] = {}
    task = None
    started = time.perf_counter()

    try:
        selected_results = (
            itertools.islice(result_stream, config.max_frames)
            if config.max_frames is not None
            else result_stream
        )
        for frame_index, result in enumerate(selected_results, start=1):
            result_names = dict(getattr(result, "names", {}))
            names.update({int(key): str(value) for key, value in result_names.items()})
            task = task or getattr(result, "task", None)
            if task in {"classify", "semantic"}:
                raise ValueError(
                    "Video detection analysis requires detect, segment, pose, or OBB weights; "
                    f"got {task}"
                )
            shape = tuple(getattr(result, "orig_shape", (0, 0)))
            height, width = (int(shape[0]), int(shape[1])) if len(shape) >= 2 else (0, 0)
            frame_area = max(1, width * height)
            speed = dict(getattr(result, "speed", {}) or {})
            preprocess = float(speed.get("preprocess") or 0.0)
            inference = float(speed.get("inference") or 0.0)
            postprocess = float(speed.get("postprocess") or 0.0)
            preprocess_times.append(preprocess)
            inference_times.append(inference)
            postprocess_times.append(postprocess)

            container = _detection_container(result)
            confidences = _list(getattr(container, "conf", None)) if container is not None else []
            class_ids = _list(getattr(container, "cls", None)) if container is not None else []
            track_ids = _list(getattr(container, "id", None)) if container is not None else []
            xyxy_rows = _list(getattr(container, "xyxy", None)) if container is not None else []
            if track_ids and len(track_ids) != len(confidences):
                track_ids = []

            pairs = zip(confidences, class_ids, strict=True)
            for detection_index, (confidence, class_id_value) in enumerate(pairs, start=1):
                class_id = int(class_id_value)
                confidence = float(confidence)
                coords = (
                    xyxy_rows[detection_index - 1]
                    if detection_index <= len(xyxy_rows)
                    else [0, 0, 0, 0]
                )
                x1, y1, x2, y2 = (float(value) for value in coords[:4])
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                area_ratio = (box_width * box_height) / frame_area
                track_id = int(track_ids[detection_index - 1]) if track_ids else None
                row = {
                    "frame": frame_index,
                    "time_seconds": (
                        (frame_index - 1) * config.video_stride / source_metadata["source_fps"]
                        if source_metadata.get("source_fps")
                        else ""
                    ),
                    "detection": detection_index,
                    "track_id": "" if track_id is None else track_id,
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": box_width,
                    "height": box_height,
                    "area_ratio": area_ratio,
                    "center_x_normalized": ((x1 + x2) / 2 / width) if width else "",
                    "center_y_normalized": ((y1 + y2) / 2 / height) if height else "",
                }
                detection_rows.append(row)
                confidence_values.append(confidence)
                box_area_ratios.append(area_ratio)
                class_counts[class_id] += 1
                per_class_confidence[class_id].append(confidence)
                per_class_frames[class_id].add(frame_index)
                if track_id is not None:
                    track_lengths[track_id] += 1
                    per_class_tracks[class_id].add(track_id)

            frame_rows.append(
                {
                    "frame": frame_index,
                    "time_seconds": (
                        (frame_index - 1) * config.video_stride / source_metadata["source_fps"]
                        if source_metadata.get("source_fps")
                        else ""
                    ),
                    "width": width,
                    "height": height,
                    "detections": len(confidences),
                    "mean_confidence": statistics.fmean(confidences) if confidences else "",
                    "preprocess_ms": preprocess,
                    "inference_ms": inference,
                    "postprocess_ms": postprocess,
                }
            )
    finally:
        close = getattr(result_stream, "close", None)
        if callable(close):
            close()

    elapsed = max(time.perf_counter() - started, 1e-12)
    frames_processed = len(frame_rows)
    if not frames_processed:
        raise RuntimeError("No video frames were produced; check the source link/path and codec")
    frames_with_detections = sum(row["detections"] > 0 for row in frame_rows)
    per_frame_counts = [int(row["detections"]) for row in frame_rows]
    track_length_values = [float(value) for value in track_lengths.values()]
    per_class = [
        {
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "detections": class_counts[class_id],
            "detection_share": class_counts[class_id] / max(1, len(detection_rows)),
            "frames_present": len(per_class_frames[class_id]),
            "frame_coverage": len(per_class_frames[class_id]) / frames_processed,
            "unique_tracks": len(per_class_tracks[class_id]),
            "confidence": _stats(per_class_confidence[class_id]),
        }
        for class_id in sorted(class_counts)
    ]
    report = {
        "schema_version": 1,
        "analysis_type": "unlabelled_video",
        "source": {"value": _plain(config.video_source), **source_metadata},
        "model": {
            "name": config.model_name,
            "family": resolve_model_family(config.model_name).name,
            "weights": config.weights_file,
            "task": task,
            "class_names": names,
        },
        "config": _plain(asdict(config)),
        "video_metrics": {
            "frames_processed": frames_processed,
            "sampled_video_seconds": (
                frames_processed * config.video_stride / source_metadata["source_fps"]
                if source_metadata.get("source_fps")
                else None
            ),
            "wall_time_seconds": elapsed,
            "processing_fps": frames_processed / elapsed,
            "total_detections": len(detection_rows),
            "frames_with_detections": frames_with_detections,
            "frames_without_detections": frames_processed - frames_with_detections,
            "detection_frame_coverage": frames_with_detections / frames_processed,
            "detections_per_frame": _stats([float(value) for value in per_frame_counts]),
            "confidence": _stats(confidence_values),
            "box_area_ratio": _stats(box_area_ratios),
            "preprocess_latency_ms": _stats(preprocess_times),
            "latency_ms": _stats(inference_times),
            "postprocess_latency_ms": _stats(postprocess_times),
            "model_pipeline_fps": (
                1000.0 / statistics.fmean(
                    [
                        a + b + c
                        for a, b, c in zip(
                            preprocess_times,
                            inference_times,
                            postprocess_times,
                            strict=True,
                        )
                    ]
                )
                if any(
                    a + b + c > 0
                    for a, b, c in zip(
                        preprocess_times,
                        inference_times,
                        postprocess_times,
                        strict=True,
                    )
                )
                else None
            ),
            "tracking": {
                "enabled": bool(config.tracker),
                "tracker": config.tracker,
                "unique_tracks": len(track_lengths),
                "track_length_sampled_frames": _stats(track_length_values),
            },
        },
        "per_class": per_class,
        "metric_boundary": {
            "available_without_labels": [
                "counts", "confidence distribution", "box occupancy", "latency", "throughput",
                "frame coverage", "class distribution", "unique tracks", "track length",
            ],
            "requires_ground_truth": [
                "precision", "recall", "F1", "AP", "mAP50", "mAP75", "mAP50-95",
                "confusion matrix", "MOTA", "MOTP", "IDF1", "HOTA",
            ],
        },
    }

    frames_csv = _write_csv(
        output_dir / "frames.csv",
        frame_rows,
        (
            "frame",
            "time_seconds",
            "width",
            "height",
            "detections",
            "mean_confidence",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
        ),
    )
    detections_csv = _write_csv(
        output_dir / "detections.csv",
        detection_rows,
        (
            "frame",
            "time_seconds",
            "detection",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "area_ratio",
            "center_x_normalized",
            "center_y_normalized",
        ),
    )
    report_json = output_dir / "video_analysis.json"
    report_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    outcome_markdown = output_dir / "outcome.md"
    outcome_markdown.write_text(_make_outcome(report), encoding="utf-8")
    media_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    annotated_media = tuple(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in media_extensions
    )
    return VideoAnalysisResult(
        output_dir, report_json, outcome_markdown, frames_csv, detections_csv, annotated_media
    )
